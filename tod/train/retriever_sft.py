"""Dual-encoder (bi-encoder) retriever -- the ALTERNATIVE stage-1 recall arm for the universal
picker's two-stage readout (``tod.eval.letter_logit --readout twostage --stage1 retriever``).

Stage-1 today is the pointwise 12B readout: one forward per candidate, order-invariant + IIA,
but ~1 forward/candidate and below the .95 recall gate on the long tail (tasksource recall@16
~.89). This module trains a cheap alternative: a bi-encoder that encodes the QUERY (state +
question) once and each CANDIDATE (label + description) once, and ranks candidates by
cosine similarity. Candidate vectors are cacheable across records, so at eval/serve time a
record with N candidates costs one query encode + (usually cached) candidate lookups instead
of N model forwards.

Design choices (justified):
  * Base ``answerdotai/ModernBERT-base`` (8192-token context) -- long states fit without a
    sliding window; MLM-pretrained encoder.
  * SHARED tower: the same encoder weights embed queries and candidates (symmetric retrieval,
    DPR/E5/GTE-style). Halves parameters vs. two towers and is the common default.
  * MEAN pooling (masked) over the last hidden state, not CLS. ModernBERT-base is MLM-only
    (no next-sentence / CLS objective), so CLS carries no special pretrained sentence meaning;
    masked mean aggregates all token evidence and is the standard choice for sentence
    embeddings / dense retrieval (Sentence-BERT, E5, GTE). Selectable via ``--pooling cls``.
  * L2-normalize + cosine similarity, scored at temperature ``--temperature`` (default 0.05).
  * InfoNCE: for a batch we pool ALL candidates across all rows. Each query's target is its own
    gold candidate; every other candidate in the pool is a negative -- so a row's OWN other
    options (hard negatives: same state/question, wrong answer) AND every other row's options
    (easy in-batch negatives) both contribute, exactly per the spec.

Query/candidate text reuse ``tod.eval.letter_logit`` verbatim (single source of truth with the
harness): the candidate text is ``pw_option_descs(rec)`` and the query text is
``pw_prefix_text(rec)`` MINUS its system prompt AND its trailing "Candidate option:" cue (that
cue is a pointwise lead-in to a single candidate; a bi-encoder query has no candidate spliced
in). Image rows render text-only with "[image]" placeholders for stage 1 (stage 2 has the
pixels). Long states are MIDDLE-truncated (head+tail kept) so state+question fit ``--max-tokens``;
the truncation is recorded.

Subcommands: ``prepare`` (render a raw pointwise jsonl to a compact, optionally per-dataset-capped
retriever jsonl), ``train`` (torchrun DDP InfoNCE), ``eval`` (per-set recall@k, k=1/8/16/24 and
by dataset, JSON in the harness's spirit).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from datetime import timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import tod.eval.letter_logit as ll
from tod.eval.letter_logit import _row_dataset, _state_text, load_records, pw_option_descs

CONFIG_NAME = "retriever_config.json"


# --------------------------------------------------------------------------- rendering
def _expected(rec: Dict[str, Any]) -> Optional[str]:
    """Gold label. Canonical train rows store it at ``target.expected``; eval/jevbench rows
    carry a top-level ``expected``. Prefer the top-level, fall back to target."""
    if rec.get("expected") is not None:
        return str(rec["expected"])
    tgt = rec.get("target")
    if isinstance(tgt, dict) and tgt.get("expected") is not None:
        return str(tgt["expected"])
    return None


def render_state_text(state: Any) -> str:
    """State as plain text for stage 1. A LIST state (image row) renders its text segments in
    order with a literal ``[image]`` at each image slot (no pixels for stage 1). A string / dict
    state uses the same ``_state_text`` the harness uses."""
    if not isinstance(state, list):
        return _state_text(state)
    buf: List[str] = []
    for seg in state:
        if isinstance(seg, dict) and "image" in seg:
            buf.append("[image]")
        elif isinstance(seg, dict):
            buf.append(str(seg.get("text", "")))
        else:
            buf.append(str(seg))
    return "".join(buf)


def render_query_parts(rec: Dict[str, Any]) -> Tuple[str, str]:
    """(state_text, question_text) -- the two halves of the query, kept apart so the STATE can be
    middle-truncated independently of the (short) question."""
    q = rec["question"]
    return render_state_text(rec["state"]).strip(), str(q["instructions"]).strip()


def render_row(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Compact retriever example for one raw record, or None if it has no usable gold-in-labels.
    ``pos`` is the index of the gold candidate in ``candidates`` (== label order)."""
    labels = [str(x) for x in (rec.get("labels") or [])]
    if len(labels) < 2:
        return None
    gold = _expected(rec)
    if gold is None or gold not in labels:
        return None
    _labels, descs = pw_option_descs(rec)
    state, question = render_query_parts(rec)
    return {"id": rec.get("id"), "dataset": _row_dataset(rec), "state": state,
            "question": question, "candidates": descs, "pos": labels.index(gold)}


def to_example(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Accept either a prepared retriever row (has state/candidates/pos) or a raw pointwise
    record (rendered on the fly). Returns a compact example or None if unusable."""
    if "candidates" in obj and "pos" in obj and "state" in obj:
        if len(obj["candidates"]) < 2 or not (0 <= int(obj["pos"]) < len(obj["candidates"])):
            return None
        return obj
    return render_row(obj)


# --------------------------------------------------------------------------- tokenization
def build_query_ids(tok, state: str, question: str, max_tokens: int,
                    query_prefix: str = "") -> Tuple[List[int], bool]:
    """Token ids for ``[prefix]state\\n\\nquestion`` with special tokens, MIDDLE-truncating the
    STATE so the whole thing fits ``max_tokens``. Returns (input_ids, truncated)."""
    sep = tok.encode("\n\n", add_special_tokens=False)
    pre = tok.encode(query_prefix, add_special_tokens=False) if query_prefix else []
    q_ids = tok.encode(question, add_special_tokens=False)
    s_ids = tok.encode(state, add_special_tokens=False)
    n_special = tok.num_special_tokens_to_add(pair=False)
    budget = max_tokens - n_special - len(pre) - len(sep) - len(q_ids)
    truncated = False
    if budget < 0:
        # pathological: question alone overflows; keep its tail and drop the state entirely.
        q_ids = q_ids[-max(1, max_tokens - n_special - len(pre)):]
        s_ids = []
        truncated = True
    elif len(s_ids) > budget:
        head = budget // 2
        tail = budget - head
        s_ids = s_ids[:head] + (s_ids[-tail:] if tail else [])
        truncated = True
    core = pre + s_ids + sep + q_ids
    return _wrap_special(tok, core), truncated


def _wrap_special(tok, core: List[int]) -> List[int]:
    """``build_inputs_with_special_tokens`` for a single sequence, robust to tokenizer backends
    that dropped that legacy method (e.g. transformers' Rust ``TokenizersBackend``). Falls back to
    wrapping with the tokenizer's own CLS/SEP (BOS/EOS) ids so the special-token count still matches
    ``num_special_tokens_to_add(pair=False)``."""
    fn = getattr(tok, "build_inputs_with_special_tokens", None)
    if fn is not None:
        try:
            return fn(core)
        except Exception:
            pass
    cls = tok.cls_token_id if tok.cls_token_id is not None else tok.bos_token_id
    sep = tok.sep_token_id if tok.sep_token_id is not None else tok.eos_token_id
    return ([cls] if cls is not None else []) + core + ([sep] if sep is not None else [])


def build_cand_ids(tok, text: str, max_tokens: int, cand_prefix: str = "") -> List[int]:
    t = (cand_prefix + text) if cand_prefix else text
    ids = tok.encode(t, add_special_tokens=True)
    if len(ids) > max_tokens:  # candidates are short; a hard tail cut is fine
        ids = ids[:max_tokens]
    return ids


def _pad(seqs: Sequence[Sequence[int]], pad_id: int):
    import torch
    m = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), m), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), m), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, : len(s)] = 1
    return ids, mask


# --------------------------------------------------------------------------- model
def _mean_pool(last_hidden, mask):
    import torch
    m = mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * m).sum(dim=1)
    counts = m.sum(dim=1).clamp(min=1e-6)
    return summed / counts


class BiEncoder:
    """Shared-tower dense encoder. Thin wrapper around a HF ``AutoModel`` backbone plus a pooling
    + optional L2-normalize head. Not an ``nn.Module`` subclass itself -- it just forwards to the
    backbone -- so ``save_pretrained`` / DDP wrap the backbone directly."""

    def __init__(self, backbone, tok, pooling: str = "mean", normalize: bool = True,
                 config: Optional[Dict[str, Any]] = None):
        self.backbone = backbone
        self.tok = tok
        self.pooling = pooling
        self.normalize = normalize
        self.config = config or {}

    @classmethod
    def from_base(cls, model_id: str, pooling: str = "mean", normalize: bool = True,
                  revision: Optional[str] = None, attn_implementation: str = "sdpa"):
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        # sdpa (not flash_attention_2) by default: we keep fp32 weights under bf16 autocast, and
        # ModernBERT's flash path expects half-precision weights -- sdpa is exact and dtype-agnostic.
        try:
            backbone = AutoModel.from_pretrained(model_id, revision=revision,
                                                 attn_implementation=attn_implementation)
        except (ValueError, TypeError):
            backbone = AutoModel.from_pretrained(model_id, revision=revision)
        return cls(backbone, tok, pooling, normalize, {"base_model": model_id})

    @classmethod
    def from_config(cls, config, tok, pooling: str = "mean", normalize: bool = True):
        """Tiny random encoder for CPU tests: build the backbone from a model config."""
        from transformers import AutoModel
        backbone = AutoModel.from_config(config)
        return cls(backbone, tok, pooling, normalize, {"base_model": "random-init"})

    @classmethod
    def load(cls, ckpt: str, device: str = "cpu"):
        from transformers import AutoModel, AutoTokenizer
        cfg: Dict[str, Any] = {}
        cpath = os.path.join(ckpt, CONFIG_NAME)
        if os.path.exists(cpath):
            with open(cpath, encoding="utf-8") as fh:
                cfg = json.load(fh)
        tok = AutoTokenizer.from_pretrained(ckpt)
        try:
            backbone = AutoModel.from_pretrained(ckpt, attn_implementation="sdpa")
        except (ValueError, TypeError):
            backbone = AutoModel.from_pretrained(ckpt)
        backbone = backbone.to(device).eval()
        return cls(backbone, tok, cfg.get("pooling", "mean"), cfg.get("normalize", True), cfg)

    def encode(self, input_ids, attention_mask):
        """Pooled (+ optionally normalized) embeddings for a padded batch."""
        import torch
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state
        pooled = h[:, 0] if self.pooling == "cls" else _mean_pool(h, attention_mask)
        if self.normalize:
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
        return pooled

    def save_pretrained(self, out: str, extra: Optional[Dict[str, Any]] = None):
        os.makedirs(out, exist_ok=True)
        self.backbone.save_pretrained(out, safe_serialization=True)
        self.tok.save_pretrained(out)
        cfg = dict(self.config)
        cfg.update({"pooling": self.pooling, "normalize": self.normalize})
        if extra:
            cfg.update(extra)
        with open(os.path.join(out, CONFIG_NAME), "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=1)


# --------------------------------------------------------------------------- loss / recall
def info_nce(q_emb, cand_emb, pos_global, temperature: float):
    """InfoNCE over the whole in-batch candidate pool.
      q_emb      [B, D]  query embeddings
      cand_emb   [T, D]  every candidate in the batch (all rows' options concatenated)
      pos_global [B]     for each query, the row index into ``cand_emb`` of its gold candidate
    Every candidate != the query's gold is a negative (its own other options + other rows').
    Returns (loss, logits[B,T])."""
    import torch
    logits = (q_emb @ cand_emb.t()) / temperature
    loss = torch.nn.functional.cross_entropy(logits, pos_global)
    return loss, logits


def inrow_recall1(logits, row_spans: Sequence[Tuple[int, int]], pos_local: Sequence[int]) -> float:
    """Fraction of rows whose gold outscores that row's OWN other candidates (recall@1 restricted
    to the row's own option set -- the picker's actual job). ``row_spans[i]=(start,end)`` slices
    row i's candidates out of the pooled ``logits`` columns; ``pos_local[i]`` is the gold's offset
    within that slice."""
    hits = 0
    for i, (s, e) in enumerate(row_spans):
        row = logits[i, s:e]
        if int(row.argmax().item()) == int(pos_local[i]):
            hits += 1
    return hits / max(1, len(row_spans))


def recall_at_k(scores: Sequence[float], gold_idx: int, ks: Sequence[int]) -> Dict[int, bool]:
    """For a single record: is the gold candidate within the top-k by score, for each k?"""
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    rank = order.index(gold_idx)
    return {int(k): rank < int(k) for k in ks}


# --------------------------------------------------------------------------- batching
class _Batch:
    __slots__ = ("q_ids", "q_mask", "c_ids", "c_mask", "row_spans", "pos_local", "pos_global",
                 "n_trunc")

    def __init__(self, q_ids, q_mask, c_ids, c_mask, row_spans, pos_local, pos_global, n_trunc):
        self.q_ids = q_ids; self.q_mask = q_mask
        self.c_ids = c_ids; self.c_mask = c_mask
        self.row_spans = row_spans; self.pos_local = pos_local
        self.pos_global = pos_global; self.n_trunc = n_trunc


def _tokenize_example(tok, ex, max_tokens, cand_max, query_prefix, cand_prefix):
    q_ids, trunc = build_query_ids(tok, ex["state"], ex["question"], max_tokens, query_prefix)
    c_ids = [build_cand_ids(tok, c, cand_max, cand_prefix) for c in ex["candidates"]]
    return {"q_ids": q_ids, "c_ids": c_ids, "pos": int(ex["pos"]), "trunc": trunc,
            "dataset": ex.get("dataset"), "id": ex.get("id")}


def make_batches(tokd: List[Dict[str, Any]], pad_id: int, token_budget: int,
                 max_rows: int, max_cands: int, reserve_hn: bool = False) -> List[Dict[str, Any]]:
    """Greedy token-budget batches. A batch closes when adding the next row would push the summed
    QUERY token length past ``token_budget``, or exceed ``max_rows`` rows, or the OWN-candidate cap.
    (Query length dominates the memory variance -- states run to 8192 tokens while candidates are
    short.) Returns lists of example dicts; ``collate`` pads them.

    ``reserve_hn`` (hard-negative training): halve the own-candidate cap so ``collate`` has room to
    append each row's mined hard negatives without the pooled candidate count blowing past
    ``max_cands`` (the total pool ceiling grad-cache chunks over)."""
    batches: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = []
    qtok = 0; ctot = 0
    own_cap = (max_cands // 2) if reserve_hn else max_cands
    # length-bucket: pad cost is (max_len * n_rows), not the raw sum, so packing a long row with
    # many short ones blows the padded tensor far past ``token_budget``. Sorting by query length
    # keeps each batch near-uniform, so padded size stays ~= budget. (Caller reshuffles batch order.)
    tokd = sorted(tokd, key=lambda e: len(e["q_ids"]), reverse=True)
    for ex in tokd:
        ql = len(ex["q_ids"]); cl = len(ex["c_ids"])
        if cur and (qtok + ql > token_budget or len(cur) + 1 > max_rows or ctot + cl > own_cap):
            batches.append(cur)
            cur = []; qtok = 0; ctot = 0
        cur.append(ex); qtok += ql; ctot += cl
    if cur:
        batches.append(cur)
    return batches


def collate(batch: List[Dict[str, Any]], pad_id: int, max_cands: int = 0) -> "_Batch":
    q_seqs = [b["q_ids"] for b in batch]
    c_seqs: List[List[int]] = []
    row_spans: List[Tuple[int, int]] = []
    pos_local: List[int] = []
    pos_global: List[int] = []
    n_trunc = 0
    for b in batch:
        start = len(c_seqs)
        c_seqs.extend(b["c_ids"])
        row_spans.append((start, len(c_seqs)))
        pos_local.append(b["pos"])
        pos_global.append(start + b["pos"])
        n_trunc += int(bool(b["trunc"]))
    # Hard negatives (ANCE-style): each row's mined top-k wrong candidates are appended to the pool
    # as extra negative columns AFTER every row's own block, so ``row_spans``/``pos_*`` (which index
    # the front own-blocks) and ``inrow_recall1`` are untouched. Dedup by token-id tuple so a mined
    # negative that is already someone's option (already "in the pool") is not duplicated -- a
    # duplicate identical column would be an ambiguous second gold for the row that owns it. Stop at
    # ``max_cands`` (0 = unbounded) so the pooled candidate count grad-cache chunks over is bounded.
    if any("hn_ids" in b for b in batch):
        seen = {tuple(s) for s in c_seqs}
        full = False
        for b in batch:
            if full:
                break
            for hn in b.get("hn_ids", ()):
                if max_cands and len(c_seqs) >= max_cands:
                    full = True
                    break
                t = tuple(hn)
                if t in seen:
                    continue
                seen.add(t)
                c_seqs.append(hn)
    q_ids, q_mask = _pad(q_seqs, pad_id)
    c_ids, c_mask = _pad(c_seqs, pad_id)
    return _Batch(q_ids, q_mask, c_ids, c_mask, row_spans, pos_local, pos_global, n_trunc)


# --------------------------------------------------------------------------- prepare
def prepare(args: argparse.Namespace) -> None:
    recs = load_records(args.jsonl)
    per_ds: Dict[str, int] = {}
    kept = 0; dropped = 0; trunc_hint = 0
    n_by_ds: Dict[str, int] = {}
    with open(args.out, "w", encoding="utf-8") as out:
        for r in recs:
            ex = render_row(r)
            if ex is None:
                dropped += 1
                continue
            ds = ex.get("dataset") or "?"
            if args.per_dataset_cap and per_ds.get(ds, 0) >= args.per_dataset_cap:
                continue
            per_ds[ds] = per_ds.get(ds, 0) + 1
            n_by_ds[ds] = n_by_ds.get(ds, 0) + 1
            kept += 1
            out.write(json.dumps(ex, ensure_ascii=False) + "\n")
    stats = {"in": len(recs), "kept": kept, "dropped_no_gold": dropped,
             "per_dataset_cap": args.per_dataset_cap, "by_dataset": n_by_ds, "out": args.out}
    print("[retriever_sft:prepare] " + json.dumps(stats), flush=True)
    if args.stats:
        with open(args.stats, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=1)


# --------------------------------------------------------------------------- gradient caching
def _rng_state(device):
    """Snapshot the RNG so a chunk's dropout mask can be reproduced in the recompute pass."""
    import torch
    st = {"cpu": torch.get_rng_state()}
    if device.type == "cuda":
        st["cuda"] = torch.cuda.get_rng_state(device)
    return st


def _set_rng_state(st, device):
    import torch
    torch.set_rng_state(st["cpu"])
    if device.type == "cuda" and "cuda" in st:
        torch.cuda.set_rng_state(st["cuda"], device)


def grad_cache_step(core, encode_fn, bt, device, temperature, chunk, ctx_factory,
                    distributed: bool = False, world: int = 1):
    """One memory-decoupled InfoNCE step (GradCache, Gao et al. 2021). The pooled candidate count
    no longer bounds peak memory -- only ``chunk`` rows' activations are ever resident -- so the
    pool can be 1024-2048 on a 40 GB card.

      1. Representation pass (no grad, chunked): embed every query and every pooled candidate.
         Each chunk's RNG state is captured so its dropout mask can be replayed. The concatenated
         embeddings are detached into leaves that require grad.
      2. Loss pass: InfoNCE on the cached embeddings; backprop to the embeddings only, giving the
         per-embedding gradients dL/dq [B,D] and dL/dc [T,D].
      3. Recompute pass (grad, chunked): restore each chunk's RNG state, re-encode it, and
         ``backward(chunk_emb, grad_tensors=dL/d chunk)``. By the chain rule the accumulated
         PARAMETER gradients are identical to a single full-batch backward (see grad_cache_selftest).

    Encoding always goes through the unwrapped ``core`` (never a DDP wrapper): DDP's per-backward
    all-reduce is incompatible with the many-backward recompute pass, so grads are averaged across
    ranks manually when ``distributed``. Returns ``(loss.detach(), logits.detach())``; parameter
    grads are left accumulated for the caller's optimizer step."""
    import torch
    q_ids = bt.q_ids.to(device); q_mask = bt.q_mask.to(device)
    c_ids = bt.c_ids.to(device); c_mask = bt.c_mask.to(device)
    pos_g = torch.tensor(bt.pos_global, dtype=torch.long, device=device)

    def _repr(ids, mask):
        embs = []; states = []
        for i in range(0, ids.shape[0], chunk):
            states.append(_rng_state(device))
            with torch.no_grad(), ctx_factory():
                embs.append(encode_fn(core, ids[i:i + chunk], mask[i:i + chunk]))
        return torch.cat(embs, dim=0), states

    q_emb, q_states = _repr(q_ids, q_mask)
    c_emb, c_states = _repr(c_ids, c_mask)
    q_emb = q_emb.detach().requires_grad_(True)
    c_emb = c_emb.detach().requires_grad_(True)
    loss, logits = info_nce(q_emb, c_emb, pos_g, temperature)
    loss.backward()
    q_grad = q_emb.grad.detach()
    c_grad = c_emb.grad.detach()

    def _recompute(ids, mask, cached_grad, states):
        for j, i in enumerate(range(0, ids.shape[0], chunk)):
            _set_rng_state(states[j], device)
            with ctx_factory():
                e = encode_fn(core, ids[i:i + chunk], mask[i:i + chunk])
            torch.autograd.backward(e, grad_tensors=cached_grad[i:i + e.shape[0]])

    _recompute(q_ids, q_mask, q_grad, q_states)
    _recompute(c_ids, c_mask, c_grad, c_states)

    if distributed and world > 1:
        import torch.distributed as dist
        for p in core.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad /= world
    return loss.detach(), logits.detach()


def grad_cache_selftest(tol: float = 1e-4, chunk: int = 2, seed: int = 0) -> Dict[str, Any]:
    """CPU check that GradCache parameter gradients equal a plain full-batch backward (allclose) on
    a tiny random encoder. eval() mode (dropout off) keeps both forwards deterministic."""
    import torch
    from transformers import BertConfig
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    cfg = BertConfig(vocab_size=60, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=64, max_position_embeddings=64)
    enc = BiEncoder.from_config(cfg, tok=None, pooling="mean", normalize=True)
    core = enc.backbone
    core.eval()

    def encode_fn(module, ids, mask):
        out = module(input_ids=ids, attention_mask=mask)
        return torch.nn.functional.normalize(_mean_pool(out.last_hidden_state, mask), p=2, dim=-1)

    B, C, vocab = 5, 4, 60
    q_seqs = [torch.randint(5, vocab, (7,), generator=gen).tolist() for _ in range(B)]
    c_seqs = [torch.randint(5, vocab, (5,), generator=gen).tolist() for _ in range(B * C)]
    q_ids, q_mask = _pad(q_seqs, 0)
    c_ids, c_mask = _pad(c_seqs, 0)
    row_spans = [(i * C, i * C + C) for i in range(B)]
    pos_local = [i % C for i in range(B)]
    pos_global = [row_spans[i][0] + pos_local[i] for i in range(B)]
    bt = _Batch(q_ids, q_mask, c_ids, c_mask, row_spans, pos_local, pos_global, 0)
    device = torch.device("cpu"); T = 0.05

    core.zero_grad(set_to_none=True)
    qe = encode_fn(core, q_ids, q_mask); ce = encode_fn(core, c_ids, c_mask)
    loss_plain, _ = info_nce(qe, ce, torch.tensor(pos_global, dtype=torch.long), T)
    loss_plain.backward()
    plain = {n: p.grad.detach().clone() for n, p in core.named_parameters() if p.grad is not None}

    core.zero_grad(set_to_none=True)
    loss_gc, _ = grad_cache_step(core, encode_fn, bt, device, T, chunk, _nullctx)
    gc = {n: p.grad.detach().clone() for n, p in core.named_parameters() if p.grad is not None}

    assert set(plain) == set(gc), "param set mismatch between plain and grad-cache"
    worst = 0.0; worst_name = None
    for n in plain:
        d = (plain[n] - gc[n]).abs().max().item()
        if d > worst:
            worst, worst_name = d, n
    loss_diff = abs(float(loss_plain.detach()) - float(loss_gc))
    ok = worst <= tol and loss_diff <= tol
    print(f"[retriever_sft:selftest] grad-cache vs plain: max|dgrad|={worst:.3e} @ {worst_name} "
          f"loss_diff={loss_diff:.3e} n_params={len(plain)} chunk={chunk} B={B} C={C} "
          f"-> {'PASS' if ok else 'FAIL'}", flush=True)
    return {"pass": bool(ok), "max_grad_diff": worst, "worst_param": worst_name,
            "loss_diff": loss_diff, "n_params": len(plain), "chunk": chunk, "tol": tol}


def gc_selftest_cli(args: argparse.Namespace) -> None:
    res = grad_cache_selftest(tol=args.tol, chunk=args.chunk, seed=args.seed)
    if not res["pass"]:
        raise SystemExit("grad-cache selftest FAILED")


# ---------------------------------------------------------------------- hard-negative mining
def mine_hard_negatives(prev_ckpt: str, exs: List[Dict[str, Any]], tokd: List[Dict[str, Any]],
                        tok, args: argparse.Namespace, device) -> Dict[str, int]:
    """ANCE-style one-shot mining. Under the PREVIOUS checkpoint's scores, find each row's top-k
    WRONG candidates from its dataset's full candidate corpus (the union of candidate texts across
    that dataset's rows -- training slates are sampled, so the hardest confusable labels are often
    off-slate) and stash their token ids on the row as ``hn_ids``. ``collate`` then guarantees they
    are pooled as negatives for that step. Hard negatives are kept within-dataset (a cross-dataset
    label is a trivial negative)."""
    import torch
    from collections import defaultdict
    enc = BiEncoder.load(prev_ckpt, device=str(device))
    enc.backbone.eval()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    cpre = enc.config.get("cand_prefix", args.cand_prefix)
    k = args.hard_neg_k

    ds_texts: Dict[str, List[str]] = defaultdict(list)
    ds_index: Dict[str, Dict[str, int]] = defaultdict(dict)
    for ex in exs:
        ds = ex.get("dataset") or "?"
        for c in ex["candidates"]:
            if c not in ds_index[ds]:
                ds_index[ds][c] = len(ds_texts[ds])
                ds_texts[ds].append(c)

    ds_emb: Dict[str, Any] = {}
    for ds, texts in ds_texts.items():
        ids_list = [build_cand_ids(tok, t, args.cand_max_tokens, cpre) for t in texts]
        ds_emb[ds] = _encode_texts(enc, ids_list, pad_id, device, 256)  # [Nds, D] cpu float

    rows_by_ds: Dict[str, List[int]] = defaultdict(list)
    for i, ex in enumerate(exs):
        rows_by_ds[ex.get("dataset") or "?"].append(i)

    stats = {"rows": 0, "mean_hn": 0.0}
    total_hn = 0
    for ds, idxs in rows_by_ds.items():
        emb = ds_emb[ds]
        if emb.shape[0] <= 1:
            for i in idxs:
                tokd[i]["hn_ids"] = []
            continue
        qemb = _encode_texts_budget(enc, [tokd[i]["q_ids"] for i in idxs], pad_id, device,
                                    args.token_budget)
        scores = qemb @ emb.t()                                   # [len, Nds]
        _v, topi = scores.topk(min(k + 1, emb.shape[0]), dim=1)   # +1 to drop the gold if it ranks
        for row_pos, i in enumerate(idxs):
            gold = exs[i]["candidates"][exs[i]["pos"]]
            gold_local = ds_index[ds].get(gold)
            picks = []
            for j in topi[row_pos].tolist():
                if j == gold_local:
                    continue
                picks.append(ds_texts[ds][j])
                if len(picks) >= k:
                    break
            tokd[i]["hn_ids"] = [build_cand_ids(tok, t, args.cand_max_tokens, cpre) for t in picks]
            total_hn += len(picks)
        stats["rows"] += len(idxs)
    stats["mean_hn"] = round(total_hn / max(1, stats["rows"]), 2)
    stats["datasets"] = len(ds_texts)
    stats["corpus_by_dataset"] = {ds: len(t) for ds, t in ds_texts.items()}
    del enc
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return stats


# --------------------------------------------------------------------------- train
def train(args: argparse.Namespace) -> None:
    import torch
    import torch.distributed as dist

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    distributed = local_rank >= 0
    if distributed:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group("nccl", timeout=timedelta(hours=args.dist_timeout_hours),
                                device_id=device)
        rank, world = dist.get_rank(), dist.get_world_size()
    else:
        device = torch.device(args.device)
        rank, world = 0, 1
    is_main = rank == 0
    use_bf16 = args.dtype == "bfloat16" and device.type == "cuda"

    enc = BiEncoder.from_base(args.model, pooling=args.pooling, normalize=not args.no_normalize,
                              revision=args.revision, attn_implementation=args.attn_implementation)
    if getattr(args, "grad_checkpointing", False):
        # trade compute for memory: recompute layer activations in backward instead of storing them.
        # essential for long (8k) query sequences on a 40 GB card. must disable use_cache.
        if hasattr(enc.backbone, "config"):
            enc.backbone.config.use_cache = False
        enc.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    enc.backbone.to(device)
    tok = enc.tok
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    # every rank tokenizes ALL examples identically (deterministic), then a per-epoch shuffled
    # batch list is sharded across ranks -- so each rank sees a disjoint slice of batches.
    raw = load_records(args.data)
    exs = [e for e in (to_example(o) for o in raw) if e is not None]
    tokd = [_tokenize_example(tok, e, args.max_tokens, args.cand_max_tokens,
                              args.query_prefix, args.cand_prefix) for e in exs]
    n_trunc = sum(int(t["trunc"]) for t in tokd)
    grad_cache = bool(getattr(args, "grad_cache", False))
    hard_neg = bool(getattr(args, "hard_neg_from", None))
    if is_main:
        print(f"[retriever_sft:train] model={args.model} rows={len(tokd)} "
              f"middle_truncated={n_trunc} world={world} pooling={args.pooling} "
              f"normalize={not args.no_normalize} T={args.temperature} bf16={use_bf16} "
              f"grad_cache={grad_cache} gc_chunk={getattr(args,'gc_chunk',None)} "
              f"hard_neg_from={args.hard_neg_from if hard_neg else None}", flush=True)

    hn_stats = None
    if hard_neg:
        hn_stats = mine_hard_negatives(args.hard_neg_from, exs, tokd, tok, args, device)
        if is_main:
            print("[retriever_sft:train] hard-neg mining " + json.dumps(hn_stats), flush=True)

    # GradCache runs its own chunked backward and (when distributed) averages grads by hand, so the
    # backbone must stay UNWRAPPED -- DDP's autograd hooks fire once per backward and would mis-count.
    if distributed and not grad_cache:
        enc.backbone = torch.nn.parallel.DistributedDataParallel(
            enc.backbone, device_ids=[local_rank], output_device=local_rank)
    core = enc.backbone.module if (distributed and not grad_cache) else enc.backbone
    opt = torch.optim.AdamW(core.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t_train0 = time.time()

    def encode(module, ids, mask):
        out = module(input_ids=ids, attention_mask=mask)
        h = out.last_hidden_state
        pooled = h[:, 0] if args.pooling == "cls" else _mean_pool(h, mask)
        return torch.nn.functional.normalize(pooled, p=2, dim=-1) if not args.no_normalize else pooled

    global_step = 0
    for epoch in range(args.epochs):
        rng = random.Random(args.seed + epoch)
        order = list(range(len(tokd)))
        rng.shuffle(order)
        shuffled = [tokd[i] for i in order]
        batches = make_batches(shuffled, pad_id, args.token_budget, args.max_rows, args.max_cands,
                                reserve_hn=hard_neg)
        rng.shuffle(batches)  # length-bucketed batches -> reshuffle order so training isn't long->short
        shard = batches[rank::world] if world > 1 else batches
        # DDP needs every rank to run the SAME number of all-reduces -> truncate to the global min.
        nloc = len(shard)
        if distributed:
            t = torch.tensor([nloc], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.MIN)
            nloc = int(t.item())
            shard = shard[:nloc]
        if is_main:
            print(f"[retriever_sft:train] epoch {epoch} batches/rank={nloc} "
                  f"(global={len(batches)})", flush=True)
        def _ctx():
            return torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else _nullctx()
        for b in shard:
            bt = collate(b, pad_id, max_cands=args.max_cands)
            opt.zero_grad(set_to_none=True)
            if grad_cache:
                loss, logits = grad_cache_step(core, encode, bt, device, args.temperature,
                                               args.gc_chunk, _ctx, distributed=distributed,
                                               world=world)
            else:
                q_ids = bt.q_ids.to(device); q_mask = bt.q_mask.to(device)
                c_ids = bt.c_ids.to(device); c_mask = bt.c_mask.to(device)
                pos_g = torch.tensor(bt.pos_global, dtype=torch.long, device=device)
                with _ctx():
                    q_emb = encode(enc.backbone, q_ids, q_mask)
                    c_emb = encode(enc.backbone, c_ids, c_mask)
                    loss, logits = info_nce(q_emb, c_emb, pos_g, args.temperature)
                loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(core.parameters(), args.grad_clip)
            opt.step()
            global_step += 1
            if is_main and (global_step % args.log_every == 0):
                r1 = inrow_recall1(logits.detach().float(), bt.row_spans, bt.pos_local)
                print("[retriever_sft:train] " + json.dumps(
                    {"epoch": epoch, "step": global_step, "rows": len(b),
                     "cands": int(bt.c_ids.shape[0]), "loss": round(float(loss.item()), 4),
                     "inrow_recall1": round(r1, 4)}), flush=True)
        if distributed:
            dist.barrier()

    wall_s = time.time() - t_train0
    peak_gib = (torch.cuda.max_memory_allocated(device) / 2**30) if device.type == "cuda" else 0.0
    if is_main:
        print(f"[retriever_sft:train] wall_time_s={wall_s:.1f} ({wall_s/60:.1f} min) "
              f"peak_mem_gib={peak_gib:.2f} steps={global_step}", flush=True)
        extra = {"trained_epochs": args.epochs, "lr": args.lr, "temperature": args.temperature,
                 "max_tokens": args.max_tokens, "cand_max_tokens": args.cand_max_tokens,
                 "query_prefix": args.query_prefix, "cand_prefix": args.cand_prefix,
                 "train_rows": len(tokd), "middle_truncated": n_trunc,
                 "grad_cache": grad_cache, "gc_chunk": args.gc_chunk, "max_cands": args.max_cands,
                 "hard_neg_from": args.hard_neg_from if hard_neg else None,
                 "hard_neg_k": args.hard_neg_k if hard_neg else None, "hard_neg_stats": hn_stats,
                 "wall_time_s": round(wall_s, 1), "peak_mem_gib": round(peak_gib, 2),
                 "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        enc.config["base_model"] = args.model
        saver = BiEncoder(core, tok, args.pooling, not args.no_normalize, enc.config)
        saver.save_pretrained(args.out, extra=extra)
        print(f"[retriever_sft:train] wrote {args.out}", flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


# --------------------------------------------------------------------------- eval
def _encode_texts(enc: BiEncoder, ids_list: List[List[int]], pad_id: int, device, batch: int):
    import torch
    embs = []
    for i in range(0, len(ids_list), batch):
        chunk = ids_list[i: i + batch]
        ids, mask = _pad(chunk, pad_id)
        with torch.no_grad():
            e = enc.encode(ids.to(device), mask.to(device))
        embs.append(e.float().cpu())
    return torch.cat(embs, dim=0) if embs else torch.zeros((0, 1))


def _encode_texts_budget(enc: BiEncoder, ids_list: List[List[int]], pad_id: int, device,
                         token_budget: int, max_batch: int = 256):
    """Length-bucketed no-grad encode returning embeddings in the INPUT order. Queries run to 8192
    tokens, so a fixed row-batch (128 x 8192) blows out attention memory; here each batch's
    padded-token count (max_len * n_rows) is capped at ``token_budget`` instead."""
    import torch
    if not ids_list:
        return torch.zeros((0, 1))
    order = sorted(range(len(ids_list)), key=lambda i: len(ids_list[i]), reverse=True)
    out: List[Any] = [None] * len(ids_list)
    i = 0
    while i < len(order):
        maxlen = len(ids_list[order[i]]); batch: List[int] = []
        j = i
        while j < len(order):
            ml = max(maxlen, len(ids_list[order[j]]))
            if batch and (ml * (len(batch) + 1) > token_budget or len(batch) >= max_batch):
                break
            batch.append(order[j]); maxlen = ml; j += 1
        ids, mask = _pad([ids_list[k] for k in batch], pad_id)
        with torch.no_grad():
            e = enc.encode(ids.to(device), mask.to(device)).float().cpu()
        for r, k in enumerate(batch):
            out[k] = e[r]
        i = j
    return torch.stack(out, dim=0)


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    import torch
    device = torch.device(args.device)
    enc = BiEncoder.load(args.ckpt, device=str(device))
    tok = enc.tok
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    ks = [int(x) for x in args.ks.split(",")]
    max_tokens = int(args.max_tokens or enc.config.get("max_tokens", 8192))
    qpre = enc.config.get("query_prefix", "")
    cpre = enc.config.get("cand_prefix", "")

    sets: Dict[str, Any] = {}
    all_hits = {k: 0 for k in ks}; all_n = 0
    dump_rows: List[Dict[str, Any]] = []
    for path in args.jsonl:
        recs = load_records(path)
        set_label = args.set_label or os.path.splitext(os.path.basename(path))[0]
        # candidate embedding cache keyed by candidate text (labels recur across records).
        cand_cache: Dict[str, int] = {}
        cand_ids: List[List[int]] = []
        rows: List[Dict[str, Any]] = []
        for r in recs:
            if (r.get("provenance") or {}).get("exclude_reason"):
                continue
            ex = render_row(r)
            if ex is None:
                continue
            idxs = []
            for c in ex["candidates"]:
                if c not in cand_cache:
                    cand_cache[c] = len(cand_ids)
                    cand_ids.append(build_cand_ids(tok, c, args.cand_max_tokens, cpre))
                idxs.append(cand_cache[c])
            q_ids, _tr = build_query_ids(tok, ex["state"], ex["question"], max_tokens, qpre)
            rows.append({"q_ids": q_ids, "cand_idx": idxs, "pos": ex["pos"],
                         "dataset": ex.get("dataset") or set_label,
                         "id": ex.get("id")})
        if not rows:
            continue
        cand_emb = _encode_texts(enc, cand_ids, pad_id, device, args.batch)
        q_emb = _encode_texts_budget(enc, [r["q_ids"] for r in rows], pad_id, device,
                                     args.token_budget)

        hits = {k: 0 for k in ks}
        ds_hits: Dict[str, Dict[int, int]] = {}; ds_n: Dict[str, int] = {}
        for i, r in enumerate(rows):
            ce = cand_emb[r["cand_idx"]]
            scores = (q_emb[i] @ ce.t()).tolist()
            if getattr(args, "dump_scores", None):
                dump_rows.append({"id": r["id"], "dataset": r["dataset"],
                                  "pos": r["pos"], "scores": scores})
            rk = recall_at_k(scores, r["pos"], ks)
            ds = r["dataset"]
            ds_n[ds] = ds_n.get(ds, 0) + 1
            ds_hits.setdefault(ds, {k: 0 for k in ks})
            for k in ks:
                if rk[k]:
                    hits[k] += 1; ds_hits[ds][k] += 1; all_hits[k] += 1
            all_n += 1
        n = len(rows)
        sets[set_label] = {
            "jsonl": path, "n": n,
            "recall_at_k": {str(k): hits[k] / n for k in ks},
            "recall_at_k_by_dataset": {
                ds: {"n": ds_n[ds], **{str(k): ds_hits[ds][k] / ds_n[ds] for k in ks}}
                for ds in sorted(ds_n)},
        }
        rkstr = " ".join(f"r@{k}={hits[k]/n:.4f}" for k in ks)
        print(f"[retriever_sft:eval] {set_label} n={n} {rkstr}", flush=True)

    result = {
        "ckpt": args.ckpt, "readout": "retriever.stage1.v1",
        "base_model": enc.config.get("base_model"), "pooling": enc.pooling,
        "normalize": enc.normalize, "temperature": enc.config.get("temperature"),
        "max_tokens": max_tokens, "ks": ks, "sets": sets,
        "overall": {"n": all_n, "recall_at_k": {str(k): (all_hits[k] / all_n if all_n else None)
                                                for k in ks}},
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if all_n:
        rkstr = " ".join(f"r@{k}={all_hits[k]/all_n:.4f}" for k in ks)
        print(f"[retriever_sft:eval] OVERALL n={all_n} {rkstr}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    print(f"[retriever_sft:eval] wrote {args.out}", flush=True)
    if getattr(args, "dump_scores", None):
        os.makedirs(os.path.dirname(os.path.abspath(args.dump_scores)), exist_ok=True)
        with open(args.dump_scores, "w", encoding="utf-8") as fh:
            for row in dump_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[retriever_sft:eval] dumped {len(dump_rows)} rows -> {args.dump_scores}",
              flush=True)
    return result


# --------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("tod.train.retriever_sft")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="render a raw pointwise jsonl to a compact retriever jsonl")
    p.add_argument("--jsonl", required=True, help="raw pointwise rows (e.g. train_v2_main.jsonl)")
    p.add_argument("--out", required=True)
    p.add_argument("--per-dataset-cap", type=int, default=0,
                   help="keep at most this many rows per source dataset (0 = no cap)")
    p.add_argument("--stats", default=None)
    p.set_defaults(func=prepare)

    t = sub.add_parser("train", help="bi-encoder InfoNCE SFT (torchrun-compatible)")
    t.add_argument("--model", default="answerdotai/ModernBERT-base")
    t.add_argument("--revision", default=None)
    t.add_argument("--data", required=True, help="prepared OR raw jsonl (auto-detected)")
    t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=int, default=2)
    t.add_argument("--lr", type=float, default=5e-5)
    t.add_argument("--weight-decay", type=float, default=0.01)
    t.add_argument("--temperature", type=float, default=0.05)
    t.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    t.add_argument("--no-normalize", action="store_true", help="dot product instead of cosine")
    t.add_argument("--max-tokens", type=int, default=8192)
    t.add_argument("--cand-max-tokens", type=int, default=256)
    t.add_argument("--token-budget", type=int, default=16384,
                   help="max summed QUERY tokens per batch (length-balanced batching)")
    t.add_argument("--max-rows", type=int, default=64, help="max queries per batch")
    t.add_argument("--max-cands", type=int, default=1024, help="max pooled candidates per batch")
    t.add_argument("--query-prefix", default="")
    t.add_argument("--cand-prefix", default="")
    t.add_argument("--grad-clip", type=float, default=1.0)
    t.add_argument("--grad-cache", action="store_true",
                   help="GradCache: chunked no-grad representation + chunked grad recompute, so the "
                        "pooled candidate count is decoupled from peak memory (1024-2048 on 40 GB)")
    t.add_argument("--gc-chunk", type=int, default=64,
                   help="rows per encode chunk under --grad-cache")
    t.add_argument("--hard-neg-from", default=None,
                   help="ANCE-style: mine each row's top-k wrong candidates under this checkpoint "
                        "(one pass before training) and guarantee they are pooled as negatives")
    t.add_argument("--hard-neg-k", type=int, default=32, help="hard negatives mined per row")
    t.add_argument("--grad-checkpointing", action="store_true",
                   help="recompute activations in backward to fit long sequences on small GPUs")
    t.add_argument("--dtype", default="bfloat16")
    t.add_argument("--attn-implementation", default="sdpa",
                   help="backbone attention kernel (sdpa is exact under bf16 autocast on fp32 weights)")
    t.add_argument("--device", default="cuda")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--log-every", type=int, default=20)
    t.add_argument("--dist-timeout-hours", type=float, default=4.0)
    t.set_defaults(func=train)

    e = sub.add_parser("eval", help="per-set recall@k (k=1/8/16/24) + by dataset -> JSON")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--jsonl", nargs="+", required=True, help="one or more eval jsonl files")
    e.add_argument("--out", required=True)
    e.add_argument("--set-label", default=None, help="override the per-file set name")
    e.add_argument("--ks", default="1,8,16,24")
    e.add_argument("--max-tokens", type=int, default=0, help="0 = use the checkpoint's config")
    e.add_argument("--cand-max-tokens", type=int, default=256)
    e.add_argument("--batch", type=int, default=256,
                   help="row-batch for CANDIDATE encoding (short); queries use --token-budget")
    e.add_argument("--token-budget", type=int, default=16384,
                   help="max padded QUERY tokens (max_len*n_rows) per encode batch")
    e.add_argument("--device", default="cuda")
    e.add_argument("--dump-scores", default=None,
                   help="also write per-row {id, dataset, pos, scores} JSONL to this path "
                        "(scores in candidate/label order, gold at pos)")
    e.set_defaults(func=evaluate)

    s = sub.add_parser("selftest", help="CPU check: grad-cache grads == plain full-batch grads")
    s.add_argument("--chunk", type=int, default=2)
    s.add_argument("--tol", type=float, default=1e-4)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=gc_selftest_cli)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()

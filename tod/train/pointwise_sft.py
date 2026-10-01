"""Pointwise-in-branch LoRA SFT -- trains THE readout of ``tod.eval.letter_logit --readout
pointwise`` (owner rule 2026-09-28: any option-capped readout is worthless).

The only readout is POINTWISE-IN-BRANCH: the prefix is state+question (NO option list); each
option is an independent branch ``description -> verdict slot``; at the slot the model's own
next-token head is read restricted to yes vs no; choice = softmax over the N branch log-odds / T;
noul = one branch each for yes/no; branches never see each other so N is bounded only by context.

This trainer targets THAT slot, so train and eval are one readout. Prompts are BYTE-IDENTICAL to
the harness: we import and call ``letter_logit``'s ``pw_option_descs`` / ``pw_prefix_text`` /
``pw_branch_text`` / ``PW_SYSTEM`` and reproduce the harness's ``_split_ids`` seam verbatim (a
newline ends the prefix, so BPE never merges across the seam; asserted per record).

Loss per record = (a) cross-branch softmax-CE over the N branch yes/no log-odds with the gold
option as target (this IS the choice/score readout), plus (b) per-branch BCE on each yes-vs-no
log-odd (gold -> yes, others -> no), weighted ``--bce-weight``. Soft targets from
``rec["target"]["probs"]`` become a soft-CE / soft-BCE when present. For very large N we sample
``--neg-per-pos`` negatives per record (gold ALWAYS included) and report the sampled softmax as an
approximation.

Two branch backends (``--branch-mode``):
  * ``flat`` (DEFAULT): pack ``[prefix + branch_i]`` as separate right-padded sequences under a
    token budget. Costs O(N.prefix) but is exact and gradient-checkpoint / DDP friendly.
  * ``cache``: run the prefix once with ``use_cache=True`` and continue every branch against the
    expanded KV-cache (the scorer's own mechanism, with gradients). Cheaper for big prefixes but
    text-only and incompatible with gradient checkpointing (pass ``--no-grad-checkpointing``).

Subcommands: ``prepare`` (curated jsonl; NO <=26-label filter, so high-N rows like banking77 are
INCLUDED -- they are the point; label-count histogram in the stats), ``train`` (torchrun DDP,
one process per GPU, hand-written loop), ``merge`` (merge_and_unload -> served dir). Checkpoints
are PEFT adapter dirs + ``sft_args.json``; the harness loads them via ``--adapter``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import Counter
from datetime import timedelta
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import tod.eval.letter_logit as ll
from tod.eval.letter_logit import (PW_SYSTEM, load_records, pw_branch_text, pw_option_descs,
                                   pw_prefix_text)

# The small letter_sft helpers are COPIED here verbatim (identical semantics) rather than imported
# so this module does not depend on the box's letter_sft.py version (which may lag the multimodal
# helpers). letter_logit's public + multimodal helpers ARE imported (stable, deployed alongside).
REASONING_FAMILIES = {
    "reasoning", "judge_hard", "multi_hop", "temporal_numeric", "trap", "reading_comprehension",
}
VISION_MARKERS = ("vision", "visual", "audio", "siglip", "multi_modal_projector")


def gold_index(rec: Dict[str, Any]) -> Optional[int]:
    exp = rec.get("expected")
    if exp is None:
        exp = (rec.get("target") or {}).get("expected")
    if exp is None:
        return None
    labels = [str(x) for x in rec.get("labels", [])]
    exp = str(exp)
    return labels.index(exp) if exp in labels else None


def _yaml_datasets(mix_path: str) -> Tuple[Dict[str, str], set]:
    """(allowed {converter->family}, excluded {converter}) from a train_release mix yaml."""
    import yaml
    with open(mix_path, "r", encoding="utf-8") as fh:
        d = yaml.safe_load(fh) or {}
    allowed: Dict[str, str] = {}
    for it in d.get("datasets", []) or []:
        name = it.get("converter") or it.get("name") or it.get("dataset")
        if name:
            allowed[name] = it.get("family", "")
    excluded = set()
    for it in d.get("excluded_datasets", []) or []:
        nm = it if isinstance(it, str) else (it.get("converter") or it.get("name"))
        if nm:
            excluded.add(nm)
    return allowed, excluded


def _is_image_row(rec: Dict[str, Any]) -> bool:
    """A row carries image content iff its state is a LIST of segments (canonical image rows)
    or a string that still holds ``<img_k>`` placeholders."""
    s = rec.get("state")
    return isinstance(s, list) or (isinstance(s, str) and "<img_" in s)


def _row_n_images(rec: Dict[str, Any]) -> int:
    m = rec.get("meta")
    if isinstance(m, dict) and m.get("n_images") is not None:
        return int(m["n_images"])
    imgs = rec.get("images")
    if isinstance(imgs, list):
        return len(imgs)
    s = rec.get("state")
    if isinstance(s, list):
        return sum(1 for seg in s if isinstance(seg, dict) and "image" in seg)
    if isinstance(s, str):
        import re
        return len(re.findall(r"<img_\d+>", s))
    return 0


def resolve_lora_targets(base, target_suffixes: List[str]) -> List[str]:
    """Full module names of nn.Linear layers under the TEXT model matching a target suffix,
    excluding any vision/audio tower."""
    import torch.nn as nn
    names: List[str] = []
    for name, mod in base.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if any(m in name for m in VISION_MARKERS):
            continue
        if name.split(".")[-1] in target_suffixes:
            names.append(name)
    return names


def forward_last(model, input_ids, mask, pos, autocast_dtype):
    """Sequences are LEFT-padded, so every row's verdict slot is the final column. We pass
    ``logits_to_keep=1`` so the model materialises only the last position's logits ([S, 1, V]
    instead of [S, L, V]) -- decisive for the 262k-vocab head over long (image) sequences."""
    import torch
    with torch.autocast("cuda", dtype=autocast_dtype, enabled=(input_ids.device.type == "cuda")):
        out = model(input_ids=input_ids, attention_mask=mask, position_ids=pos,
                    use_cache=False, logits_to_keep=1)
    return out.logits[:, -1, :].float()


def forward_last_mm(model, enc: Dict[str, Any], autocast_dtype):
    """Multimodal forward: LEFT-padded pixel_values + input_ids -> last-position logits only."""
    import torch
    dev = enc["input_ids"].device
    with torch.autocast("cuda", dtype=autocast_dtype, enabled=(dev.type == "cuda")):
        out = model(**enc, use_cache=False, logits_to_keep=1)
    return out.logits[:, -1, :].float()


def _lr_at(step: int, warmup: int, total: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))


def _log(rank: int, path: Optional[str], obj: Dict[str, Any]) -> None:
    if rank != 0:
        return
    print("[pointwise_sft] " + json.dumps(obj), flush=True)
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(obj) + "\n")


# --------------------------------------------------------------------------- yes/no + seam
def yes_no_ids(tok: Any) -> Tuple[List[int], List[int]]:
    """Single-token ids for yes/no (both cases, both leading-space variants) -- union matches
    ``PointwiseScorer._variants`` so the trained slot is read exactly as at eval."""
    def variants(w: str) -> List[int]:
        out = []
        for v in (w, " " + w):
            t = tok.encode(v, add_special_tokens=False)
            if len(t) == 1:
                out.append(t[0])
        return out
    yes = sorted(set(variants("yes") + variants("Yes")))
    no = sorted(set(variants("no") + variants("No")))
    return yes, no


def split_prefix_branch(tok: Any, has_chat: bool, prefix_text: str,
                        branch_text: str) -> Tuple[List[int], List[int]]:
    """Verbatim copy of ``PointwiseScorer._split_ids`` so training == eval tokenisation: the user
    turn is ``prefix_text + branch_text`` wrapped by the chat template; return
    ``(head_ids, tail_ids)`` where ``head_ids + tail_ids`` equals the full-turn tokenisation."""
    if has_chat:
        full = tok.apply_chat_template([{"role": "user", "content": prefix_text + branch_text}],
                                       add_generation_prompt=True, tokenize=False)
        head = tok.apply_chat_template([{"role": "user", "content": prefix_text}],
                                       add_generation_prompt=False, tokenize=False)
        while head and not full.startswith(head):
            head = head[:-1]
    else:
        full = prefix_text + branch_text + "\nAnswer:"
        head = prefix_text
    head_ids = tok.encode(head, add_special_tokens=not has_chat)
    full_ids = tok.encode(full, add_special_tokens=not has_chat)
    if full_ids[: len(head_ids)] != head_ids:
        k = 0
        while k < min(len(head_ids), len(full_ids)) and head_ids[k] == full_ids[k]:
            k += 1
        head_ids = full_ids[:k]
    return head_ids, full_ids[len(head_ids):]


def _lcp(seqs: List[List[int]]) -> List[int]:
    if not seqs:
        return []
    m = min(len(s) for s in seqs)
    k = 0
    while k < m and all(s[k] == seqs[0][k] for s in seqs):
        k += 1
    return seqs[0][:k]


def build_branches(tok: Any, has_chat: bool, rec: Dict[str, Any],
                   max_tokens: int) -> Tuple[List[str], List[int], List[List[int]], bool]:
    """(labels, shared_prefix_ids, per-branch tail ids, truncated). One branch per option; the
    shared prefix is the longest common token prefix of every branch's full-turn tokenisation, so
    ``prefix + branch_i`` equals scoring that branch alone (exactness, not hope). The prefix ends
    with a newline seam (asserted) so the concatenation equals the full-turn tokenisation."""
    labels, descs = pw_option_descs(rec)
    ptxt = pw_prefix_text(rec)
    assert ptxt.endswith("\n"), f"{rec.get('id')}: pointwise prefix must end with a newline seam"
    fulls: List[List[int]] = []
    for d in descs:
        head, tail = split_prefix_branch(tok, has_chat, ptxt, pw_branch_text(d))
        fulls.append(head + tail)                    # == full-turn tokenisation for this branch
    prefix = _lcp(fulls)
    assert prefix, f"{rec.get('id')}: empty shared prefix across branches (seam broke)"
    branches = [f[len(prefix):] for f in fulls]
    assert all(branches), f"{rec.get('id')}: a branch tail is empty after the seam"
    truncated = False
    longest = max(len(b) for b in branches)
    if len(prefix) + longest > max_tokens:
        keep = max_tokens - longest
        keep_tail = min(1024, max(0, keep) // 4)
        prefix = prefix[: keep - keep_tail] + prefix[-keep_tail:] if keep_tail else prefix[:keep]
        truncated = True
    return labels, prefix, branches, truncated


# --------------------------------------------------------------------------- image content
def pw_build_prefix_content(rec: Dict[str, Any],
                            canonical_root: Optional[str] = None) -> List[Dict[str, Any]]:
    """Ordered user-turn content parts for the pointwise PREFIX of an image row (state+question,
    no option list). Mirrors ``letter_logit.build_user_content`` structure with the pointwise
    head/tail so image-row prefixes render identically to ``pw_prefix_text`` on the text path."""
    state = rec.get("state")
    if not isinstance(state, list):
        return [{"type": "text", "text": pw_prefix_text(rec)}]
    q = rec["question"]
    head = PW_SYSTEM + "\n\n"
    tail = "\n\n" + str(q["instructions"]).strip() + "\n\nCandidate option:\n"
    dataset = ll._row_dataset(rec)
    parts: List[Dict[str, Any]] = []
    buf = head
    for seg in state:
        if isinstance(seg, dict) and "image" in seg:
            if buf:
                parts.append({"type": "text", "text": buf})
                buf = ""
            parts.append({"type": "image",
                          "path": ll._resolve_image_path(str(seg["image"]), canonical_root, dataset)})
        elif isinstance(seg, dict):
            buf += str(seg.get("text", ""))
        else:
            buf += str(seg)
    buf += tail
    if buf:
        parts.append({"type": "text", "text": buf})
    return parts


def pw_branch_content(prefix_parts: List[Dict[str, Any]], branch_text: str) -> List[Dict[str, Any]]:
    """Prefix content + this branch's text appended (merged into the trailing text part)."""
    parts = [dict(p) for p in prefix_parts]
    if parts and parts[-1]["type"] == "text":
        parts[-1] = {"type": "text", "text": parts[-1]["text"] + branch_text}
    else:
        parts.append({"type": "text", "text": branch_text})
    return parts


# --------------------------------------------------------------------------- example
def soft_probs(rec: Dict[str, Any], labels: List[str]) -> Optional[List[float]]:
    """Soft target distribution over the record's options from ``rec['target']['probs']`` (a
    label->prob map), aligned to ``labels`` order. ``None`` when absent/incomplete (hard gold)."""
    tgt = rec.get("target") or {}
    p = tgt.get("probs")
    if isinstance(p, dict):
        m = {str(k): float(v) for k, v in p.items()}
        if all(l in m for l in labels):
            return [m[l] for l in labels]
    return None


@dataclass
class Example:
    gold: int                          # gold option index (rec["expected"])
    labels: List[str]
    probs: Optional[List[float]] = None  # soft target over options (soft-CE/BCE) or None
    rid: Optional[str] = None
    truncated: bool = False
    # text rows:
    prefix: Optional[List[int]] = None       # shared prefix token ids
    branches: Optional[List[List[int]]] = None  # per-option tail token ids (verdict slot last)
    # image rows:
    content: Optional[List[Dict[str, Any]]] = None  # prefix content parts (text + image paths)
    branch_texts: Optional[List[str]] = None        # per-option branch text (appended at collate)

    @property
    def is_image(self) -> bool:
        return self.content is not None

    @property
    def n_options(self) -> int:
        return len(self.branch_texts) if self.is_image else len(self.branches)


def build_example(rec: Dict[str, Any], tok: Any, has_chat: bool, max_tokens: int, *,
                  processor: Any = None, canonical_root: Optional[str] = None) -> Optional[Example]:
    gold = gold_index(rec)
    if gold is None:
        return None
    labels = [str(x) for x in rec.get("labels", [])]
    if len(labels) < 2:                      # need >=2 options to form a cross-branch choice
        return None
    if _is_image_row(rec):
        if processor is None:
            return None                       # image row but the model has no processor -> skip
        _labels, descs = pw_option_descs(rec)
        return Example(gold=gold, labels=labels, probs=soft_probs(rec, labels),
                       rid=rec.get("id"), content=pw_build_prefix_content(rec, canonical_root),
                       branch_texts=[pw_branch_text(d) for d in descs])
    labels, prefix, branches, trunc = build_branches(tok, has_chat, rec, max_tokens)
    return Example(gold=gold, labels=labels, probs=soft_probs(rec, labels), rid=rec.get("id"),
                   truncated=trunc, prefix=prefix, branches=branches)


# --------------------------------------------------------------------------- prepare
def _keep_row(rec: Dict[str, Any], max_state_chars: int, include_images: bool = False) -> bool:
    if gold_index(rec) is None:
        return False
    labels = rec.get("labels") or []
    if len(labels) < 2:                      # NO <=26 cap: high-N rows (banking77) are kept
        return False
    if _is_image_row(rec) and not include_images:
        return False
    s = rec.get("state")
    st = s if isinstance(s, str) else json.dumps(s, ensure_ascii=False)
    return len(st) <= max_state_chars


_LEN_BUCKETS = [4096, 8192, 16384, 32768, 49152]


def _bucket(n: int) -> str:
    for b in _LEN_BUCKETS:
        if n <= b:
            return f"<={b}"
    return f">{_LEN_BUCKETS[-1]}"


def _prompt_token_len(rec: Dict[str, Any], tok: Any) -> int:
    """Full pointwise prompt length in tokens = prefix + the longest branch (chat-templated).
    Estimated as prefix tokens + a small branch allowance so we tokenise once per row, not N."""
    try:
        ptxt = pw_prefix_text(rec)
    except Exception:
        return 0
    prefix_n = len(tok.encode(ptxt, add_special_tokens=False))
    _labels, descs = pw_option_descs(rec)
    branch_n = max((len(tok.encode(pw_branch_text(d), add_special_tokens=False)) for d in descs),
                   default=0)
    return prefix_n + branch_n + 8            # +8 chat-template/seam slack


def prepare(args: argparse.Namespace) -> None:
    from tod.data.dataset import iter_canonical_rows
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tok_model)
    allowed, excluded = _yaml_datasets(args.mix)
    names = sorted(n for n in allowed if n not in excluded)
    rng = random.Random(args.seed)
    hist: Counter = Counter()          # label-count histogram
    lenhist: Counter = Counter()       # token-length bucket histogram
    stats: Dict[str, Any] = {"per_dataset": {}, "filtered": 0, "total": 0, "n_images": 0,
                             "dropped_over_max_tokens": 0, "max_tokens": args.max_tokens,
                             "max_prompt_tokens": 0, "seed": args.seed,
                             "per_dataset_cap": args.per_dataset, "reason_weight": args.reason_weight,
                             "include_images": bool(getattr(args, "include_images", False)),
                             "label_count_histogram": {}, "token_length_histogram": {}}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as out:
        for name in names:
            shard = os.path.join(args.canonical_root, name, f"{args.split}.jsonl.zst")
            if not os.path.exists(shard):
                shard = shard[:-4]            # allow uncompressed .jsonl
            if not os.path.exists(shard):
                continue
            fam = allowed[name]
            cap = int(args.per_dataset * (args.reason_weight if fam in REASONING_FAMILIES else 1.0))
            reservoir: List[Dict[str, Any]] = []
            seen = 0
            for rec in iter_canonical_rows(shard):
                # char filter is only a cheap pre-filter now (default ~200k = effectively none);
                # the real length gate is TOKENS <= max_tokens, applied on the sampled reservoir.
                if not _keep_row(rec, args.max_state_chars, getattr(args, "include_images", False)):
                    stats["filtered"] += 1
                    continue
                seen += 1
                if len(reservoir) < cap:
                    reservoir.append(rec)
                else:                          # reservoir sampling: uniform over the filtered stream
                    j = rng.randint(0, seen - 1)
                    if j < cap:
                        reservoir[j] = rec
            written = 0
            n_img = n_img_tot = max_n = dropped = 0
            for rec in reservoir:
                L = _prompt_token_len(rec, tok)
                if L > args.max_tokens:        # drop over-length rows (owner accepts dropping >max)
                    dropped += 1
                    continue
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
                hist[len(rec.get("labels") or [])] += 1
                lenhist[_bucket(L)] += 1
                stats["max_prompt_tokens"] = max(stats["max_prompt_tokens"], L)
                if _is_image_row(rec):
                    n_img += 1
                    n_img_tot += _row_n_images(rec)
                max_n = max(max_n, len(rec.get("labels") or []))
            stats["per_dataset"][name] = {"family": fam, "cap": cap, "written": written,
                                          "dropped_over_max_tokens": dropped, "image_rows": n_img,
                                          "n_images": n_img_tot, "max_labels": max_n}
            stats["total"] += written
            stats["n_images"] += n_img_tot
            stats["dropped_over_max_tokens"] += dropped
            print(f"[prepare] {name}: family={fam} cap={cap} written={written} dropped>{args.max_tokens}={dropped} "
                  f"image_rows={n_img} n_images={n_img_tot} max_labels={max_n}", flush=True)
    stats["label_count_histogram"] = {str(k): hist[k] for k in sorted(hist)}
    stats["token_length_histogram"] = {b: lenhist[b] for b in
                                       [f"<={x}" for x in _LEN_BUCKETS] + [f">{_LEN_BUCKETS[-1]}"]
                                       if lenhist[b]}
    with open(args.out + ".stats.json", "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=1)
    print(f"[prepare] wrote {stats['total']} rows to {args.out} "
          f"({len(stats['per_dataset'])} datasets); dropped>{args.max_tokens}tok="
          f"{stats['dropped_over_max_tokens']}; max_prompt_tokens={stats['max_prompt_tokens']}; "
          f"labels={stats['label_count_histogram']}; token_len={stats['token_length_histogram']}",
          flush=True)


# --------------------------------------------------------------------------- batching
def n_selected(ex: Example, neg: int) -> int:
    n = ex.n_options
    return n if (neg is None or neg <= 0 or n <= 1 + neg) else 1 + neg


def select_branches(ex: Example, neg: int, rng: random.Random) -> Tuple[List[int], int]:
    """(selected option indices, gold's position within the selection). Gold is ALWAYS present;
    for N > 1+neg we sample ``neg`` negatives and shuffle so gold's position is not fixed."""
    n = ex.n_options
    if neg is None or neg <= 0 or n <= 1 + neg:
        sel = list(range(n))
    else:
        others = [k for k in range(n) if k != ex.gold]
        sel = [ex.gold] + rng.sample(others, neg)
        rng.shuffle(sel)
    return sel, sel.index(ex.gold)


def _rec_seqlen(ex: Example) -> int:
    return len(ex.prefix) + max(len(b) for b in ex.branches)


def plan_text_batches(exs: List[Example], batch_tokens: int, max_branches: int,
                      neg: int) -> List[List[int]]:
    """Greedy length-sorted bucketing over RECORDS. A micro-batch holds whole records (the
    cross-branch CE needs all of a record's selected branches together); we cap total branches at
    ``max_branches`` and padded token area (``max_seqlen * total_branches``) at ``batch_tokens``."""
    order = sorted(range(len(exs)), key=lambda i: _rec_seqlen(exs[i]))
    batches: List[List[int]] = []
    cur: List[int] = []
    cur_nb = 0
    cur_max = 0
    for i in order:
        nb_i = n_selected(exs[i], neg)
        sl = _rec_seqlen(exs[i])
        nnb = cur_nb + nb_i
        nmax = max(cur_max, sl)
        if cur and (nnb > max_branches or nmax * nnb > batch_tokens):
            batches.append(cur)
            cur, cur_nb, cur_max = [], 0, 0
            nnb, nmax = nb_i, sl
        cur.append(i)
        cur_nb, cur_max = nnb, nmax
    if cur:
        batches.append(cur)
    return batches


def build_schedule(text_exs: List[Example], img_exs: List[Example], batch_tokens: int,
                   max_branches: int, neg: int, mm_batch_size: int) -> List[Tuple[str, List[int]]]:
    """One flat list of ``("text"|"image", record-index-list)`` micro-batch descriptors; text
    batches are token-budgeted, image batches are fixed-size. Both share ONE optimizer schedule
    so every DDP rank runs the same number of forward/backward calls."""
    sched: List[Tuple[str, List[int]]] = [("text", b) for b in
                                          plan_text_batches(text_exs, batch_tokens, max_branches, neg)]
    k = max(1, mm_batch_size)
    for i in range(0, len(img_exs), k):
        sched.append(("image", list(range(i, min(i + k, len(img_exs))))))
    return sched


# --------------------------------------------------------------------------- collate
def collate_text(exs: List[Example], idx: List[int], pad_id: int, device: Any, neg: int,
                 rng: random.Random):
    """Flat mode: each selected branch becomes its own LEFT-padded ``prefix + branch`` sequence
    (left padding puts every verdict slot in the final column, so ``logits_to_keep=1`` reads them
    all). Returns (input_ids, mask, position_ids, groups) where ``groups`` = per-record
    (sequence indices, gold position, soft probs over the selection)."""
    import torch
    seqs: List[List[int]] = []
    groups: List[Tuple[List[int], int, Optional[List[float]]]] = []
    for i in idx:
        ex = exs[i]
        sel, gold_pos = select_branches(ex, neg, rng)
        seq_idxs = []
        for k in sel:
            seq_idxs.append(len(seqs))
            seqs.append(ex.prefix + ex.branches[k])
        probs_sel = [ex.probs[k] for k in sel] if ex.probs is not None else None
        groups.append((seq_idxs, gold_pos, probs_sel))
    mlen = max(len(s) for s in seqs)
    input_ids = torch.full((len(seqs), mlen), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), mlen), dtype=torch.long)
    for b, s in enumerate(seqs):
        input_ids[b, mlen - len(s):] = torch.tensor(s, dtype=torch.long)   # LEFT pad
        mask[b, mlen - len(s):] = 1
    pos = (mask.long().cumsum(-1) - 1).clamp_min(0)                        # left-pad position ids
    return input_ids.to(device), mask.to(device), pos.to(device), groups


def pw_processor_inputs(processor: Any, contents: List[List[Dict[str, Any]]]):
    """Like ``letter_logit.build_processor_inputs`` but passes ``images`` NESTED (one sublist per
    text). A pointwise micro-batch has several branch sequences that each carry the SAME image(s),
    so the flat-images list that ``build_processor_inputs`` uses trips gemma4's per-text image
    validation ("images (1) vs text (N)"). Nested images map one image group to each text.
    Forces LEFT padding so every row's verdict slot is the final column (aligns with
    ``forward_last_mm``'s ``logits_to_keep=1``). Returns ``enc``."""
    from PIL import Image
    texts: List[str] = []
    images: List[List[Any]] = []
    for parts in contents:
        content: List[Dict[str, Any]] = []
        imgs: List[Any] = []
        for p in parts:
            if p["type"] == "text":
                content.append({"type": "text", "text": p["text"]})
            else:
                imgs.append(Image.open(p["path"]).convert("RGB"))
                content.append({"type": "image"})
        texts.append(processor.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False))
        images.append(imgs)
    tk = getattr(processor, "tokenizer", None)
    if tk is not None:
        tk.padding_side = "left"
    if any(images):
        enc = processor(text=texts, images=images, return_tensors="pt", padding=True)
    else:
        enc = processor(text=texts, return_tensors="pt", padding=True)
    return enc


def collate_image(exs: List[Example], idx: List[int], processor: Any, device: Any, neg: int,
                  rng: random.Random):
    """Image rows go through the native processor in flat mode: each selected branch is
    ``prefix content + branch text`` (images duplicated per branch). Returns (enc, groups)."""
    import torch
    contents: List[List[Dict[str, Any]]] = []
    groups: List[Tuple[List[int], int, Optional[List[float]]]] = []
    for i in idx:
        ex = exs[i]
        sel, gold_pos = select_branches(ex, neg, rng)
        seq_idxs = []
        for k in sel:
            seq_idxs.append(len(contents))
            contents.append(pw_branch_content(ex.content, ex.branch_texts[k]))
        probs_sel = [ex.probs[k] for k in sel] if ex.probs is not None else None
        groups.append((seq_idxs, gold_pos, probs_sel))
    enc = pw_processor_inputs(processor, contents)
    enc = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in enc.items()}
    return enc, groups


# --------------------------------------------------------------------------- log-odds + loss
def branch_log_odds(logits_last, yes_ids, no_ids):
    """[S, V] logits at each branch's verdict slot -> [S] yes-vs-no log-odds (logsumexp over the
    yes variants minus logsumexp over the no variants); identical to ``PointwiseScorer._logodds``."""
    import torch
    dev = logits_last.device
    yi = torch.as_tensor(yes_ids, device=dev, dtype=torch.long)
    ni = torch.as_tensor(no_ids, device=dev, dtype=torch.long)
    return (torch.logsumexp(logits_last.index_select(1, yi), 1)
            - torch.logsumexp(logits_last.index_select(1, ni), 1))


def compute_pw_loss(logodds, groups, bce_weight: float):
    """logodds: [S] per-branch log-odds. Returns (loss, metrics). Loss = cross-branch softmax-CE
    (choice/score readout) + ``bce_weight`` * per-branch yes/no BCE. Soft targets -> soft-CE/BCE."""
    import torch
    import torch.nn.functional as F
    dev = logodds.device
    ce_terms = []
    bce_logits = []
    bce_targets = []
    correct = 0
    yn_correct = 0
    yn_total = 0
    for seq_idxs, gold_pos, probs_sel in groups:
        sel = torch.as_tensor(seq_idxs, device=dev, dtype=torch.long)
        lo = logodds.index_select(0, sel)                     # [k]
        logp = F.log_softmax(lo, dim=0)
        if probs_sel is not None:
            p = torch.as_tensor(probs_sel, device=dev, dtype=lo.dtype)
            p = p / p.sum().clamp_min(1e-8)
            ce_terms.append(-(p * logp).sum())
            tgt = p
        else:
            ce_terms.append(-logp[gold_pos])
            tgt = torch.zeros(lo.shape[0], device=dev, dtype=lo.dtype)
            tgt[gold_pos] = 1.0
        correct += int(torch.argmax(lo).item() == gold_pos)
        bce_logits.append(lo)
        bce_targets.append(tgt)
        yn_correct += int(((lo > 0).to(tgt.dtype) == (tgt > 0.5).to(tgt.dtype)).sum().item())
        yn_total += lo.shape[0]
    ce_loss = torch.stack(ce_terms).mean()
    bce_loss = F.binary_cross_entropy_with_logits(torch.cat(bce_logits), torch.cat(bce_targets))
    loss = ce_loss + bce_weight * bce_loss
    metrics = {"cross_acc": correct / max(1, len(groups)), "yn_acc": yn_correct / max(1, yn_total),
               "ce": float(ce_loss.detach()), "bce": float(bce_loss.detach())}
    return loss, metrics


# --------------------------------------------------------------------------- cache backend
def _expand_cache(cache, B: int):
    """Repeat a batch-1 KV-cache to batch B while preserving the autograd graph (no deepcopy)."""
    if hasattr(cache, "batch_repeat_interleave"):
        cache.batch_repeat_interleave(B)
        return cache
    return tuple(tuple(t.expand(B, *t.shape[1:]) for t in layer) for layer in cache)


def forward_record_cache(model, ex: Example, sel: List[int], yes_ids, no_ids, pad_id: int,
                         device, autocast_dtype):
    """Cache mode (text only): run the prefix once (use_cache=True) then continue every selected
    branch against the expanded cache -- the scorer's exactness route, with gradients. Returns the
    [k] log-odds for the selection."""
    import torch
    with torch.autocast("cuda", dtype=autocast_dtype, enabled=(device.type == "cuda")):
        pre = model(input_ids=torch.tensor([ex.prefix], device=device), use_cache=True)
        cache = pre.past_key_values
        branches = [ex.branches[k] for k in sel]
        B = len(branches)
        L = max(len(b) for b in branches)
        P = len(ex.prefix)
        ids = torch.full((B, L), pad_id, dtype=torch.long)
        am = torch.zeros((B, P + L), dtype=torch.long)
        am[:, :P] = 1
        blen = []
        for j, b in enumerate(branches):
            ids[j, : len(b)] = torch.tensor(b, dtype=torch.long)
            am[j, P: P + len(b)] = 1
            blen.append(len(b))
        pos = torch.arange(P, P + L, device=device)[None].expand(B, L)
        o = model(input_ids=ids.to(device), attention_mask=am.to(device), position_ids=pos,
                  past_key_values=_expand_cache(cache, B), use_cache=True)
        lg = o.logits.float()
    yi = torch.as_tensor(yes_ids, device=device, dtype=torch.long)
    ni = torch.as_tensor(no_ids, device=device, dtype=torch.long)
    return torch.stack([torch.logsumexp(lg[j, blen[j] - 1].index_select(0, yi), 0)
                        - torch.logsumexp(lg[j, blen[j] - 1].index_select(0, ni), 0)
                        for j in range(B)])


def forward_batch_cache(model, exs: List[Example], idx: List[int], yes_ids, no_ids, pad_id: int,
                        device, autocast_dtype, neg: int, rng: random.Random):
    """Cache mode over a micro-batch: concatenated log-odds + per-record groups."""
    import torch
    lo_all = []
    groups = []
    n = 0
    for i in idx:
        ex = exs[i]
        sel, gold_pos = select_branches(ex, neg, rng)
        lo = forward_record_cache(model, ex, sel, yes_ids, no_ids, pad_id, device, autocast_dtype)
        groups.append((list(range(n, n + len(sel))), gold_pos,
                       [ex.probs[k] for k in sel] if ex.probs is not None else None))
        n += len(sel)
        lo_all.append(lo)
    return torch.cat(lo_all), groups


# --------------------------------------------------------------------------- solo (padding-free)
# PADDING-FREE forward: every branch is forwarded at batch 1 (a single sequence, no padding).
# Rationale: (1) sdpa is numerically wrong on this model even unpadded (measured up to 20 log-odds
# vs eager-fp32), so we use chunked_eager; (2) batching a record's differing-length branches would
# require padding, and while chunked_eager handles a padding mask correctly for text, the
# multimodal position path drifts under left-padding (~1.4 log-odds, measured) -- batch 1 removes
# padding entirely for BOTH modalities, so each verdict-slot log-odd equals its solo value (bf16
# vs fp32 noise only). A record's branches stay together for the cross-branch CE; the (few) LoRA
# grads are all-reduced by hand at each optimizer boundary, so variable per-rank forward counts
# are fine. chunked_eager keeps a long single sequence's attention at O(chunk*L), not O(L^2).
def forward_record_solo(model, ex: Example, sel: List[int], yes_ids, no_ids, processor,
                        device, autocast_dtype, branch_ckpt: bool = True):
    """Per-branch batch-1 forward for one record (text OR image). Returns ([k] log-odds, n_tok).

    BRANCH-LEVEL CHECKPOINTING (branch_ckpt, default on): each branch's prefix+branch -> scalar
    yes/no log-odds is computed under ``torch.utils.checkpoint.checkpoint(..., use_reentrant=False)``.
    Only the scalar log-odds is retained after the branch's forward; the whole branch graph is
    recomputed in backward. Without it, all N branch graphs stay alive until the cross-branch CE, so
    peak activation memory scales with N (a 100-label row keeps 100 graphs). With it, peak is ~one
    branch's activations. Non-reentrant checkpoint tracks grads to the (inner) LoRA params even
    though the branch inputs (int ids) do not require grad, so gradients still flow.
    """
    import torch
    from torch.utils.checkpoint import checkpoint
    los = []
    ntok = 0
    dev_cuda = (device.type == "cuda")
    use_ckpt = branch_ckpt and torch.is_grad_enabled()
    for k in sel:
        if ex.is_image:
            enc = pw_processor_inputs(processor, [pw_branch_content(ex.content, ex.branch_texts[k])])
            enc = {kk: (v.to(device) if torch.is_tensor(v) else v) for kk, v in enc.items()}
            ntok += int(enc["attention_mask"].sum().item())
            tkeys = [kk for kk, v in enc.items() if torch.is_tensor(v)]
            others = {kk: v for kk, v in enc.items() if not torch.is_tensor(v)}

            def _branch_fn(*tvals, _tkeys=tkeys, _others=others):
                d = dict(zip(_tkeys, tvals))
                d.update(_others)
                with torch.autocast("cuda", dtype=autocast_dtype, enabled=dev_cuda):
                    out = model(**d, use_cache=False, logits_to_keep=1)
                return branch_log_odds(out.logits[:, -1, :].float(), yes_ids, no_ids)[0]

            targs = tuple(enc[kk] for kk in tkeys)
        else:
            seq = ex.prefix + ex.branches[k]
            ii = torch.tensor([seq], device=device)
            mm = torch.ones_like(ii)
            pos = torch.arange(len(seq), device=device)[None]
            ntok += len(seq)

            def _branch_fn(ii, mm, pos):
                with torch.autocast("cuda", dtype=autocast_dtype, enabled=dev_cuda):
                    out = model(input_ids=ii, attention_mask=mm, position_ids=pos,
                                use_cache=False, logits_to_keep=1)
                return branch_log_odds(out.logits[:, -1, :].float(), yes_ids, no_ids)[0]

            targs = (ii, mm, pos)
        if use_ckpt:
            lo_k = checkpoint(_branch_fn, *targs, use_reentrant=False)
        else:
            lo_k = _branch_fn(*targs)
        los.append(lo_k)
    return torch.stack(los), ntok


def forward_batch_solo(model, exs: List[Example], idx: List[int], yes_ids, no_ids, processor,
                       device, autocast_dtype, neg: int, rng: random.Random,
                       branch_ckpt: bool = True):
    """A micro-batch of records: concatenated per-branch batch-1 log-odds + per-record groups +
    total token count. Each record's branches occupy a contiguous group for the cross-branch CE."""
    import torch
    lo_all = []
    groups = []
    n = 0
    ntok = 0
    for i in idx:
        ex = exs[i]
        sel, gold_pos = select_branches(ex, neg, rng)
        lo, t = forward_record_solo(model, ex, sel, yes_ids, no_ids, processor, device,
                                    autocast_dtype, branch_ckpt=branch_ckpt)
        groups.append((list(range(n, n + len(sel))), gold_pos,
                       [ex.probs[k] for k in sel] if ex.probs is not None else None))
        n += len(sel)
        ntok += t
        lo_all.append(lo)
    return torch.cat(lo_all), groups, ntok


def _one_branch_logodd(model, ex: Example, k: int, yes_ids, no_ids, processor, device,
                       autocast_dtype, text_model=None, ckpt_min_tokens: int = 0):
    """One branch's batch-1 forward -> (scalar yes/no log-odd with grad graph, n_tok).

    Conditional per-layer checkpointing: a text branch whose length is <= ``ckpt_min_tokens`` runs
    WITHOUT per-layer recompute (serial already bounds memory to one branch, and a <=14k branch
    fits an 80 GB H100 unchecked) -- roughly halving its backward cost; longer branches keep it."""
    import torch
    if ex.is_image:
        _set_layer_ckpt(text_model, True)        # image branches keep per-layer ckpt (long, safe)
        enc = pw_processor_inputs(processor, [pw_branch_content(ex.content, ex.branch_texts[k])])
        enc = {kk: (v.to(device) if torch.is_tensor(v) else v) for kk, v in enc.items()}
        ntok = int(enc["attention_mask"].sum().item())
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=(device.type == "cuda")):
            out = model(**enc, use_cache=False, logits_to_keep=1)
    else:
        seq = ex.prefix + ex.branches[k]
        ntok = len(seq)
        if text_model is not None:
            _set_layer_ckpt(text_model, ntok > ckpt_min_tokens)
        ii = torch.tensor([seq], device=device)
        mm = torch.ones_like(ii)
        pos = torch.arange(len(seq), device=device)[None]
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=(device.type == "cuda")):
            out = model(input_ids=ii, attention_mask=mm, position_ids=pos, use_cache=False,
                        logits_to_keep=1)
    return branch_log_odds(out.logits[:, -1, :].float(), yes_ids, no_ids)[0], ntok


def _head_logits(model, hidden_last):
    """Apply the LM head (+ final logit softcapping, if the config sets it) to gathered
    last-token hidden states ``[B, H]`` -> ``[B, V]`` float logits. Mirrors exactly what the
    model's own ``.logits`` path does, so a gathered last-real-token logit equals the value the
    solo (batch-1, last-column) forward reads."""
    import torch
    lm = model.get_output_embeddings()
    logits = lm(hidden_last.to(lm.weight.dtype))
    cfg = getattr(model, "config", None)
    sc = getattr(cfg, "final_logit_softcapping", None) if cfg is not None else None
    if sc:
        logits = torch.tanh(logits / sc) * sc
    return logits.float()


def forward_batch_text_padded(model, exs: List[Example], idx: List[int], yes_ids, no_ids,
                              pad_id: int, device, autocast_dtype, neg: int, rng: random.Random,
                              text_model=None, ckpt_min_tokens: int = 0):
    """BATCHED text path: every selected branch of every (text) record in the micro-batch becomes a
    RIGHT-padded ``prefix+branch`` row of ONE [B, Lmax] forward (chunked_eager).

    Right padding keeps each row's real tokens at positions ``0..len-1`` (natural causal order, no
    left-pad position drift), so we pass EXPLICIT ``position_ids`` (``arange(Lmax)`` per row; pad
    columns are attention-masked out and their positions never matter) and gather each row's LAST
    REAL token's hidden state (index ``len(seq)-1``, which differs per row under right padding).
    The LM head is then applied only to those [B, H] gathered vectors -> [B, V] logits (via
    ``_head_logits``), avoiding the [B, Lmax, V] full-sequence logit tensor over the 262k vocab.
    This replaces B batch-1 forwards with ONE. Numerical parity vs the solo per-branch path must
    be verified before this path is enabled (it is OFF by default). Returns (log-odds [B], groups,
    n_tok).

    Per-layer checkpointing is enabled only when the batch's summed real tokens exceed
    ``ckpt_min_tokens`` (small batches run unchecked -> faster; the schedule's --max-branch-tokens
    keeps the batch within the ckpt-on memory envelope)."""
    import torch
    seqs: List[List[int]] = []
    groups: List[Tuple[List[int], int, Optional[List[float]]]] = []
    n = 0
    for i in idx:
        ex = exs[i]
        sel, gold_pos = select_branches(ex, neg, rng)
        seq_idxs = []
        for k in sel:
            seq_idxs.append(len(seqs))
            seqs.append(ex.prefix + ex.branches[k])
        groups.append((seq_idxs, gold_pos,
                       [ex.probs[k] for k in sel] if ex.probs is not None else None))
        n += len(sel)
    ntok = sum(len(s) for s in seqs)
    mlen = max(len(s) for s in seqs)
    B = len(seqs)
    input_ids = torch.full((B, mlen), pad_id, dtype=torch.long)
    mask = torch.zeros((B, mlen), dtype=torch.long)
    last_idx = torch.empty(B, dtype=torch.long)
    for b, s in enumerate(seqs):
        input_ids[b, :len(s)] = torch.tensor(s, dtype=torch.long)          # RIGHT pad
        mask[b, :len(s)] = 1
        last_idx[b] = len(s) - 1                                           # last REAL token
    pos = torch.arange(mlen, dtype=torch.long)[None].expand(B, mlen)       # explicit position_ids
    if text_model is not None:
        _set_layer_ckpt(text_model, ntok > ckpt_min_tokens)
    with torch.autocast("cuda", dtype=autocast_dtype, enabled=(device.type == "cuda")):
        out = model(input_ids=input_ids.to(device), attention_mask=mask.to(device),
                    position_ids=pos.to(device), use_cache=False, logits_to_keep=1,
                    output_hidden_states=True)
    hidden = out.hidden_states[-1]                                         # [B, Lmax, H] (post-norm)
    last = last_idx.to(hidden.device)
    gathered = hidden[torch.arange(B, device=hidden.device), last]         # [B, H]
    lo = branch_log_odds(_head_logits(model, gathered), yes_ids, no_ids)   # [B]
    return lo, groups, ntok


def _plan_groups(exs, selections):
    groups = []
    n = 0
    for i, sel, gold_pos in selections:
        ex = exs[i]
        groups.append((list(range(n, n + len(sel))), gold_pos,
                       [ex.probs[k] for k in sel] if ex.probs is not None else None))
        n += len(sel)
    return groups


def train_batch_serial(model, exs: List[Example], idx: List[int], yes_ids, no_ids, processor,
                       device, autocast_dtype, neg: int, rng: random.Random, bce_weight: float,
                       loss_scale: float = 1.0, text_model=None, ckpt_min_tokens: int = 0):
    """SERIAL branch backward with manual gradient seeding -- the memory-bounded training path.

    The cross-branch softmax-CE couples all of a record's branches, so if every branch's autograd
    graph is materialised together (whether kept from forward, or recomputed by branch-level
    checkpointing) peak memory scales with the micro-batch's TOTAL branch tokens (measured on
    gemma-4-12B: ~24 GB + 1.28 MB/token; a 3x14k-token record OOMs an 80 GB H100). Instead:

      1. forward every branch ONCE under no_grad -> scalar log-odds (cheap; peak = one branch);
      2. compute the exact loss and its gradient d(loss)/d(log-odd) for every branch on those
         scalars (a tiny autograd over [total_branches] numbers);
      3. re-forward each branch ONE AT A TIME with grad and call ``lo_k.backward(seed_k)``, which
         accumulates the exact parameter gradient (chain rule) and frees the branch graph before
         the next -- so peak = a SINGLE branch regardless of option count or micro-batch size.

    The accumulated ``p.grad`` equals what ``compute_pw_loss(...).backward()`` would produce
    (verified against the coupled path on CPU). ``loss_scale`` folds in 1/grad_accum. Returns
    (detached loss, metrics, n_tok)."""
    import torch
    selections = []
    lo_list = []
    ntok = 0
    for i in idx:
        ex = exs[i]
        sel, gold_pos = select_branches(ex, neg, rng)
        selections.append((i, sel, gold_pos))
        los = []
        with torch.no_grad():
            for k in sel:
                lo_k, t = _one_branch_logodd(model, ex, k, yes_ids, no_ids, processor, device,
                                             autocast_dtype, text_model, ckpt_min_tokens)
                los.append(lo_k.detach())
                ntok += t
        lo_list.append(torch.stack(los))
    groups = _plan_groups(exs, selections)
    flat = torch.cat(lo_list).detach().requires_grad_(True)
    loss, metrics = compute_pw_loss(flat, groups, bce_weight)
    (seed,) = torch.autograd.grad(loss, flat)          # [total_branches] exact seeds
    seed = seed * loss_scale
    p = 0
    for i, sel, gold_pos in selections:
        ex = exs[i]
        for k in sel:
            lo_k, _ = _one_branch_logodd(model, ex, k, yes_ids, no_ids, processor, device,
                                         autocast_dtype, text_model, ckpt_min_tokens)
            lo_k.backward(seed[p])
            p += 1
    return loss.detach(), metrics, ntok


def plan_record_batches(exs: List[Example], max_branches: int, neg: int,
                        max_branch_tokens: int = 0) -> List[List[int]]:
    """Group record indices into micro-batches so the total selected branches per micro-batch is
    <= max_branches AND (if max_branch_tokens > 0) the summed branch tokens are <= max_branch_tokens.
    Length-sorted so a micro-batch's records have similar cost. The token cap matters for LONG rows:
    branch-level checkpointing recomputes each branch in backward, but the cross-branch softmax-CE
    needs every branch's grad, so recomputed branch graphs COEXIST during backward -- peak scales
    with the batch's total branch tokens, not with a single branch. The token cap bounds that peak
    so a micro-batch of several long rows cannot OOM (a single record over the cap goes alone)."""
    order = sorted(range(len(exs)), key=lambda i: _rec_maxlen(exs[i]))
    batches: List[List[int]] = []
    cur: List[int] = []
    cnt = 0
    ctok = 0
    for i in order:
        nb = n_selected(exs[i], neg)
        nt = nb * _rec_maxlen(exs[i])
        over_branches = cnt + nb > max_branches
        over_tokens = max_branch_tokens and ctok + nt > max_branch_tokens
        if cur and (over_branches or over_tokens):
            batches.append(cur)
            cur, cnt, ctok = [], 0, 0
        cur.append(i)
        cnt += nb
        ctok += nt
    if cur:
        batches.append(cur)
    return batches


def _rec_maxlen(ex: Example) -> int:
    if ex.is_image:
        return 10 ** 6 + ex.n_options          # image rows sort last (heaviest per branch)
    return len(ex.prefix) + max(len(b) for b in ex.branches)


# --------------------------------------------------------------------- per-window cost balancing
# Measured serial-branch wall-cost model (gemma-4-12B, chunked_eager, one branch's forward+backward
# at prefix+branch length L): c(L) = 5.92e-8 * L^2 + 4.42e-4 * L seconds. The L^2 term dominates for
# long rows, so a plain token count (linear) badly underestimates a 32k-token record -- which is why
# balancing on summed tokens still let one rank hold a 32k x 7-branch record (~9 min/micro-batch)
# while the others idled at the all-reduce barrier. We balance on THIS estimate instead.
_PW_COST_A = 5.92e-8          # quadratic (attention) coefficient
_PW_COST_B = 4.42e-4          # linear coefficient
_IMG_COST_L = 2048            # nominal per-branch token proxy for image rows (they run per-branch
#                               regardless; the live imbalance is text, so a rough proxy suffices)


def _branch_cost_s(L: int) -> float:
    """Estimated seconds for one branch's serial forward+backward at prefix+branch length ``L``."""
    return _PW_COST_A * L * L + _PW_COST_B * L


def _rec_cost_s(ex: Example, neg: int) -> float:
    """Estimated serial cost (seconds) of one record = Σ over its selected branches of the branch
    cost at the record's longest prefix+branch length (matches how the padded batch is planned)."""
    L = _IMG_COST_L if ex.is_image else (len(ex.prefix) + max(len(b) for b in ex.branches))
    return n_selected(ex, neg) * _branch_cost_s(L)


def _batch_cost_s(bidx: List[int], exs: List[Example], neg: int) -> float:
    """Estimated serial cost (seconds) of a whole micro-batch = Σ of its records' costs."""
    return sum(_rec_cost_s(exs[i], neg) for i in bidx)


def _lpt_assign(costs: List[float], capacities: List[int]) -> List[List[int]]:
    """Longest-Processing-Time greedy bin packing: assign each item (index into ``costs``, in
    descending-cost order) to the currently least-loaded bin that still has spare capacity. Returns
    ``bin -> [item indices]``. Deterministic (heap ties break on bin index). ``capacities[b]`` caps
    bin ``b``'s item COUNT; ``sum(capacities)`` must be >= ``len(costs)``. LPT gives a near-optimal
    makespan, so all bins end within a tight band -> per-(rank,window) costs are balanced."""
    import heapq
    nbin = len(capacities)
    assign: List[List[int]] = [[] for _ in range(nbin)]
    load = [0.0] * nbin
    cnt = [0] * nbin
    heap = [(0.0, b) for b in range(nbin) if capacities[b] > 0]
    heapq.heapify(heap)
    for i in sorted(range(len(costs)), key=lambda i: (-costs[i], i)):
        while True:
            _l, b = heapq.heappop(heap)          # least-loaded non-full bin (heap holds only these)
            if cnt[b] < capacities[b]:
                break
        assign[b].append(i)
        load[b] += costs[i]
        cnt[b] += 1
        if cnt[b] < capacities[b]:
            heapq.heappush(heap, (load[b], b))   # full bins are dropped, never re-pushed
    return assign


# A micro-step is one unit of work every rank performs before the next; grad_accum micro-steps make
# one WINDOW (optimizer step / all-reduce). Two kinds, both identical on every rank:
#   ("solo",  [idxlist_rank0, ..., idxlist_rank{W-1}])  each rank runs its own micro-batch (may be [])
#   ("shared", record_index)                            all ranks split ONE heavy record's branches
def build_window_schedule(exs: List[Example], neg: int, max_branches: int, max_branch_tokens: int,
                          *, world: int, grad_accum: int, seed: int, epoch: int,
                          max_record_cost_s: float, enable_split: bool
                          ) -> Tuple[List[Tuple], int]:
    """Cost-balance the epoch PER WINDOW (not just per epoch). Returns ``(steps, n_windows)`` where
    ``steps`` is the flat micro-step list (identical on every rank; boundary every ``grad_accum``
    steps). Very heavy single records (cost > ``max_record_cost_s``) become ``shared`` steps whose
    branches are split across the ranks of the window; the rest are packed into ``solo`` micro-batches
    and LPT-assigned to (window, rank) slots so every optimizer step's per-rank cost is balanced and
    heavy work is spread across steps rather than stacked. Every rank runs ``grad_accum`` micro-steps
    per window (short solo slots are padded empty)."""
    from tod.data import sharding
    world = max(1, int(world))
    ga = max(1, int(grad_accum))
    split = bool(enable_split) and world > 1

    # 1. Pull heavy single records out to be branch-split; plan the REST into solo micro-batches.
    heavy = [i for i in range(len(exs))
             if split and _rec_cost_s(exs[i], neg) > max_record_cost_s]
    hset = set(heavy)
    normal = [i for i in range(len(exs)) if i not in hset]
    sub = [exs[i] for i in normal]
    solo = [[normal[p] for p in b]                      # map sub-positions back to exs indices
            for b in plan_record_batches(sub, max_branches, neg, max_branch_tokens)]
    costs = [_batch_cost_s(b, exs, neg) for b in solo]

    h = len(heavy)
    n_solo_steps = (len(solo) + world - 1) // world     # each solo step serves up to `world` batches
    total_steps = h + n_solo_steps
    if total_steps == 0:
        return [], 0
    n_windows = (total_steps + ga - 1) // ga

    # 2. Spread the shared (heavy) steps across windows; the remaining slots per window are solo.
    shared_per_win = [0] * n_windows
    for j in range(h):
        shared_per_win[j % n_windows] += 1
    solo_per_win = [ga - shared_per_win[w] for w in range(n_windows)]   # >= 0; sums to >= n_solo_steps

    # 3. LPT the solo micro-batches into (window, rank) bins (count-capacity = solo_per_win[w]).
    caps = [solo_per_win[w] for w in range(n_windows) for _r in range(world)]
    assign = _lpt_assign(costs, caps)                   # bin (w*world + r) -> [solo positions]

    # 4. Emit micro-steps window by window.
    heavy_by_win: List[List[int]] = [[] for _ in range(n_windows)]
    for j, rec_i in enumerate(heavy):
        heavy_by_win[j % n_windows].append(rec_i)
    steps: List[Tuple] = []
    for w in range(n_windows):
        per_rank = [assign[w * world + r] for r in range(world)]
        for t in range(solo_per_win[w]):                # one solo micro-step per solo slot
            slot = [list(solo[per_rank[r][t]]) if t < len(per_rank[r]) else [] for r in range(world)]
            steps.append(("solo", slot))
        for rec_i in heavy_by_win[w]:                   # shared micro-steps (all ranks together)
            steps.append(("shared", rec_i))

    # 5. Shuffle WINDOW order (deterministic, cross-rank-identical) so cost-banding is not a
    #    curriculum; intra-window order is left as-is (summed before the barrier, so it is free).
    chunks = [steps[i * ga:(i + 1) * ga] for i in range(n_windows)]
    random.Random(sharding._seed_int(seed, epoch, "pw-windows")).shuffle(chunks)
    steps = [st for chunk in chunks for st in chunk]
    return steps, n_windows


# ---------------------------------------------------------------- branch-split (heavy record) path
def _all_gather_logodds(local, world: int, dist):
    """All-gather the per-rank branch log-odds and sum them into the full ``[k]`` vector on every
    rank. Each rank's ``local`` is a ``[k]`` tensor with ONLY its owned branch positions populated
    (zeros elsewhere); since every position is owned by exactly one rank, the elementwise sum selects
    each owner's value. One ``all_gather`` of ``k`` floats -- no other cross-rank communication."""
    import torch
    out = [torch.zeros_like(local) for _ in range(world)]
    dist.all_gather(out, local)
    return torch.stack(out, 0).sum(0)


def train_shared_record(model, ex: Example, yes_ids, no_ids, processor, device, autocast_dtype,
                        bce_weight: float, *, world: int, rank: int, gather_fn, sel: List[int],
                        gold_pos: int, probs_sel: Optional[List[float]] = None,
                        loss_scale: float = 1.0, text_model=None, ckpt_min_tokens: int = 0):
    """Branch-split training of ONE heavy record across the ``world`` ranks of a window.

    Each rank OWNS the selected branch positions ``p`` with ``p % world == rank`` (a deterministic
    partition all ranks agree on). It (1) forwards its owned branches no-grad -> scalar log-odds;
    (2) ``gather_fn`` all-gathers them into the full ``[k]`` log-odds (identical on every rank);
    (3) computes the exact cross-branch softmax-CE + BCE and its per-branch seeds d(loss)/d(logodd)
    (identical on every rank); (4) re-forwards ONLY its owned branches with grad and seeds each
    backward. Summing the ranks' parameter grads (the window all-reduce) reproduces the whole
    record's gradient, so this equals the single-rank serial path. Returns (detached loss, metrics,
    n_tok) -- ``n_tok`` counts only this rank's branches."""
    import torch
    k = len(sel)
    owned = [p for p in range(k) if p % world == rank]
    local = torch.zeros(k, device=device)
    ntok = 0
    with torch.no_grad():
        for p in owned:
            lo, t = _one_branch_logodd(model, ex, sel[p], yes_ids, no_ids, processor, device,
                                       autocast_dtype, text_model, ckpt_min_tokens)
            local[p] = lo.detach()
            ntok += t
    full = gather_fn(local).detach().requires_grad_(True)
    groups = [(list(range(k)), gold_pos, probs_sel)]
    loss, metrics = compute_pw_loss(full, groups, bce_weight)
    (seed,) = torch.autograd.grad(loss, full)          # [k] exact seeds (same on every rank)
    seed = seed * loss_scale
    for p in owned:                                    # backprop ONLY this rank's branches
        lo, _ = _one_branch_logodd(model, ex, sel[p], yes_ids, no_ids, processor, device,
                                   autocast_dtype, text_model, ckpt_min_tokens)
        lo.backward(seed[p])
    return loss.detach(), metrics, ntok


# --------------------------------------------------------------------------- block checkpointing
def _find_text_decoder(model):
    """Locate the text decoder stack (the module whose ``.layers`` is the ModuleList of decoder
    layers, with ``embed_tokens``/``rotary_emb``/``norm`` and a ``layer_types`` config). Works
    whether the model is text-only (CausalLM) or multimodal (the text decoder nested under a
    vision-language wrapper)."""
    import torch.nn as nn
    for m in model.modules():
        layers = getattr(m, "layers", None)
        cfg = getattr(m, "config", None)
        if (isinstance(layers, nn.ModuleList) and cfg is not None
                and hasattr(cfg, "layer_types") and hasattr(m, "embed_tokens")
                and hasattr(m, "rotary_emb") and hasattr(m, "norm")):
            return m
    return None


def _set_layer_ckpt(text_model, on: bool) -> None:
    """Flip HF per-layer gradient checkpointing on/off for every decoder layer (fast: a bool per
    layer). ``model.gradient_checkpointing_enable`` must have been called once first so each layer
    has its ``_gradient_checkpointing_func`` -- this only toggles the per-layer switch so we can
    checkpoint LONG branches (needed to fit) while running SHORT branches without recompute."""
    if text_model is None:
        return
    text_model.gradient_checkpointing = on
    for lyr in text_model.layers:
        if hasattr(lyr, "gradient_checkpointing"):
            lyr.gradient_checkpointing = on


def install_block_checkpointing(model, block_size: int) -> int:
    """Replace HF's per-layer gradient checkpointing with block-of-``block_size`` non-reentrant
    checkpoint regions on the text decoder. HF checkpoints EACH decoder layer, so it saves one
    residual-stream tensor per layer (48 x [1, L, hidden] at 48.7k tokens ~= 18 GB bf16). Grouping
    G consecutive layers into one checkpoint region saves only ceil(L_layers/G) residual tensors
    (block-of-6 over 48 layers -> 8 instead of 48). The chunked_eager per-chunk checkpoint inside
    each layer is untouched, so the attention matrix stays O(chunk*L) during recompute.

    Requires no cross-layer KV sharing (``first_kv_shared_layer_idx is None``) so a block boundary
    carries no state other than the residual stream. Returns the number of blocks installed.
    """
    import sys
    import types
    import torch
    from torch.utils.checkpoint import checkpoint

    tm = _find_text_decoder(model)
    if tm is None:
        raise SystemExit("block-ckpt: could not locate the text decoder (.layers stack)")
    cfg = tm.config
    if getattr(cfg, "first_kv_shared_layer_idx", None) is not None:
        raise SystemExit("block-ckpt: model uses cross-layer KV sharing "
                         "(first_kv_shared_layer_idx set); block boundaries would break it")

    # Turn OFF HF's per-layer checkpointing so layers do not self-checkpoint (no nested recompute).
    tm.gradient_checkpointing = False
    for lyr in tm.layers:
        if hasattr(lyr, "gradient_checkpointing"):
            lyr.gradient_checkpointing = False

    mod = sys.modules[type(tm).__module__]
    DynamicCache = getattr(mod, "DynamicCache")
    create_causal_mask = getattr(mod, "create_causal_mask")
    create_sliding_window_causal_mask = getattr(mod, "create_sliding_window_causal_mask")
    UserDict = getattr(mod, "UserDict")
    OutCls = getattr(mod, "Gemma4UnifiedTextModelOutputWithPast")

    n_layers = cfg.num_hidden_layers
    n_blocks = (n_layers + block_size - 1) // block_size

    def block_forward(self, input_ids=None, attention_mask=None, position_ids=None,
                      past_key_values=None, inputs_embeds=None, use_cache=None, **kwargs):
        # Faithful copy of Gemma4UnifiedTextModel.forward (transformers 5.14.1) with the per-layer
        # loop replaced by block-of-block_size non-reentrant checkpoint regions. Everything else --
        # embeddings, mask setup, per-layer-type rotary, final norm, output -- is byte-for-byte the
        # original path (verified numerically against the stock forward before launch).
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if use_cache is None:
            use_cache = False
        if input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if position_ids is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen
            position_ids = position_ids.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {"config": self.config, "inputs_embeds": inputs_embeds,
                           "attention_mask": attention_mask, "past_key_values": past_key_values,
                           "position_ids": position_ids}
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds
        position_embeddings = {}
        for layer_type in self.unique_layer_types:
            position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type)

        shared_kv_states = kwargs.pop("shared_kv_states", UserDict())
        layer_types = self.config.layer_types

        def run_span(start, end, hs):
            for i in range(start, end):
                hs = self.layers[i](
                    hs,
                    shared_kv_states=shared_kv_states,
                    position_embeddings=position_embeddings[layer_types[i]],
                    attention_mask=causal_mask_mapping[layer_types[i]],
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    **kwargs,
                )
            return hs

        do_ckpt = self.training and torch.is_grad_enabled()
        for b in range(n_blocks):
            start = b * block_size
            end = min(start + block_size, n_layers)
            if do_ckpt:
                hidden_states = checkpoint(
                    (lambda hs, s=start, e=end: run_span(s, e, hs)),
                    hidden_states, use_reentrant=False)
            else:
                hidden_states = run_span(start, end, hidden_states)

        hidden_states = self.norm(hidden_states)
        return OutCls(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            shared_kv_states=shared_kv_states if kwargs.get("return_shared_kv_states", False) else None,
        )

    tm.forward = types.MethodType(block_forward, tm)
    tm._block_ckpt_size = block_size
    return n_blocks


# --------------------------------------------------------------------------- lora
def _save(core, args: argparse.Namespace, out: str) -> None:
    os.makedirs(out, exist_ok=True)
    core.save_pretrained(out)
    with open(os.path.join(out, "sft_args.json"), "w", encoding="utf-8") as fh:
        json.dump({k: v for k, v in vars(args).items() if k != "func"}, fh, indent=1)


def evaluate_val(model, exs: List[Example], pad_id: int, device, args, yes_ids, no_ids,
                 autocast_dtype, processor: Any = None) -> Dict[str, float]:
    import torch
    model.eval()
    rng = random.Random(1234)                 # fixed selection for a stable val number
    tot_loss = tot_cross = tot_yn = 0.0
    nb = 0
    with torch.no_grad():
        for idx in plan_record_batches(exs, args.max_branches_per_step, args.neg_per_pos,
                                       args.max_branch_tokens):
            lo, groups, _ = forward_batch_solo(model, exs, idx, yes_ids, no_ids, processor,
                                               device, autocast_dtype, args.neg_per_pos, rng)
            loss, m = compute_pw_loss(lo, groups, args.bce_weight)
            tot_loss += float(loss)
            tot_cross += m["cross_acc"]
            tot_yn += m["yn_acc"]
            nb += 1
    model.train()
    d = max(1, nb)
    return {"val_loss": tot_loss / d, "val_cross_acc": tot_cross / d, "val_yn_acc": tot_yn / d}


# --------------------------------------------------------------------------- train
def train(args: argparse.Namespace) -> None:
    import torch
    import torch.distributed as dist
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.branch_mode == "cache" and args.grad_checkpointing:
        raise SystemExit("--branch-mode cache needs use_cache; pass --no-grad-checkpointing "
                         "(cache mode is text-only and best on a single GPU; flat is the default)")

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    distributed = local_rank >= 0
    if distributed:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        # LONG process-group timeout + explicit device_id. A single long-row micro-batch can run
        # several hundred seconds; with grad-accum the per-rank work in an accumulation window can
        # differ across ranks by more than NCCL's default 600 s watchdog, so a rank that reaches the
        # optimizer-boundary all-reduce first would abort waiting for a straggler. We raise the
        # collective timeout well above the heaviest step and bind the PG to this rank's device.
        dist.init_process_group("nccl", timeout=timedelta(hours=args.dist_timeout_hours),
                                device_id=device)
        rank, world = dist.get_rank(), dist.get_world_size()
    else:
        device = torch.device(args.device)
        rank, world = 0, 1
    autocast_dtype = torch.bfloat16

    tok = AutoTokenizer.from_pretrained(args.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    has_chat = tok.chat_template is not None
    yes_ids, no_ids = yes_no_ids(tok)
    assert yes_ids and no_ids, "yes/no are not single tokens for this tokenizer"

    processor = None
    try:
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(args.model)
        if getattr(proc, "image_processor", None) is not None:
            processor = proc
    except Exception:
        processor = None

    def load_split(path: Optional[str], shard: bool) -> List[Example]:
        if not path:
            return []
        recs = load_records(path)
        if shard and world > 1:
            recs = recs[rank::world]
        out: List[Example] = []
        for r in recs:
            ex = build_example(r, tok, has_chat, args.max_tokens, processor=processor,
                               canonical_root=args.canonical_root)
            if ex is not None:
                out.append(ex)
        return out

    # GLOBAL build (shard=False): every rank builds ALL examples so it knows the full, identical
    # micro-batch list -- the length-balanced round-robin schedule (below) then deals disjoint,
    # cost-matched batch slices to each rank. (Records are NOT re-sharded at load; the schedule is
    # what partitions the epoch's work across ranks.)
    train_exs = load_split(args.train_jsonl, shard=False)
    val_exs = load_split(args.val_jsonl, shard=False) if rank == 0 else []
    n_img_tr = sum(1 for e in train_exs if e.is_image)
    print(f"[pointwise_sft] rank{rank}: {len(train_exs)} train records "
          f"({n_img_tr} image, processor={'yes' if processor else 'none'}); "
          f"branch_mode={args.branch_mode} neg_per_pos={args.neg_per_pos}", flush=True)

    if args.attn == "chunked_eager":
        from tod.model import attn_chunked  # noqa: F401  (registers the attn implementation)
        attn_chunked.register()
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=autocast_dtype,
                                                attn_implementation=args.attn)
    base.config.use_cache = False
    text_model = None
    if args.grad_checkpointing:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base.enable_input_require_grads()
        if args.block_ckpt_size and args.block_ckpt_size > 1:
            nblk = install_block_checkpointing(base, args.block_ckpt_size)
            if rank == 0:
                print(f"[pointwise_sft] block checkpointing: {nblk} blocks of "
                      f"{args.block_ckpt_size} decoder layers (replaces HF per-layer)", flush=True)
        else:
            # conditional per-layer ckpt: toggled per micro-batch by _set_layer_ckpt (short branches
            # run unchecked for speed; long ones keep recompute to fit). Grab the decoder handle now.
            text_model = _find_text_decoder(base)
    targets = resolve_lora_targets(base, args.lora_targets.split(","))
    if rank == 0:
        print(f"[pointwise_sft] LoRA on {len(targets)} modules "
              f"(e.g. {targets[:4]}{' ...' if len(targets) > 4 else ''})", flush=True)
    base = base.to(device)
    if args.resume:
        model = PeftModel.from_pretrained(base, args.resume, is_trainable=True)
    else:
        cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                         bias="none", task_type="CAUSAL_LM", target_modules=targets)
        model = get_peft_model(base, cfg)
    model.train()
    core = model
    # Manual data-parallel: NO DistributedDataParallel wrapper. A pointwise micro-batch may call
    # the model MANY times (per-branch batch-1 image forwards; cache-mode prefix+branch forwards),
    # which deadlocks DDP's reducer (it expects one forward per backward). Instead every rank runs
    # its own shard, and we all-reduce the (few) LoRA grads by hand at each optimizer boundary.
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))

    def _batch_total_tok(bidx):
        return sum(n_selected(train_exs[i], args.neg_per_pos) * _rec_maxlen(train_exs[i])
                   for i in bidx)

    def _batch_has_image(bidx):
        return any(train_exs[i].is_image for i in bidx)

    from tod.data import sharding                     # deterministic per-epoch/-record seeding

    def _step_cost_s(idx):
        return _batch_cost_s(idx, train_exs, args.neg_per_pos)

    # PER-WINDOW cost balancing (identical on every rank, deterministic in seed+epoch). Micro-batches
    # are cost-estimated with the measured c = Σ_branch (5.92e-8·L² + 4.42e-4·L) model and LPT-assigned
    # to (window, rank) slots so EVERY optimizer step's per-rank cost is balanced (~within 20%), not
    # just the epoch total -- so no rank holds a 32k×N record for ~40 min while the others idle at the
    # barrier. Very heavy single records (> --max-record-cost-s) are split branch-wise across the
    # window's ranks. n_windows = optimizer steps/epoch; identical across epochs (cost is epoch-free).
    split_enabled = world > 1
    _, n_windows = build_window_schedule(
        train_exs, args.neg_per_pos, args.max_branches_per_step, args.max_branch_tokens,
        world=world, grad_accum=args.grad_accum, seed=args.seed, epoch=0,
        max_record_cost_s=args.max_record_cost_s, enable_split=split_enabled)
    opt_total = max(1, n_windows * args.epochs)
    if args.max_steps:
        opt_total = min(opt_total, args.max_steps)
    warmup = int(args.warmup_ratio * opt_total)
    if rank == 0:
        n_heavy = sum(1 for e in train_exs
                      if split_enabled and _rec_cost_s(e, args.neg_per_pos) > args.max_record_cost_s)
        print(f"[pointwise_sft] {len(train_exs)} records -> {n_windows} windows/epoch x "
              f"{args.epochs} epochs = {opt_total} optimizer steps; grad_accum={args.grad_accum}; "
              f"{world} ranks; {n_heavy} heavy records branch-split (>{args.max_record_cost_s}s); "
              f"warmup={warmup}; bce_weight={args.bce_weight}; batch_text_branches="
              f"{args.batch_text_branches}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "train_log.jsonl") if rank == 0 else None
    step = 0
    micro = 0
    tok_seen = 0
    t0 = time.time()
    opt.zero_grad(set_to_none=True)

    def _gather_logodds(local):
        return _all_gather_logodds(local, world, dist)

    win_cost = 0.0                                     # this rank's estimated cost in the open window
    win_wall_t0 = time.time()
    last_m = {"cross_acc": 0.0, "yn_acc": 0.0, "ce": 0.0, "bce": 0.0}
    last_loss = 0.0
    for epoch in range(args.epochs):
        # Cost-balanced per-window micro-step list; identical on every rank. Window order is reshuffled
        # per epoch and negative sampling reshuffles branches per epoch, so this is not a fixed
        # curriculum. Optimizer boundary fires every grad_accum micro-steps (one window).
        steps, _ = build_window_schedule(
            train_exs, args.neg_per_pos, args.max_branches_per_step, args.max_branch_tokens,
            world=world, grad_accum=args.grad_accum, seed=args.seed, epoch=epoch,
            max_record_cost_s=args.max_record_cost_s, enable_split=split_enabled)
        rng = random.Random(args.seed * 100003 + epoch)   # negative-sampling rng (per epoch)
        for st in steps:
            is_boundary = (micro + 1) % args.grad_accum == 0
            if st[0] == "shared":
                # HEAVY record split branch-wise across the window's ranks (all ranks participate;
                # deterministic per-record selection so every rank agrees on the branch set).
                rec_i = st[1]
                ex = train_exs[rec_i]
                srng = random.Random(sharding._seed_int(args.seed, epoch, "pw-sel",
                                                        ex.rid if ex.rid is not None else rec_i))
                sel, gold_pos = select_branches(ex, args.neg_per_pos, srng)
                probs_sel = [ex.probs[sel[p]] for p in range(len(sel))] if ex.probs is not None else None
                loss_v, m, ntok = train_shared_record(
                    model, ex, yes_ids, no_ids, processor, device, autocast_dtype, args.bce_weight,
                    world=world, rank=rank, gather_fn=_gather_logodds, sel=sel, gold_pos=gold_pos,
                    probs_sel=probs_sel, loss_scale=1.0 / args.grad_accum, text_model=text_model,
                    ckpt_min_tokens=args.ckpt_min_tokens)
                last_loss, last_m = float(loss_v), m
                win_cost += _rec_cost_s(ex, args.neg_per_pos) / world
                tok_seen += ntok
            else:
                idx = st[1][rank]                          # this rank's own micro-batch (may be empty)
                if idx:
                    # BATCHED text path when the whole micro-batch is text and fits the batched-forward
                    # memory envelope; otherwise SERIAL per-branch (images, or a single over-cap record).
                    use_batched = (args.batch_text_branches and not _batch_has_image(idx)
                                   and _batch_total_tok(idx) <= args.batched_text_limit)
                    if use_batched:
                        lo, groups, ntok = forward_batch_text_padded(
                            model, train_exs, idx, yes_ids, no_ids, pad_id, device, autocast_dtype,
                            args.neg_per_pos, rng, text_model, args.ckpt_min_tokens)
                        loss, m = compute_pw_loss(lo, groups, args.bce_weight)
                        (loss / args.grad_accum).backward()
                    elif args.serial_branch:
                        loss, m, ntok = train_batch_serial(
                            model, train_exs, idx, yes_ids, no_ids, processor, device, autocast_dtype,
                            args.neg_per_pos, rng, args.bce_weight, loss_scale=1.0 / args.grad_accum,
                            text_model=text_model, ckpt_min_tokens=args.ckpt_min_tokens)
                    else:
                        lo, groups, ntok = forward_batch_solo(model, train_exs, idx, yes_ids, no_ids,
                                                              processor, device, autocast_dtype,
                                                              args.neg_per_pos, rng,
                                                              branch_ckpt=args.branch_ckpt)
                        loss, m = compute_pw_loss(lo, groups, args.bce_weight)
                        (loss / args.grad_accum).backward()
                    last_loss, last_m = float(loss), m
                    win_cost += _step_cost_s(idx)
                    tok_seen += ntok
            micro += 1
            if is_boundary:
                if distributed:                       # manual grad all-reduce (mean over ranks)
                    for p in params:
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
                        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                        p.grad.mul_(1.0 / world)
                lr = _lr_at(step, warmup, opt_total, args.lr)
                for g in opt.param_groups:
                    g["lr"] = lr
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                # Per-step imbalance: max/min per-rank estimated window cost + the actual wall time.
                # The min/max reduce is a (tiny) collective, so it runs only on logging steps --
                # ``step % log_every`` is identical on every rank, so all ranks agree to call it.
                win_wall = time.time() - win_wall_t0
                if step % args.log_every == 0:
                    cmax = cmin = win_cost
                    if distributed:
                        ct = torch.tensor([win_cost], device=device)
                        mx = ct.clone(); dist.all_reduce(mx, op=dist.ReduceOp.MAX)
                        mn = ct.clone(); dist.all_reduce(mn, op=dist.ReduceOp.MIN)
                        cmax, cmin = float(mx.item()), float(mn.item())
                    dt = time.time() - t0
                    _log(rank, log_path, {"step": step, "epoch": epoch,
                                          "loss": round(last_loss, 4),
                                          "cross_acc": round(last_m["cross_acc"], 4),
                                          "yn_acc": round(last_m["yn_acc"], 4),
                                          "ce": round(last_m["ce"], 4), "bce": round(last_m["bce"], 4),
                                          "lr": round(lr, 8),
                                          "win_cost_max_s": round(cmax, 1),
                                          "win_cost_min_s": round(cmin, 1),
                                          "win_wall_s": round(win_wall, 1),
                                          "tok_s": round(tok_seen / max(1e-6, dt), 1)})
                win_cost = 0.0
                win_wall_t0 = time.time()
                if val_exs and args.eval_every and step % args.eval_every == 0:
                    vm = evaluate_val(core, val_exs, pad_id, device, args, yes_ids, no_ids,
                                      autocast_dtype, processor=processor)
                    _log(rank, log_path, {"step": step, **{k: round(v, 4) for k, v in vm.items()}})
                if args.save_every and step % args.save_every == 0 and rank == 0:
                    _save(core, args, os.path.join(args.out, f"step{step}"))
                if distributed:
                    dist.barrier()
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break
    if rank == 0:
        _save(core, args, args.out)
        print(f"[pointwise_sft] done: {step} steps, adapter at {args.out}", flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


# --------------------------------------------------------------------------- merge
def merge(args: argparse.Namespace) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, args.adapter)
    merged = model.merge_and_unload()
    os.makedirs(args.out, exist_ok=True)
    merged.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.model).save_pretrained(args.out)
    print(f"[pointwise_sft] merged {args.adapter} into {args.model} -> {args.out}", flush=True)


# --------------------------------------------------------------------------- check
def check(args: argparse.Namespace) -> None:
    """Dry run over a prepared jsonl: build every record's branches on CPU, asserting the seam
    (prefix ends in a newline; prefix+branch == full-turn tokenisation) for each. Reports seam
    failures, the label-count histogram of BUILT records, how many high-N (>26) rows built, and
    -- if a processor loads -- confirms a sample of image rows tokenise through it."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    has_chat = tok.chat_template is not None
    processor = None
    if not args.no_processor:
        try:
            from transformers import AutoProcessor
            proc = AutoProcessor.from_pretrained(args.model)
            if getattr(proc, "image_processor", None) is not None:
                processor = proc
        except Exception:
            processor = None
    recs = load_records(args.jsonl)
    hist: Counter = Counter()
    seam_fail = []
    built = skipped = built_gt26 = img_built = img_proc_ok = img_proc_fail = 0
    max_n = 0
    rng = random.Random(0)
    for rec in recs:
        rid = rec.get("id")
        is_img = _is_image_row(rec)
        try:
            if not is_img:
                # exercise the seam asserts directly (build_branches raises on any seam break)
                build_branches(tok, has_chat, rec, args.max_tokens)
            ex = build_example(rec, tok, has_chat, args.max_tokens, processor=processor,
                               canonical_root=args.canonical_root)
        except AssertionError as e:
            seam_fail.append(f"{rid}: {e}")
            continue
        except Exception as e:  # image path may miss files on this host; record, don't crash
            seam_fail.append(f"{rid}: {type(e).__name__}: {e}")
            continue
        if ex is None:
            skipped += 1
            continue
        built += 1
        hist[ex.n_options] += 1
        max_n = max(max_n, ex.n_options)
        if ex.n_options > 26:
            built_gt26 += 1
        if ex.is_image:
            img_built += 1
            if processor is not None and img_proc_ok < args.max_image_check:
                try:
                    collate_image([ex], [0], processor, "cpu", args.neg_per_pos, rng)
                    img_proc_ok += 1
                except Exception as e:
                    img_proc_fail += 1
                    seam_fail.append(f"{rid} (processor): {type(e).__name__}: {e}")
    summary = {"records": len(recs), "built": built, "skipped": skipped,
               "seam_failures": len(seam_fail), "built_gt26_labels": built_gt26,
               "max_labels": max_n, "image_rows_built": img_built,
               "image_rows_processor_ok": img_proc_ok, "image_rows_processor_fail": img_proc_fail,
               "label_count_histogram": {str(k): hist[k] for k in sorted(hist)}}
    print("[check] " + json.dumps(summary), flush=True)
    for f in seam_fail[:20]:
        print("[check] FAIL " + f, flush=True)
    if seam_fail:
        raise SystemExit(f"{len(seam_fail)} record(s) failed the seam/build check")
    print("[check] all records passed the seam + build check", flush=True)


# --------------------------------------------------------------------------- cli
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("tod.train.pointwise_sft")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="curated jsonl from /data/canonical (NO <=26-label filter)")
    p.add_argument("--canonical-root", default="/data/canonical")
    p.add_argument("--mix", default="data_pipeline/mixes/train_release.yaml")
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="train", help="canonical split to sample (train|val)")
    p.add_argument("--per-dataset", type=int, default=1000)
    p.add_argument("--max-state-chars", type=int, default=200000,
                   help="cheap char pre-filter only (default ~200k = effectively none); the real "
                        "length gate is --max-tokens on the sampled reservoir")
    p.add_argument("--tok-model", default="google/gemma-4-12B-it",
                   help="tokenizer for the token-length gate + histogram")
    p.add_argument("--max-tokens", type=int, default=49152,
                   help="drop rows whose full pointwise prompt exceeds this many tokens "
                        "(owner rule: never below 49152)")
    p.add_argument("--reason-weight", type=float, default=1.0,
                   help="oversample cap multiplier for reasoning-heavy families")
    p.add_argument("--include-images", action="store_true",
                   help="keep image rows (scienceqa/ai2d/chartqa) for multimodal SFT")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=prepare)

    t = sub.add_parser("train", help="pointwise LoRA SFT (torchrun-compatible)")
    t.add_argument("--model", required=True)
    t.add_argument("--seed", type=int, default=0, help="epoch shuffle / negative-sampling seed")
    t.add_argument("--train-jsonl", required=True)
    t.add_argument("--val-jsonl", default=None)
    t.add_argument("--out", required=True)
    t.add_argument("--lora-r", type=int, default=16)
    t.add_argument("--lora-alpha", type=int, default=32)
    t.add_argument("--lora-dropout", type=float, default=0.05)
    t.add_argument("--lora-targets", default="q_proj,k_proj,v_proj,o_proj",
                   help="comma list; may add gate_proj,up_proj,down_proj")
    t.add_argument("--lr", type=float, default=3e-5)
    t.add_argument("--epochs", type=int, default=2)
    t.add_argument("--max-steps", type=int, default=0, help="cap optimizer steps (0=unlimited)")
    t.add_argument("--max-tokens", type=int, default=49152,
                   help="max prefix+branch tokens per branch (owner rule: never below 49152; rows "
                        "were already dropped above this at prepare time)")
    t.add_argument("--batch-tokens", type=int, default=16384,
                   help="branch-token budget per micro-batch (sum over the batch's branches)")
    t.add_argument("--grad-accum", type=int, default=4,
                   help="micro-batches per optimizer step. Default 4: fewer cross-rank barriers "
                        "(the barrier only fires at optimizer boundaries), and the length-balanced "
                        "round-robin schedule keeps each rank's per-step cost similar so no rank "
                        "idles waiting on a straggler.")
    t.add_argument("--warmup-ratio", type=float, default=0.03)
    t.add_argument("--bce-weight", type=float, default=1.0,
                   help="weight on the per-branch yes/no BCE term (cross-branch CE weight is 1.0)")
    t.add_argument("--branch-mode", choices=["flat", "cache"], default="flat",
                   help="flat (default): pack prefix+branch as separate sequences (exact, "
                        "grad-ckpt/DDP friendly); cache: shared-prefix KV-cache (text-only, "
                        "cheaper, needs --no-grad-checkpointing)")
    t.add_argument("--max-branches-per-step", type=int, default=64,
                   help="cap on total branches (sequences) in a text micro-batch")
    t.add_argument("--max-branch-tokens", type=int, default=0,
                   help="cap on the SUMMED branch tokens in a micro-batch (0=off). Bounds the "
                        "coexisting recomputed-branch memory during the cross-branch backward so a "
                        "batch of several long rows cannot OOM; a single record over the cap goes "
                        "alone. Short-row batches stay branch-count bound.")
    t.add_argument("--neg-per-pos", type=int, default=8,
                   help="for N > 1+this, sample this many negatives per record (gold always kept); "
                        "0 = use all branches (exact softmax over full N)")
    t.add_argument("--canonical-root", default="/data/canonical",
                   help="root for resolving relative image paths in image rows")
    t.add_argument("--mm-batch-size", type=int, default=1,
                   help="image rows per micro-batch (1-4; keep small/robust)")
    t.add_argument("--no-grad-checkpointing", dest="grad_checkpointing", action="store_false",
                   default=True)
    t.add_argument("--batch-text-branches", dest="batch_text_branches", action="store_true",
                   default=False,
                   help="BATCHED text path (default OFF -- launch serial until parity is forwarded): "
                        "forward a micro-batch's text branches as one RIGHT-padded [B, Lmax] pass "
                        "(chunked_eager, explicit position_ids, per-row last-real-token logits via "
                        "the LM head on the gathered hidden state) instead of B batch-1 forwards. "
                        "Images always stay per-branch batch-1. Numerical parity vs the solo path "
                        "must be verified before this is enabled.")
    t.add_argument("--batched-text-limit", type=int, default=30000,
                   help="a text micro-batch with summed branch tokens <= this uses the batched "
                        "forward (fits with per-layer ckpt); larger ones fall back to serial")
    t.add_argument("--ckpt-min-tokens", dest="ckpt_min_tokens", type=int, default=14000,
                   help="per-layer gradient checkpointing is OFF for a branch (serial) or batch "
                        "(batched) with <= this many tokens -- it fits an 80 GB H100 unchecked, so "
                        "recompute is pure overhead; sequences ABOVE this keep per-layer ckpt to fit")
    t.add_argument("--no-serial-branch", dest="serial_branch", action="store_false", default=True,
                   help="SERIAL branch backward with manual grad seeding (default on): forward all "
                        "branches no-grad -> scalars, seed d(loss)/d(log-odd), then backward each "
                        "branch one at a time. Peak = ONE branch regardless of option count / batch "
                        "size (the cross-branch CE otherwise forces all branch graphs to coexist "
                        "and OOMs on long rows). Off -> the coupled forward_batch_solo path.")
    t.add_argument("--no-branch-ckpt", dest="branch_ckpt", action="store_false", default=True,
                   help="(coupled path only) branch-level checkpointing: recompute each branch in "
                        "backward. NOTE: does NOT bound peak because the cross-branch CE makes the "
                        "recomputed branch graphs coexist; use --serial-branch instead.")
    t.add_argument("--block-ckpt-size", type=int, default=0,
                   help="group this many consecutive decoder layers into ONE non-reentrant "
                        "checkpoint region instead of HF per-layer (fewer SAVED residual tensors, "
                        "but recompute materialises the whole block's activations at once). "
                        "MEASURED on gemma-4-12B: block-of-6 is WORSE than HF per-layer at 32k+ "
                        "tokens (the block recompute transient exceeds the saved-residual saving), "
                        "so 0 (HF per-layer, default) is best; keep block ckpt for experiments.")
    t.add_argument("--attn", default="chunked_eager",
                   help="chunked_eager (default): exact eager attention, query-chunked for O(L) "
                        "memory on long rows (sdpa is numerically wrong on this model; FA2 "
                        "unavailable: head_dim>256). Also accepts eager/sdpa for ablation.")
    t.add_argument("--device", default="cuda")
    t.add_argument("--max-record-cost-s", type=float, default=240.0,
                   help="a single record whose estimated serial cost (Σ_branch 5.92e-8·L²+4.42e-4·L "
                        "seconds) exceeds this is BRANCH-SPLIT across the ranks of its window "
                        "(prefix re-encoded per rank, branches partitioned, cross-branch CE from an "
                        "all_gather of the scalar log-odds) instead of stacking on one rank. Only "
                        "active when world_size > 1.")
    t.add_argument("--dist-timeout-hours", type=float, default=4.0,
                   help="NCCL process-group collective timeout (hours). Must exceed the slowest "
                        "cross-rank imbalance in one optimizer step; long-row micro-batches can be "
                        "several hundred seconds, so the 600 s default is too small.")
    t.add_argument("--log-every", type=int, default=10)
    t.add_argument("--eval-every", type=int, default=200)
    t.add_argument("--save-every", type=int, default=0)
    t.add_argument("--resume", default=None, help="adapter dir to resume weights from")
    t.set_defaults(func=train)

    m = sub.add_parser("merge", help="merge_and_unload adapter into base -> served dir")
    m.add_argument("--model", required=True)
    m.add_argument("--adapter", required=True)
    m.add_argument("--out", required=True)
    m.set_defaults(func=merge)

    c = sub.add_parser("check", help="CPU dry run: build every record + assert the seam")
    c.add_argument("--model", required=True, help="tokenizer/processor source (no weights loaded)")
    c.add_argument("--jsonl", required=True)
    c.add_argument("--max-tokens", type=int, default=8192)
    c.add_argument("--canonical-root", default="/data/canonical")
    c.add_argument("--neg-per-pos", type=int, default=8)
    c.add_argument("--max-image-check", type=int, default=5,
                   help="image rows to actually push through the processor")
    c.add_argument("--no-processor", action="store_true",
                   help="skip processor load (validate text/seam only)")
    c.set_defaults(func=check)
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

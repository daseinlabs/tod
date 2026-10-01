"""Letter-logit ("SemIf protocol") JevBench harness for a FROZEN or lightly-tuned causal LM.

This is the recipe every top open JevBench submission uses (research/26 §1.2, §2.2; jevbench
issue #31): keep the decoder causal + instruct-tuned, render ``state + instructions + lettered
options`` through the model's chat template, take ONE forward pass, read the logits of the
option letters at the last position, softmax over exactly those letters (restricted softmax),
optionally divide by one temperature. No generation, no custom heads, no mask surgery.

It reuses ``tod.eval.metrics.full_report`` so the numbers are directly comparable with the
UTOD trajectory (``eval_latest``): same tiers, same 10-bin ECE / debiased ECE / Brier, same
argmax rule (ties -> lexicographically smallest label, matching jevbench/scoring.py).

Temperature: fitted (Brier-optimal, grid) on an OFF-BENCHMARK calibration file (canonical
val rows in the JevBench/canonical record schema), never on JevBench itself. Both the raw
(T=1) and calibrated numbers are reported.

Usage (box, one GPU per model):
  python -m tod.eval.letter_logit --model google/gemma-4-12B-it \
      --jevbench-root /data/raw/jevbench/datasets/public --calib-jsonl /data/calib_val.jsonl \
      --out /data/letter_logit/gemma-4-12B-it.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import string
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import metrics

LETTERS = list(string.ascii_uppercase)  # JevBench max |labels| is well under 26 (asserted)
# Ablation knobs (set from the CLI; defaults reproduce the committed zero-shot numbers).
OPTION_STYLE = "label_desc"   # "label_desc" (ours) | "desc" (Cygnet verbatim: description only)


# ----------------------------------------------------------------------------- records
def load_records(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _state_text(state: Any) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=1)


def render_options(rec: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Return (labels, option_lines). One letter per label, in the record's label order.

    noul  -> labels ["no","yes"]; criteria (if any) is keyed true/false -> mapped to yes/no.
    choice-> labels are the criteria keys; description from criteria[label].
    score -> labels are level indices "0".."k-1"; description = criteria[int(label)].
    """
    q = rec["question"]
    qtype = q["type"]
    crit = q.get("criteria")
    labels = [str(x) for x in rec["labels"]]
    assert len(labels) <= len(LETTERS), f"{rec['id']}: {len(labels)} labels > 26"
    lines = []
    for i, lab in enumerate(labels):
        desc: Optional[str] = None
        if qtype == "noul" and isinstance(crit, dict):
            desc = crit.get({"yes": "true", "no": "false"}.get(lab, lab))
        elif qtype == "score" and isinstance(crit, (list, tuple)):
            try:
                desc = crit[int(lab)]
            except (ValueError, IndexError):
                desc = None
        elif isinstance(crit, dict):
            desc = crit.get(lab)
        # Cygnet renders "{letter}. {description}" (description only). We keep the label
        # name too when it is not already the description's prefix: label names are the
        # only content some rows carry (e.g. intent keys), and a letter->label map is what
        # the scorer needs anyway.
        d = str(desc).strip() if desc else ""
        if OPTION_STYLE == "desc" and d:
            lines.append(f"{LETTERS[i]}. {d}")
        elif d and d != lab and not d.lower().startswith(lab.lower()):
            lines.append(f"{LETTERS[i]}. {lab}: {d}")
        elif d:
            lines.append(f"{LETTERS[i]}. {d}")
        else:
            lines.append(f"{LETTERS[i]}. {lab}")
    return labels, lines


# Verbatim structure of the Cygnet shim (blockbrain-ai/cygnet-recipe, jevbench issue #71,
# frozen gemma-4-12B-it, 203/231 public): a "calibration engine" system prompt, then
# state / instructions / "Options:" / letter lines / "Answer with the letter of exactly one
# option, and nothing else:". Gemma chat templates fold the system prompt into the first
# user turn, so we do that explicitly to be template-agnostic.
PROMPT_VERSION = "cygnet.v1"
SYSTEM_PROMPT = (
    "You are a calibration engine. You never answer in prose. You are given a state, a "
    "question about it, and a fixed list of lettered options. You reply with that option's "
    "LETTER and nothing else - a single character, no words, no punctuation, no explanation."
)


def _message_head() -> str:
    return SYSTEM_PROMPT + "\n\n"


def _message_tail(rec: Dict[str, Any]) -> str:
    q = rec["question"]
    _labels, lines = render_options(rec)
    return (
        "\n\n" + str(q["instructions"]).strip() + "\n\n"
        + "Options:\n" + "\n".join(lines) + "\n\n"
        + "Answer with the letter of exactly one option, and nothing else:"
    )


def build_user_message(rec: Dict[str, Any]) -> str:
    # Text (string / dict) state: byte-identical to the committed harness. This is the ONLY
    # renderer the text tokenizer path ever sees, so text-row prompts never change.
    return _message_head() + _state_text(rec["state"]).strip() + _message_tail(rec)


# --------------------------------------------------------------------------- multimodal
def _row_dataset(rec: Dict[str, Any]) -> Optional[str]:
    """Canonical dataset name for a record (used to resolve relative image paths)."""
    src = rec.get("source")
    if isinstance(src, dict) and src.get("dataset"):
        return str(src["dataset"])
    meta = rec.get("meta")
    if isinstance(meta, dict) and meta.get("dataset"):
        return str(meta["dataset"])
    return rec.get("dataset")


def _resolve_image_path(rel: str, canonical_root: Optional[str], dataset: Optional[str]) -> str:
    """``images/<sha>.png`` is stored RELATIVE to ``<canonical_root>/<dataset>/``."""
    if os.path.isabs(rel) or not canonical_root:
        return rel
    base = os.path.join(canonical_root, dataset) if dataset else canonical_root
    return os.path.join(base, rel)


def build_user_content(rec: Dict[str, Any],
                       canonical_root: Optional[str] = None) -> List[Dict[str, Any]]:
    """Ordered user-turn content parts for a record.

    Returns a list of ``{"type": "text", "text": ...}`` and ``{"type": "image", "path": ...}``
    dicts, in the order they appear in the state. The SYSTEM_PROMPT is folded into the first
    text part and the instructions / options / answer-cue are appended to the last text part,
    exactly as ``build_user_message`` does.

    A text row (string / dict / JSON-ish ``state``) yields a SINGLE text part whose text is
    byte-identical to ``build_user_message(rec)`` -- so the text tokenizer path is unchanged.
    An image row (``state`` is a LIST of ``{"text": ...}`` / ``{"image": ...}`` segments)
    renders the text segments in order with an image slot at each image segment.
    """
    state = rec.get("state")
    if not isinstance(state, list):
        return [{"type": "text", "text": build_user_message(rec)}]

    dataset = _row_dataset(rec)
    parts: List[Dict[str, Any]] = []
    buf = _message_head()
    for seg in state:
        if isinstance(seg, dict) and "image" in seg:
            if buf:
                parts.append({"type": "text", "text": buf})
                buf = ""
            parts.append({"type": "image",
                          "path": _resolve_image_path(str(seg["image"]), canonical_root, dataset)})
        elif isinstance(seg, dict):
            buf += str(seg.get("text", ""))
        else:
            buf += str(seg)
    buf += _message_tail(rec)
    if buf:
        parts.append({"type": "text", "text": buf})
    return parts


def render_content_text(parts: Sequence[Dict[str, Any]],
                        image_placeholder: str = "[image omitted]") -> str:
    """Flatten content parts to plain text, substituting ``image_placeholder`` for each image
    part. This is exactly the text the processor sees on the image-ablation ("without image")
    arm, and lets the CPU tests check rendering without loading a model/processor."""
    out = []
    for p in parts:
        out.append(p["text"] if p["type"] == "text" else image_placeholder)
    return "".join(out)


def build_processor_inputs(processor: Any, contents: Sequence[Sequence[Dict[str, Any]]],
                           *, drop_images: bool = False):
    """Run ``processor.apply_chat_template`` + ``processor(text=, images=)`` over a batch of
    content-part lists, so ``input_ids`` (with expanded image placeholder tokens) and
    ``pixel_values`` come out consistently. Returns ``(enc, last)`` where ``last`` is the index
    of each row's final real token (predicts the answer letter).

    ``drop_images`` is the image-ablation arm: each image part is replaced by the literal text
    ``"[image omitted]"`` and no pixels are sent, so the model reads the identical prompt with
    the image content removed.
    """
    from PIL import Image
    texts: List[str] = []
    images: List[Any] = []
    for parts in contents:
        content: List[Dict[str, Any]] = []
        for p in parts:
            if p["type"] == "text":
                content.append({"type": "text", "text": p["text"]})
            elif drop_images:
                content.append({"type": "text", "text": "[image omitted]"})
            else:
                images.append(Image.open(p["path"]).convert("RGB"))
                content.append({"type": "image"})
        texts.append(processor.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False))
    if images:
        enc = processor(text=texts, images=images, return_tensors="pt", padding=True)
    else:
        enc = processor(text=texts, return_tensors="pt", padding=True)
    import torch
    am = enc["attention_mask"]
    tk = getattr(processor, "tokenizer", None)
    if getattr(tk, "padding_side", "left") == "left":
        last = torch.full((am.size(0),), am.size(1) - 1, dtype=torch.long)
    else:
        last = am.sum(1).long() - 1
    return enc, last


# ----------------------------------------------------------------------------- model
def _forward_last(scorer: Any, **kw):
    """Model forward that keeps ONLY the last position's logits when the model supports it
    (transformers >= 4.45 ``logits_to_keep=1``), so the fp32 LM head no longer materialises
    full-sequence logits and activation memory stops scaling with prefix length. This is safe
    for every caller here because the position they read is always the final column (the answer
    / verdict slot, or -- in the cache route's prefix pass -- no logits at all): callers index
    the sequence dim with ``-1`` regardless, which selects that column in both the kept
    ``[B, 1, V]`` and the full ``[B, L, V]`` output. If the model's forward rejects the kwarg the
    decision is cached once (TypeError) and every later forward falls back to full logits."""
    if scorer._logits_to_keep_ok is None:
        try:
            out = scorer.model(logits_to_keep=1, **kw)
            scorer._logits_to_keep_ok = True
            return out
        except TypeError:
            scorer._logits_to_keep_ok = False
    if scorer._logits_to_keep_ok:
        return scorer.model(logits_to_keep=1, **kw)
    return scorer.model(**kw)


def _load_causal(model_id: str, device: str, kw: dict):
    """Load the base model on ``device``. ``device="auto"`` shards across every visible GPU with
    ``device_map="auto"`` (used when one GPU cannot hold the fp32 12B, e.g. 2x40 GB); the
    returned model's ``.device`` is then the first shard's device, which is where inputs go.
    Every other value is a single device and the model is moved to it as before."""
    from transformers import AutoModelForCausalLM
    if device == "auto":
        kw = dict(kw); kw["device_map"] = "auto"
        return AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()
    return AutoModelForCausalLM.from_pretrained(model_id, **kw).to(device).eval()


class LetterLogitScorer:
    def __init__(self, model_id: str, device: str = "cuda", dtype: str = "bfloat16",
                 attn: str = "sdpa", max_tokens: int = 16384, revision: Optional[str] = None,
                 adapter: Optional[str] = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        # A native multimodal checkpoint (Gemma4Unified / Gemma4) ships an AutoProcessor with an
        # image_processor; text-only checkpoints do not. When present we score image rows through
        # it (pixel_values + expanded image tokens); text rows always keep the tokenizer path.
        self.processor = None
        try:
            from transformers import AutoProcessor
            proc = AutoProcessor.from_pretrained(model_id, revision=revision)
            if getattr(proc, "image_processor", None) is not None:
                self.processor = proc
        except Exception:
            self.processor = None
        if attn == "chunked_eager":
            # exact query-chunked eager registered via transformers AttentionInterface -- same
            # side-effect import PointwiseScorer does; needed on the stage-1 retriever path too,
            # where this scorer is the only one that loads the letter model.
            import tod.model.attn_chunked  # noqa: F401  (import side-effect registers it)
        kw = dict(dtype=getattr(torch, dtype), attn_implementation=attn, revision=revision)
        t0 = time.time()
        self.model = _load_causal(model_id, device, kw)
        self.device = self.model.device if device == "auto" else device
        if adapter:
            from peft import PeftModel  # optional: a LoRA trained by tod.train.letter_sft
            self.model = PeftModel.from_pretrained(self.model, adapter).eval()
        self.load_s = time.time() - t0
        self._logits_to_keep_ok: Optional[bool] = None  # see _forward_last (calib memory)
        # Letter token ids: the assistant turn starts right after the template's generation
        # prompt, so the first token is "A" (no leading space) -- but some templates/models
        # emit " A". We take logsumexp over both variants per letter (restricted softmax is
        # then over the union), which is order-preserving and never hurts a model that only
        # uses one of them.
        self.letter_ids: List[List[int]] = []
        for L in LETTERS:
            ids = []
            for v in (L, " " + L):
                t = self.tok.encode(v, add_special_tokens=False)
                if len(t) == 1:
                    ids.append(t[0])
            assert ids, f"letter {L!r} is not a single token for {model_id}"
            self.letter_ids.append(sorted(set(ids)))
        self.has_chat = self.tok.chat_template is not None

    def prompt_ids(self, user: str) -> List[int]:
        if self.has_chat:
            msgs = [{"role": "user", "content": user}]
            # tokenize=False then encode: the tokenize=True return type differs across
            # transformers versions (list / BatchEncoding / dict-like). The template already
            # emits BOS, so add_special_tokens=False.
            text = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            ids = self.tok.encode(text, add_special_tokens=False)
        else:
            ids = self.tok.encode(user + "\nAnswer:", add_special_tokens=True)
        return list(ids)

    def score(self, rec: Dict[str, Any], order_avg: bool = False) -> Dict[str, Any]:
        """Letter logits in the record's label order. With ``order_avg`` the item is also
        read with the options REVERSED and the two restricted log-softmaxes are averaged
        (reflex-27b's two-order read; kills position bias)."""
        out = self._score_one(rec)
        if not order_avg or len(rec["labels"]) < 2:
            return out
        rev = dict(rec); rev["labels"] = list(rec["labels"])[::-1]
        o2 = self._score_one(rev)
        import torch
        a = torch.log_softmax(torch.tensor(out["logits"]), 0)
        b = torch.log_softmax(torch.tensor(o2["logits"][::-1]), 0)
        out["logits_fwd"] = out["logits"]; out["logits_rev"] = o2["logits"][::-1]
        out["logits"] = (0.5 * (a + b)).tolist()
        out["n_tokens"] = max(out["n_tokens"], o2["n_tokens"])
        return out

    def _score_one(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        user = build_user_message(rec)
        ids = self.prompt_ids(user)
        truncated = False
        if len(ids) > self.max_tokens:
            # keep the head (BOS+template+State start) and the tail (question+options+answer
            # cue); drop from the middle of the state. Recorded, never silent.
            keep_tail = min(4096, self.max_tokens // 4)
            ids = ids[: self.max_tokens - keep_tail] + ids[-keep_tail:]
            truncated = True
        x = torch.tensor([ids], device=self.device)
        with torch.no_grad():
            out = _forward_last(self, input_ids=x, use_cache=False)
            logits = out.logits[0, -1].float()
        n = len(rec["labels"])
        letter_logits = torch.stack([torch.logsumexp(logits[self.letter_ids[i]], 0) for i in range(n)])
        return {"logits": letter_logits.cpu().tolist(), "n_tokens": len(ids), "truncated": truncated}

    def score_mm(self, rec: Dict[str, Any], canonical_root: Optional[str] = None,
                 drop_images: bool = False) -> Dict[str, Any]:
        """Letter logits for a (possibly multimodal) record through the native processor.

        Text rows fall back to the tokenizer path (byte-identical to ``score``). Image rows are
        rendered with ``build_user_content`` and scored through ``self.processor`` so pixels reach
        the vision tower; ``drop_images`` is the ablation arm (image parts -> "[image omitted]").
        """
        import torch
        parts = build_user_content(rec, canonical_root)
        has_image = any(p["type"] == "image" for p in parts)
        if self.processor is None or not has_image:
            # no image content (or no processor): keep the exact tokenizer readout
            out = self._score_one(rec)
            out["modality"] = "text"
            return out
        enc, last = build_processor_inputs(self.processor, [parts], drop_images=drop_images)
        enc = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in enc.items()}
        with torch.no_grad():
            out = _forward_last(self, **enc, use_cache=False)
            # single row -> last[0] is always the final column, so -1 == last[0] here and also
            # selects the one kept position when logits_to_keep=1 was applied.
            logits = out.logits[0, -1].float()
        n = len(rec["labels"])
        letter_logits = torch.stack([torch.logsumexp(logits[self.letter_ids[i]], 0) for i in range(n)])
        return {"logits": letter_logits.cpu().tolist(),
                "n_tokens": int(enc["input_ids"].shape[1]), "truncated": False,
                "modality": "text_ablation" if drop_images else "image"}

    @classmethod
    def share(cls, base: Any, model_id: str) -> "LetterLogitScorer":
        """Build a LetterLogitScorer that REUSES an already-loaded model+tokenizer (e.g. the
        stage-1 PointwiseScorer's), so a two-stage run loads the base once. Only the letter-id
        table (which is tokenizer-derived, not weight-derived) is rebuilt."""
        self = cls.__new__(cls)
        self.model_id = model_id
        self.max_tokens = base.max_tokens
        self.device = base.device
        self.tok = base.tok
        self.processor = getattr(base, "processor", None)
        self.model = base.model
        self.load_s = 0.0
        self._logits_to_keep_ok = None  # see _forward_last (calib memory)
        self.letter_ids = []
        for L in LETTERS:
            ids = []
            for v in (L, " " + L):
                t = self.tok.encode(v, add_special_tokens=False)
                if len(t) == 1:
                    ids.append(t[0])
            assert ids, f"letter {L!r} is not a single token for {model_id}"
            self.letter_ids.append(sorted(set(ids)))
        self.has_chat = self.tok.chat_template is not None
        return self


# ------------------------------------------------------------------- pointwise (unbounded N)
# The product readout (owner rule 2026-09-28: any option cap is worthless). No option list in
# the prompt; each option is an independent BRANCH "description -> verdict slot" appended to a
# shared prefix (state + question); at the slot the model's own head is read restricted to
# yes/no; choice = softmax over branch log-odds / T. N is bounded only by context. Branches
# never see each other -> order-invariant + IIA by construction (the original VBS property).
# Exactness route: the prefix KV-cache is computed ONCE and every branch continues it as its
# own batch row, so branch_i's logits equal scoring "prefix + branch_i" alone (bf16 noise) --
# no custom 4D mask (HF's Gemma-4 sliding-window path does not honour one exactly; measured).
PW_PROMPT_VERSION = "pointwise.v1"
PW_SYSTEM = (
    "You are a calibration engine. You never answer in prose. You are given a state, a "
    "question about it, and ONE candidate option. You judge whether that candidate is the "
    "correct answer to the question for this state. You reply with yes or no and nothing else."
)


def pw_option_descs(rec: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """(labels, one description per label) -- same label/description resolution as
    render_options but without any letters or list rendering."""
    q = rec["question"]; qtype = q["type"]; crit = q.get("criteria")
    labels = [str(x) for x in rec["labels"]]
    descs = []
    for lab in labels:
        desc: Optional[str] = None
        if qtype == "noul" and isinstance(crit, dict):
            desc = crit.get({"yes": "true", "no": "false"}.get(lab, lab))
        elif qtype == "score" and isinstance(crit, (list, tuple)):
            try:
                desc = crit[int(lab)]
            except (ValueError, IndexError):
                desc = None
        elif isinstance(crit, dict):
            desc = crit.get(lab)
        d = str(desc).strip() if desc else ""
        if qtype == "noul" and not d:
            d = "The proposition holds." if lab == "yes" else "The proposition does not hold."
        # Opaque label (e.g. SalesBench "c17", or label == its own description): the branch is
        # the DESCRIPTION ALONE -- prefixing "c17: ..." would leak a meaningless id into the
        # verdict and (worse) let branch identity ride on the id rather than the semantics.
        opaque = bool(re.match(r"^c\d+$", lab)) or (bool(d) and d == lab)
        if d and not opaque and d != lab and not d.lower().startswith(lab.lower()):
            descs.append(f"{lab}: {d}")
        else:
            descs.append(d or lab)
    return labels, descs


def pw_prefix_text(rec: Dict[str, Any]) -> str:
    q = rec["question"]
    return (PW_SYSTEM + "\n\n" + _state_text(rec["state"]).strip() + "\n\n"
            + str(q["instructions"]).strip() + "\n\nCandidate option:\n")


def pw_branch_text(desc: str) -> str:
    return desc.strip() + "\n\nIs this candidate the correct answer? Answer yes or no:"


class PointwiseScorer:
    """Per-option yes/no log-odds under a shared prefix KV-cache; any N."""

    def __init__(self, model_id: str, device: str = "cuda", dtype: str = "bfloat16",
                 attn: str = "sdpa", max_tokens: int = 16384, revision: Optional[str] = None,
                 adapter: Optional[str] = None, branch_batch: int = 16, route: str = "cache"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_id = model_id; self.device = device; self.max_tokens = max_tokens
        self.branch_batch = branch_batch
        # "cache": prefix KV computed once, branches continue it (fast, O(prefix + N*branch)).
        # "solo_batched": each branch scored as a full [prefix+branch] sequence, right-padded
        #   and batched (exact by construction, O(N*prefix)); the fallback if the cached path
        #   is not bit-exact on this model's attention (e.g. sliding-window mask drift).
        self.route = route
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        # Native multimodal checkpoints ship an AutoProcessor with an image_processor; image rows
        # are scored through it (pixel_values + expanded image tokens) on the --multimodal path.
        self.processor = None
        try:
            from transformers import AutoProcessor
            proc = AutoProcessor.from_pretrained(model_id, revision=revision)
            if getattr(proc, "image_processor", None) is not None:
                self.processor = proc
        except Exception:
            self.processor = None
        if attn == "chunked_eager":
            # exact query-chunked eager (O(L*chunk) memory, honours the model's own mask incl.
            # sliding window) registered via transformers AttentionInterface -- used for prefixes
            # that would exceed plain eager's O(L^2) memory, so long context is never truncated.
            import tod.model.attn_chunked  # noqa: F401  (import side-effect registers it)
        t0 = time.time()
        self.model = _load_causal(model_id, device, dict(
            dtype=getattr(torch, dtype), attn_implementation=attn, revision=revision))
        self.device = self.model.device if device == "auto" else device
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter).eval()
        self.load_s = time.time() - t0
        self.has_chat = self.tok.chat_template is not None
        self._logits_to_keep_ok: Optional[bool] = None  # see _forward_last (calib memory)
        self.yes_ids = self._variants("yes") + self._variants("Yes")
        self.no_ids = self._variants("no") + self._variants("No")
        assert self.yes_ids and self.no_ids, "yes/no are not single tokens for this tokenizer"

    def _variants(self, w: str) -> List[int]:
        out = []
        for v in (w, " " + w):
            t = self.tok.encode(v, add_special_tokens=False)
            if len(t) == 1:
                out.append(t[0])
        return sorted(set(out))

    # -- tokenisation: the user turn is prefix_text + branch_text; the chat template wraps it.
    #    We tokenise the template's head (up to and including the prefix text) once, and each
    #    branch as the remainder. A newline ends the prefix so BPE merges never cross the seam;
    #    an assertion checks the seam for every record (exactness, not hope).
    def _split_ids(self, prefix_text: str, branch_text: str) -> Tuple[List[int], List[int]]:
        if self.has_chat:
            full = self.tok.apply_chat_template([{"role": "user", "content": prefix_text + branch_text}],
                                                add_generation_prompt=True, tokenize=False)
            head = self.tok.apply_chat_template([{"role": "user", "content": prefix_text}],
                                                add_generation_prompt=False, tokenize=False)
            # the template may append an end-of-turn after the prefix; keep only the part that
            # is a literal prefix of `full`.
            while head and not full.startswith(head):
                head = head[:-1]
        else:
            full = prefix_text + branch_text + "\nAnswer:"; head = prefix_text
        head_ids = self.tok.encode(head, add_special_tokens=not self.has_chat)
        full_ids = self.tok.encode(full, add_special_tokens=not self.has_chat)
        if full_ids[: len(head_ids)] != head_ids:
            # rare seam merge: retreat the head to the longest common prefix
            k = 0
            while k < min(len(head_ids), len(full_ids)) and head_ids[k] == full_ids[k]:
                k += 1
            head_ids = full_ids[:k]
        return head_ids, full_ids[len(head_ids):]

    def branch_ids(self, rec: Dict[str, Any]) -> Tuple[List[int], List[List[int]], bool]:
        labels, descs = pw_option_descs(rec)
        ptxt = pw_prefix_text(rec)
        prefix, first = self._split_ids(ptxt, pw_branch_text(descs[0]))
        branches = [first] + [self._split_ids(ptxt, pw_branch_text(d))[1] for d in descs[1:]]
        truncated = False
        longest = max(len(b) for b in branches)
        if len(prefix) + longest > self.max_tokens:
            keep = self.max_tokens - longest
            keep_tail = min(1024, keep // 4)
            prefix = prefix[: keep - keep_tail] + prefix[-keep_tail:]
            truncated = True
        return prefix, branches, truncated

    def _logodds(self, logits) -> float:
        import torch
        return (torch.logsumexp(logits[self.yes_ids], 0) - torch.logsumexp(logits[self.no_ids], 0)).item()

    def _score_cache(self, prefix: List[int], branches: List[List[int]]) -> List[float]:
        """Prefix KV computed once; branches batched as continuations of a repeated copy."""
        import copy
        import torch
        dev = self.device
        out: List[float] = [0.0] * len(branches)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        with torch.no_grad():
            # prefix pass: only past_key_values is used (its logits are discarded), so keeping a
            # single position stops the fp32 LM head from materialising [1, prefix_len, V].
            pre = _forward_last(self, input_ids=torch.tensor([prefix], device=dev), use_cache=True)
            cache = pre.past_key_values
            for s in range(0, len(branches), self.branch_batch):
                chunk = branches[s: s + self.branch_batch]; B = len(chunk)
                L = max(len(b) for b in chunk)
                ids = torch.full((B, L), pad, dtype=torch.long)
                am = torch.zeros((B, len(prefix) + L), dtype=torch.long); am[:, : len(prefix)] = 1
                for j, b in enumerate(chunk):
                    ids[j, : len(b)] = torch.tensor(b); am[j, len(prefix): len(prefix) + len(b)] = 1
                pos = torch.arange(len(prefix), len(prefix) + L)[None].expand(B, L)
                c = copy.deepcopy(cache)
                c.batch_repeat_interleave(B)  # in-place (transformers 5.14 DynamicCache)
                o = self.model(input_ids=ids.to(dev), attention_mask=am.to(dev),
                               position_ids=pos.to(dev), past_key_values=c, use_cache=True)
                lg = o.logits.float()
                for j, b in enumerate(chunk):
                    out[s + j] = self._logodds(lg[j, len(b) - 1])
        return out

    def _score_solo_batched(self, prefix: List[int], branches: List[List[int]]) -> List[float]:
        """Each branch as a full [prefix+branch] sequence, right-padded and batched. Exact by
        construction (no cache surgery): row j's last-real-token logits attend only to its own
        prefix+branch tokens under the causal mask."""
        import torch
        dev = self.device
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        seqs = [prefix + b for b in branches]
        out: List[float] = [0.0] * len(seqs)
        with torch.no_grad():
            for s in range(0, len(seqs), self.branch_batch):
                chunk = seqs[s: s + self.branch_batch]; B = len(chunk)
                L = max(len(x) for x in chunk)
                ids = torch.full((B, L), pad, dtype=torch.long)
                am = torch.zeros((B, L), dtype=torch.long)
                for j, x in enumerate(chunk):
                    ids[j, : len(x)] = torch.tensor(x); am[j, : len(x)] = 1
                o = self.model(input_ids=ids.to(dev), attention_mask=am.to(dev), use_cache=False)
                lg = o.logits.float()
                for j, x in enumerate(chunk):
                    out[s + j] = self._logodds(lg[j, len(x) - 1])
        return out

    def _score_packed4d(self, prefix: List[int], branches: List[List[int]]) -> List[float]:
        """Single forward over [prefix, branch_0, branch_1, ...] with a 4D block mask: each
        branch token attends to the prefix + its own branch (causal), never other branches;
        position_ids restart at len(prefix) for every branch. One pass, O(prefix + sum branch)
        -- the fastest route -- but only exact if the model honours a custom 4D mask on its
        sliding-window layers (verify with the self-test; exactness needs fp32)."""
        import torch
        dev = self.device
        P = len(prefix)
        ids: List[int] = list(prefix)
        pos: List[int] = list(range(P))
        seg: List[int] = [-1] * P            # -1 = prefix, else branch index
        last_idx: List[int] = []
        for i, b in enumerate(branches):
            ids += b
            pos += list(range(P, P + len(b)))
            seg += [i] * len(b)
            last_idx.append(len(ids) - 1)
        S = len(ids)
        segt = torch.tensor(seg)
        j = torch.arange(S)[:, None]; k = torch.arange(S)[None, :]
        allow = (k <= j) & ((segt[k] == -1) | (segt[j] == segt[k]))
        min_val = torch.finfo(getattr(torch, "float32")).min
        mask = torch.where(allow, torch.zeros(1), torch.full((1,), min_val))[None, None].to(dev)
        with torch.no_grad():
            o = self.model(input_ids=torch.tensor([ids], device=dev),
                           attention_mask=mask.to(self.model.dtype),
                           position_ids=torch.tensor([pos], device=dev), use_cache=False)
        lg = o.logits[0].float()
        return [self._logodds(lg[t]) for t in last_idx]

    def score(self, rec: Dict[str, Any], **_ignored) -> Dict[str, Any]:
        prefix, branches, truncated = self.branch_ids(rec)
        if self.route == "flat":
            # one plain UNPADDED sequence per branch (batch 1). O(N*prefix) compute but O(L)
            # attention memory (sdpa/flash) -> handles arbitrarily long prefixes with NO
            # truncation, and no padding (so bf16 batched non-associativity never enters).
            out = self.score_solo(rec)
        elif self.route == "solo_batched":
            out = self._score_solo_batched(prefix, branches)
        elif self.route == "packed4d":
            out = self._score_packed4d(prefix, branches)
        else:
            out = self._score_cache(prefix, branches)
        return {"logits": out, "n_tokens": len(prefix) + sum(len(b) for b in branches),
                "n_prefix": len(prefix), "n_branches": len(branches), "truncated": truncated}

    def score_solo(self, rec: Dict[str, Any]) -> List[float]:
        """Reference: each option scored alone as one full sequence (no cache). Used by the
        packed-vs-solo self-test; must match ``score`` to bf16 noise."""
        import torch
        prefix, branches, _ = self.branch_ids(rec)
        out = []
        with torch.no_grad():
            for b in branches:
                lg = _forward_last(self, input_ids=torch.tensor([prefix + b], device=self.device),
                                   use_cache=False).logits[0, -1].float()
                out.append(self._logodds(lg))
        return out

    def score_mm(self, rec: Dict[str, Any], canonical_root: Optional[str] = None,
                 drop_images: bool = False) -> Dict[str, Any]:
        """Pointwise yes/no log-odds for a (possibly multimodal) record through the native
        processor. Each branch is an INDEPENDENT full sequence (prefix content incl. images +
        that branch's text); the verdict slot is the last real token. Order-invariant by
        construction. Render is byte-identical to the trainer (``pw_build_prefix_content`` /
        ``pw_branch_content`` imported from ``tod.train.pointwise_sft``). ``drop_images`` is the
        ablation arm (image parts -> "[image omitted]", no pixels)."""
        import torch
        from tod.train.pointwise_sft import pw_build_prefix_content, pw_branch_content, pw_processor_inputs
        labels, descs = pw_option_descs(rec)
        prefix_parts = pw_build_prefix_content(rec, canonical_root)
        has_image = any(p["type"] == "image" for p in prefix_parts)
        if self.processor is None or not has_image:
            out = self.score(rec); out["modality"] = "text"; return out
        contents = [pw_branch_content(prefix_parts, pw_branch_text(d)) for d in descs]
        if drop_images:  # ablation arm: image parts -> literal "[image omitted]" text, no pixels
            contents = [[({"type": "text", "text": "[image omitted]"} if p["type"] == "image" else p)
                         for p in parts] for parts in contents]
        out: List[float] = [0.0] * len(contents)
        ntok = 0
        for s in range(0, len(contents), self.branch_batch):
            chunk = contents[s: s + self.branch_batch]
            enc = pw_processor_inputs(self.processor, chunk)  # nested images, LEFT-padded
            enc = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in enc.items()}
            ntok = max(ntok, int(enc["input_ids"].shape[1]))
            with torch.no_grad():
                o = _forward_last(self, **enc, use_cache=False)
            lg = o.logits.float()  # left-padded -> verdict slot is the final column
            for j in range(len(chunk)):
                out[s + j] = self._logodds(lg[j, -1])
        return {"logits": out, "n_tokens": ntok, "n_branches": len(contents), "truncated": False,
                "modality": "text_ablation" if drop_images else "image"}


def pointwise_selftest(scorer: PointwiseScorer, recs: Sequence[Dict[str, Any]], tol: float = 0.05) -> Dict[str, Any]:
    """Packed(active-route)-vs-solo equality + branch-deletion invariance. Returns a record;
    the caller decides whether to refuse. ``score_solo`` (each option scored alone, no cache,
    one full sequence) is the ground truth; ``score`` uses the active route."""
    worst_solo = 0.0; worst_del = 0.0; n = 0
    for r in recs:
        a = scorer.score(r)["logits"]; b = scorer.score_solo(r)
        worst_solo = max(worst_solo, max(abs(x - y) for x, y in zip(a, b))); n += 1
        if len(r["labels"]) > 2:  # deleting an option must not move the others (IIA)
            sub = dict(r); sub["labels"] = list(r["labels"])[1:]
            c = scorer.score(sub)["logits"]
            worst_del = max(worst_del, max(abs(x - y) for x, y in zip(a[1:], c)))
    worst = max(worst_solo, worst_del)
    ok = worst <= tol
    print(f"[pointwise:selftest] route={scorer.route} n={n} max|route-solo|={worst_solo:.4f} "
          f"max|deletion|={worst_del:.4f} tol={tol} -> {'PASS' if ok else 'FAIL'}", flush=True)
    return {"n": n, "route": scorer.route, "max_abs_diff": worst,
            "max_abs_diff_vs_solo": worst_solo, "max_abs_diff_deletion": worst_del,
            "tol": tol, "pass": ok}


# ---------------------------------------------------------------- two-stage (recall + re-rank)
# Owner decision 2026-09-29: the picker is TWO-STAGE. N is NEVER capped.
#   Stage 1 = the pointwise readout (PointwiseScorer): every one of the N candidates is an
#     independent yes/no branch under a shared prefix; log-odds rank all N. Cheap recall over
#     unbounded N, order-invariant + IIA (branches never see each other).
#   Stage 2 = the comparative letter read (LetterLogitScorer) over the top-k=16 SHORTLIST only:
#     the k survivors are rendered as a lettered A..P list and read by the letter-logit head.
#     k is a beam width, not a cap on N.
# Final distribution over ALL N labels (owner's accepted scheme, documented here):
#   let p1 = stage-1 calibrated branch softmax over all N (softmax of log-odds / T1), and
#   let q  = stage-2 letter softmax over the k shortlisted labels (softmax of letter logits / T2).
#   shortlisted label j : p(j) = (Sum_{i in shortlist} p1(i)) * q(j)     [stage-2 re-rank,
#                                                                          scaled to the shortlist's
#                                                                          own stage-1 mass]
#   non-shortlisted   i : p(i) = p1(i)                                    [stage-1 tail mass]
#   These sum to 1 exactly (the non-shortlist p1 mass is 1 - Sum_shortlist p1). For N <= k the
#   shortlist is all labels, the shortlist mass is 1, and p == q == the PURE letter read on the
#   record (exact; verified in the self-test). The shortlist is built in ORIGINAL record order
#   among the selected labels, NOT stage-1 rank order, so stage-1's ranking cannot leak into the
#   letter positions the stage-2 read sees.
class RetrieverStage1:
    """Alternative stage-1 recall arm: a trained bi-encoder (``tod.train.retriever_sft``) instead
    of the pointwise 12B readout. ``score(rec)`` returns the SAME contract the two-stage machinery
    expects from ``PointwiseScorer.score`` -- ``logits`` (one score per label, here the cosine/dot
    retriever score), ``truncated``, ``n_branches``, ``n_tokens`` -- so the top-k shortlist and the
    stage-1 tail mass (softmax of these scores / T1) are computed unchanged. Candidate embeddings
    are cached by text (labels recur), so a record costs one query encode + cached lookups."""

    route = "retriever"

    def __init__(self, ckpt: str, device: str = "cuda", max_tokens: int = 8192, batch: int = 64):
        import torch
        from tod.train import retriever_sft as R
        self._R = R
        self._torch = torch
        t0 = time.time()
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.enc = R.BiEncoder.load(ckpt, device=device)
        self.device = torch.device(device)
        self.max_tokens = int(self.enc.config.get("max_tokens", max_tokens) or max_tokens)
        self.query_prefix = self.enc.config.get("query_prefix", "")
        self.cand_prefix = self.enc.config.get("cand_prefix", "")
        self.cand_max_tokens = int(self.enc.config.get("cand_max_tokens", 256))
        self.batch = batch
        self.tok = self.enc.tok
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        self._cand_cache: Dict[str, Any] = {}  # text -> normalized embedding (cpu tensor)
        self.load_s = time.time() - t0

    def _embed_query(self, state: str, question: str) -> Tuple[Any, bool]:
        ids, trunc = self._R.build_query_ids(self.tok, state, question, self.max_tokens,
                                             self.query_prefix)
        pad_ids, mask = self._R._pad([ids], self.pad_id)
        with self._torch.no_grad():
            e = self.enc.encode(pad_ids.to(self.device), mask.to(self.device))
        return e[0].float().cpu(), trunc, len(ids)

    def _embed_cands(self, texts: Sequence[str]) -> Any:
        miss = [t for t in texts if t not in self._cand_cache]
        for i in range(0, len(miss), self.batch):
            chunk = miss[i: i + self.batch]
            ids_list = [self._R.build_cand_ids(self.tok, t, self.cand_max_tokens, self.cand_prefix)
                        for t in chunk]
            pad_ids, mask = self._R._pad(ids_list, self.pad_id)
            with self._torch.no_grad():
                e = self.enc.encode(pad_ids.to(self.device), mask.to(self.device)).float().cpu()
            for t, v in zip(chunk, e):
                self._cand_cache[t] = v
        return self._torch.stack([self._cand_cache[t] for t in texts], dim=0)

    def score(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        state, question = self._R.render_query_parts(rec)
        _labels, descs = pw_option_descs(rec)
        q_emb, trunc, qlen = self._embed_query(state, question)
        c_emb = self._embed_cands(descs)
        scores = (c_emb @ q_emb).tolist()
        return {"logits": scores, "truncated": bool(trunc), "n_branches": len(descs),
                "n_tokens": int(qlen), "route": self.route}


class TwoStageScorer:
    """One model load; stage 1 = PointwiseScorer, stage 2 = LetterLogitScorer sharing it.

    ``adapter`` is stage 1's PEFT adapter (or none), ``adapter2`` is stage 2's; if ``adapter2``
    is absent stage 2 reuses ``adapter`` (or none). When they differ, ``adapter2`` is loaded as a
    named "stage2" adapter and the stages switch weights per forward: if stage 1 also has an
    adapter, ``set_adapter`` toggles between the two; if stage 1 has NO adapter (``--adapter2``
    alone), stage 2 sets "stage2" active while stage 1 runs the FROZEN base under
    ``model.disable_adapter()``."""

    def __init__(self, model_id: str, device: str = "cuda", dtype: str = "bfloat16",
                 attn: str = "sdpa", max_tokens: int = 16384, revision: Optional[str] = None,
                 adapter: Optional[str] = None, adapter2: Optional[str] = None,
                 branch_batch: int = 16, route: str = "cache", k: int = 16,
                 stage1_kind: str = "pointwise", retriever_ckpt: Optional[str] = None,
                 retriever_batch: int = 64, k_each: int = 8):
        self.model_id = model_id
        self.k = k
        self.k_each = k_each
        self.stage1_kind = stage1_kind
        # adapter2-only bookkeeping (set in the diff-adapter branches below): when stage 2 has an
        # adapter but stage 1 has none, stage 1 runs the FROZEN base under a held disable_adapter
        # context and stage 2 sets the "stage2" adapter active.
        self._adapter2_only = False
        self._disable_ctx = None
        if stage1_kind == "union":
            # BOTH stage-1 arms: the pointwise 12B readout (also the shared base for the stage-2
            # letter re-rank) AND the trained bi-encoder. The shortlist is their union (see
            # twostage_union_shortlist); the stage-1 tail mass is the average of the two calibrated
            # branch distributions (softmax(scores / T1_x), one T1 fit per arm). Stage 2 unchanged.
            if not retriever_ckpt:
                raise SystemExit("--stage1 union requires --retriever-ckpt")
            self.stage1 = PointwiseScorer(model_id, device=device, dtype=dtype, attn=attn,
                                          max_tokens=max_tokens, revision=revision, adapter=adapter,
                                          branch_batch=branch_batch, route=route)
            self.stage1_ret = RetrieverStage1(retriever_ckpt, device=device, max_tokens=max_tokens,
                                              batch=retriever_batch)
            self.stage2 = LetterLogitScorer.share(self.stage1, model_id)
            self.load_s = self.stage1.load_s + self.stage1_ret.load_s
            eff2 = adapter2 if adapter2 is not None else adapter
            self._diff_adapter = (eff2 != adapter)
            self._s1_name = None; self._s2_name = None
            if self._diff_adapter:
                self._attach_stage2_adapter(eff2)
            return
        if stage1_kind == "retriever":
            # Stage 1 is the trained bi-encoder; stage 2 (the letter re-rank) still needs the base
            # letter model, loaded STANDALONE (the retriever is a different model, so no sharing).
            if not retriever_ckpt:
                raise SystemExit("--stage1 retriever requires --retriever-ckpt")
            self.stage1 = RetrieverStage1(retriever_ckpt, device=device, max_tokens=max_tokens,
                                          batch=retriever_batch)
            eff2 = adapter2 if adapter2 is not None else adapter
            self.stage2 = LetterLogitScorer(model_id, device=device, dtype=dtype, attn=attn,
                                            max_tokens=max_tokens, revision=revision, adapter=eff2)
            self.load_s = self.stage1.load_s + self.stage2.load_s
            self._diff_adapter = False; self._s1_name = None; self._s2_name = None
            return
        self.stage1 = PointwiseScorer(model_id, device=device, dtype=dtype, attn=attn,
                                      max_tokens=max_tokens, revision=revision, adapter=adapter,
                                      branch_batch=branch_batch, route=route)
        self.stage2 = LetterLogitScorer.share(self.stage1, model_id)
        self.load_s = self.stage1.load_s
        eff2 = adapter2 if adapter2 is not None else adapter
        self._diff_adapter = (eff2 != adapter)
        self._s1_name = None; self._s2_name = None
        if self._diff_adapter:
            self._attach_stage2_adapter(eff2)

    def _attach_stage2_adapter(self, eff2: str) -> None:
        """Wire the stage-2 adapter onto the shared base (stage 2 REUSES stage 1's model via
        ``LetterLogitScorer.share``), for the two --adapter2 cases:

          * stage 1 already carries an adapter (``--adapter`` given) -> load ``eff2`` as a second
            named adapter "stage2"; ``set_adapter`` switches the active weights per stage.
          * stage 1 has NO adapter (``--adapter2`` alone) -> attach ``eff2`` as the named adapter
            "stage2" on the base. Stage-2 forwards set it active; stage-1 (pointwise) forwards run
            under ``with model.disable_adapter():`` so stage 1 is the FROZEN base model.
        """
        from peft import PeftModel
        m = self.stage1.model
        if isinstance(m, PeftModel):
            self._s1_name = m.active_adapter  # the stage-1 adapter loaded by PointwiseScorer
            m.load_adapter(eff2, adapter_name="stage2")
            self._s2_name = "stage2"
        else:
            pm = PeftModel.from_pretrained(m, eff2, adapter_name="stage2").eval()
            self.stage1.model = pm
            self.stage2.model = pm  # stage 2 shares stage 1's model object
            self._adapter2_only = True
            self._s2_name = "stage2"

    def _activate_stage1(self) -> None:
        if self._adapter2_only:
            # stage 1 = FROZEN base: hold a disable_adapter context open across stage-1 forwards
            # (closed when stage 2 is activated). Idempotent if already open.
            if self._disable_ctx is None:
                self._disable_ctx = self.stage1.model.disable_adapter()
                self._disable_ctx.__enter__()
        elif self._diff_adapter:
            self.stage1.model.set_adapter(self._s1_name)

    def _activate_stage2(self) -> None:
        if self._adapter2_only:
            if self._disable_ctx is not None:
                self._disable_ctx.__exit__(None, None, None)
                self._disable_ctx = None
            self.stage1.model.set_adapter(self._s2_name)
        elif self._diff_adapter:
            self.stage1.model.set_adapter(self._s2_name)


def record_has_image(rec: Dict[str, Any]) -> bool:
    """True when ``rec['state']`` is a segment list carrying at least one ``{"image": path}``."""
    state = rec.get("state")
    return isinstance(state, list) and any(isinstance(s, dict) and "image" in s for s in state)


def _score_stage(scorer: Any, rec: Dict[str, Any], canonical_root: Optional[str]) -> Dict[str, Any]:
    """Text rows -> the exact tokenizer path. Image rows -> ``score_mm`` (pixels through the
    native processor) when the scorer has one; never silently JSON-dump an image segment as text."""
    if record_has_image(rec) and getattr(scorer, "processor", None) is not None             and hasattr(scorer, "score_mm"):
        return scorer.score_mm(rec, canonical_root)
    return scorer.score(rec)


def twostage_score_record(ts: TwoStageScorer, rec: Dict[str, Any],
                          order_avg: bool = False,
                          canonical_root: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Score one record through both stages. Returns ``(s1, s2)``:
      s1 = PointwiseScorer.score(rec): per-branch yes/no log-odds over all N (+ n_tokens etc.),
      s2 = LetterLogitScorer read over the top-k shortlist, with the shortlist bookkeeping
           (``shortlist_idx`` = indices into rec['labels'] in ORIGINAL order; ``shortlist_labels``).
    Image rows (segment-list state with ``{"image": path}``) go through each stage's ``score_mm``
    so the pixels reach the vision tower; text rows take the byte-identical tokenizer path.
    """
    labels = [str(x) for x in rec["labels"]]
    n = len(labels)
    k = ts.k
    ts._activate_stage1()
    st = time.time()
    if getattr(ts, "stage1_kind", "pointwise") == "union":
        # run BOTH arms; shortlist = their union (see twostage_union_shortlist). s1 carries both
        # raw score vectors (over all N) so the tail mass can average the two calibrated dists.
        s1_pw = _score_stage(ts.stage1, rec, canonical_root)
        s1_ret = ts.stage1_ret.score(rec)
        shortlist_idx = twostage_union_shortlist(s1_pw["logits"], s1_ret["logits"], n, k, ts.k_each)
        s1 = {"kind": "union", "logits_pw": s1_pw["logits"], "logits_ret": s1_ret["logits"],
              "truncated": bool(s1_pw.get("truncated") or s1_ret.get("truncated")),
              "n_branches": s1_pw["n_branches"],
              "n_tokens": max(int(s1_pw["n_tokens"]), int(s1_ret["n_tokens"])),
              "elapsed_s": time.time() - st}
    else:
        s1 = _score_stage(ts.stage1, rec, canonical_root)
        s1["elapsed_s"] = time.time() - st
        logodds = s1["logits"]
        # top-k by log-odds; keep the selected in ORIGINAL record order (no rank leak into letters).
        order = sorted(range(n), key=lambda i: logodds[i], reverse=True)
        topk = set(order[: min(k, n)])
        shortlist_idx = [i for i in range(n) if i in topk]
    srec = dict(rec)
    srec["labels"] = [rec["labels"][i] for i in shortlist_idx]
    ts._activate_stage2()
    st2 = time.time()
    if order_avg:
        s2 = ts.stage2.score(srec, order_avg=order_avg)
    elif record_has_image(srec) and ts.stage2.processor is not None:
        s2 = ts.stage2.score_mm(srec, canonical_root)
    else:
        s2 = ts.stage2._score_one(srec)
    s2["elapsed_s"] = time.time() - st2
    s2["shortlist_idx"] = shortlist_idx
    s2["shortlist_labels"] = [labels[i] for i in shortlist_idx]
    return s1, s2


def twostage_final_probs(rec: Dict[str, Any], s1: Dict[str, Any], s2: Dict[str, Any],
                         T1: float, T2: float, T1_ret: Optional[float] = None) -> List[float]:
    """Final probability over ALL N labels, in rec['labels'] order (see the module block above:
    shortlist labels get the stage-2 re-rank scaled to their stage-1 mass; the rest keep stage-1
    tail mass; sums to 1; == pure stage-2 for N <= k).

    For ``--stage1 union`` the stage-1 distribution is the AVERAGE of the two arms' calibrated
    branch softmaxes -- softmax(logits_pw / T1) and softmax(logits_ret / T1_ret) (T1_ret defaults
    to T1). For N <= k the shortlist is all labels so the averaged tail is fully overwritten and
    the result is still exactly the pure stage-2 letter read."""
    n = len(rec["labels"])
    if s1.get("kind") == "union":
        p_pw = probs_from_logits(s1["logits_pw"], T1)
        p_rt = probs_from_logits(s1["logits_ret"], T1 if T1_ret is None else T1_ret)
        p1 = [0.5 * (a + b) for a, b in zip(p_pw, p_rt)]
    else:
        p1 = probs_from_logits(s1["logits"], T1)
    q = probs_from_logits(s2["logits"], T2)
    shortlist_idx = s2["shortlist_idx"]
    shortlist_mass = sum(p1[i] for i in shortlist_idx)
    final = list(p1)  # non-shortlisted labels keep their stage-1 calibrated mass
    for pos, i in enumerate(shortlist_idx):
        final[i] = shortlist_mass * q[pos]
    return final


def _rank_of_gold(rec: Dict[str, Any], scores: Sequence[float]) -> Optional[int]:
    """0-based rank of the gold label in a per-label score ordering (None if gold absent /
    length mismatch). Ties keep original record order (stable sort), as elsewhere."""
    labels = [str(x) for x in rec["labels"]]
    gold = str(rec.get("expected"))
    if gold not in labels or len(scores) != len(labels):
        return None
    gi = labels.index(gold)
    order = sorted(range(len(labels)), key=lambda i: scores[i], reverse=True)
    return order.index(gi)


def twostage_recall_rank(rec: Dict[str, Any], s1: Dict[str, Any]) -> Optional[int]:
    """0-based rank of the gold label in stage-1's log-odds ordering (None if gold absent)."""
    return _rank_of_gold(rec, s1["logits"])


def twostage_union_shortlist(pw_scores: Sequence[float], ret_scores: Sequence[float],
                             n: int, k: int, k_each: int) -> List[int]:
    """Indices (in ORIGINAL record order) of the size-min(k, n) UNION shortlist for --stage1 union:

      1. take the top-``k_each`` of the pointwise ranking and the top-``k_each`` of the retriever
         ranking and UNION them (dedup) -- at most 2*k_each, capped at k;
      2. if fewer than k, fill the remaining slots ALTERNATELY (pointwise, retriever, pointwise,
         ...) from each full ranking, skipping already-chosen indices, until k is reached;
      3. return the selected indices sorted ascending (ORIGINAL record order) so stage-1 ranking
         never leaks into the letter positions stage 2 sees.

    For N <= k every index is selected, so the shortlist is all labels in original order and the
    picker reduces exactly to the pure letter read (verified in the self-test / tests). Ties in
    either ranking keep original record order (stable sort), matching the single-arm picker."""
    k = min(int(k), n)
    ke = max(0, min(int(k_each), n))
    pw_rank = sorted(range(n), key=lambda i: pw_scores[i], reverse=True)
    ret_rank = sorted(range(n), key=lambda i: ret_scores[i], reverse=True)
    seen: set = set(); sel: List[int] = []

    def add(i: int) -> None:
        if i not in seen and len(sel) < k:
            seen.add(i); sel.append(i)

    for i in pw_rank[:ke]:
        add(i)
    for i in ret_rank[:ke]:
        add(i)
    pi = ri = 0; turn = 0  # 0 = pointwise arm's turn, 1 = retriever arm's turn
    while len(sel) < k and (pi < n or ri < n):
        if turn == 0:
            while pi < n and pw_rank[pi] in seen:
                pi += 1
            if pi < n:
                add(pw_rank[pi]); pi += 1
        else:
            while ri < n and ret_rank[ri] in seen:
                ri += 1
            if ri < n:
                add(ret_rank[ri]); ri += 1
        turn ^= 1
    return sorted(sel)


def twostage_selftest(ts: TwoStageScorer, recs: Sequence[Dict[str, Any]],
                      st_recs: Sequence[Dict[str, Any]], tol: float = 0.05) -> Dict[str, Any]:
    """Stage-1 pointwise self-test (exactness + IIA) PLUS: for a record with N <= k the two-stage
    final distribution equals the PURE letter read on that record (T1=T2=1)."""
    ts._activate_stage1()
    kind = getattr(ts, "stage1_kind", "pointwise")
    if kind == "retriever":
        # A trained bi-encoder is an APPROXIMATE recall arm: there is no exactness/IIA property to
        # prove (that was the pointwise readout's guarantee). Only the stage-2 reduce-to-letter
        # invariant below still holds and is checked.
        st = {"n": len(st_recs), "route": ts.stage1.route, "pass": True,
              "note": "stage-1 = trained retriever (approximate recall); no exactness/IIA proof"}
    elif kind == "union":
        # union's pointwise arm keeps the exactness/IIA guarantee (proved here); its retriever arm
        # is approximate. The reduce-to-letter invariant below still holds for the whole picker.
        st = pointwise_selftest(ts.stage1, st_recs, tol=tol)
        st["note"] = ("stage-1 = union(pointwise exact + trained retriever approx); the retriever "
                      "arm is not proven exact")
    else:
        st = pointwise_selftest(ts.stage1, st_recs, tol=tol)
    small = next((r for r in recs if 2 <= len(r["labels"]) <= ts.k), None)
    reduce_diff = None
    if small is not None:
        s1, s2 = twostage_score_record(ts, small, order_avg=False)
        final = twostage_final_probs(small, s1, s2, 1.0, 1.0)
        ts._activate_stage2()
        pure = probs_from_logits(ts.stage2._score_one(small)["logits"], 1.0)
        reduce_diff = max(abs(a - b) for a, b in zip(final, pure))
    st["twostage_reduce_to_letter_diff"] = reduce_diff
    st["twostage_reduce_ok"] = (reduce_diff is None) or (reduce_diff <= 1e-6)
    st["pass"] = bool(st["pass"]) and st["twostage_reduce_ok"]
    print(f"[twostage:selftest] N<=k reduces-to-letter max|diff|="
          f"{('n/a' if reduce_diff is None else f'{reduce_diff:.2e}')} -> "
          f"{'PASS' if st['twostage_reduce_ok'] else 'FAIL'}", flush=True)
    return st


# ----------------------------------------------------------------------------- probs/preds
def probs_from_logits(logits: Sequence[float], T: float) -> List[float]:
    z = [l / T for l in logits]
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


def pred_record(rec: Dict[str, Any], probs: List[float]) -> Dict[str, Any]:
    labels = [str(x) for x in rec["labels"]]
    qtype = rec["question"]["type"]
    pd = {lab: p for lab, p in zip(labels, probs)}
    # official argmax: ties -> lexicographically smallest label (jevbench/scoring.py)
    best = None; bp = -1.0
    for k in sorted(pd):
        if pd[k] > bp:
            best, bp = k, pd[k]
    out: Dict[str, Any] = {"type": qtype, "id": rec["id"], "probabilities": pd, "confidence": bp}
    if qtype == "noul":
        out["noul"] = pd.get("yes", 0.0)
    elif qtype == "score":
        out["score"] = float(best)  # argmax level (official rule; EV reported separately)
        out["ordinal_ev"] = sum(float(k) * v for k, v in pd.items())
    else:
        out["choice"] = best
    return out


def gold_record(rec: Dict[str, Any], dataset: str) -> Dict[str, Any]:
    labels = [str(x) for x in rec["labels"]]
    exp = rec.get("expected")
    exp = "" if exp is None else str(exp)
    gp = (rec.get("provenance") or {}).get("gold_probs")
    if isinstance(gp, dict) and set(map(str, gp)) == set(labels):
        probs = {str(k): float(v) for k, v in gp.items()}
    else:
        probs = {lab: (1.0 if lab == exp else 0.0) for lab in labels}
    return {"id": rec["id"], "type": rec["question"]["type"], "expected": exp, "probs": probs,
            "group": rec.get("group"), "family": rec.get("family"), "dataset": dataset}


def fit_temperature(items: List[Tuple[Dict[str, Any], List[float]]],
                    grid: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    """Brier-optimal single temperature (the repo convention, research/13 §4)."""
    if grid is None:
        grid = [round(0.05 * 1.08 ** i, 4) for i in range(120)]  # 0.05 .. ~500
        grid = [t for t in grid if t <= 50.0]
    best_T, best_b = 1.0, float("inf")
    curve = []
    for T in grid:
        b = 0.0
        for rec, lg in items:
            labels = [str(x) for x in rec["labels"]]
            p = probs_from_logits(lg, T)
            exp = str(rec.get("expected"))
            b += sum((pi - (1.0 if lab == exp else 0.0)) ** 2 for lab, pi in zip(labels, p))
        b /= max(1, len(items))
        curve.append((T, b))
        if b < best_b:
            best_T, best_b = T, b
    return {"T": best_T, "brier_at_T": best_b, "n": len(items),
            "brier_at_1": next(b for t, b in curve if abs(t - 1.0) < 0.04) if any(abs(t - 1.0) < 0.04 for t, _ in curve) else None}


# ----------------------------------------------------------------------------- main
def evaluate(scorer: LetterLogitScorer, root: str, tiers: Sequence[str], T: float,
             cache: Dict[str, Dict[str, Any]], order_avg: bool = False) -> Dict[str, Any]:
    report: Dict[str, Any] = {"tiers": {}, "predictions": {}}
    all_p: Dict[str, Dict] = {}; all_g: Dict[str, Dict] = {}
    for tier in tiers:
        recs = [r for r in load_records(os.path.join(root, f"{tier}.jsonl"))
                if not (r.get("provenance") or {}).get("exclude_reason") and r.get("expected") is not None]
        preds: Dict[str, Dict] = {}; golds: Dict[str, Dict] = {}
        t0 = time.time()
        for r in recs:
            key = r["id"]
            if key not in cache:
                cache[key] = scorer.score(r, order_avg=order_avg)
            p = probs_from_logits(cache[key]["logits"], T)
            preds[key] = pred_record(r, p)
            golds[key] = gold_record(r, f"jevbench_{tier}")
        rep = metrics.full_report(preds, golds, bins=10)
        rep["elapsed_s"] = round(time.time() - t0, 1)
        rep["n_truncated"] = sum(1 for r in recs if cache[r["id"]]["truncated"])
        rep["max_tokens_seen"] = max(cache[r["id"]]["n_tokens"] for r in recs) if recs else 0
        report["tiers"][tier] = rep
        report["predictions"].update({k: {"type": v["type"], "prediction": metrics.predicted_label(v),
                                          "confidence": v["confidence"]} for k, v in preds.items()})
        all_p.update(preds); all_g.update(golds)
        print(f"[letter_logit] {scorer.model_id} T={T:.3g} tier={tier} n={rep['n']} "
              f"acc={rep['accuracy']:.4f} ece={rep['ece']:.4f} brier={rep['brier']:.4f}", flush=True)
    report["overall"] = metrics.full_report(all_p, all_g, bins=10)
    ov = report["overall"]
    print(f"[letter_logit] {scorer.model_id} T={T:.3g} OVERALL n={ov['n']} acc={ov['accuracy']:.4f} "
          f"ece={ov['ece']:.4f} ece_db={ov['ece_debiased']:.4f} brier={ov['brier']:.4f}", flush=True)
    return report


def evaluate_jsonl(scorer: "PointwiseScorer", path: str, T: float,
                   cache: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Score an arbitrary canonical-schema jsonl (any N labels) with the same report layout as
    ``evaluate``. Golds carry each row's dataset (source.dataset/meta.dataset/dataset) so the
    per-dataset OOD breakdown in ``metrics.full_report`` is populated. No <=26 filter."""
    recs = [r for r in load_records(path)
            if r.get("expected") is not None and not (r.get("provenance") or {}).get("exclude_reason")]
    preds: Dict[str, Dict] = {}; golds: Dict[str, Dict] = {}
    t0 = time.time()
    for r in recs:
        key = r["id"]
        if key not in cache:
            st = time.time()
            sc = scorer.score(r)
            sc["elapsed_s"] = time.time() - st
            cache[key] = sc
        p = probs_from_logits(cache[key]["logits"], T)
        preds[key] = pred_record(r, p)
        golds[key] = gold_record(r, _row_dataset(r) or os.path.basename(path))
    rep = metrics.full_report(preds, golds, bins=10)
    rep["elapsed_s"] = round(time.time() - t0, 1)
    rep["n_truncated"] = sum(1 for r in recs if cache[r["id"]]["truncated"])
    rep["max_tokens_seen"] = max((cache[r["id"]]["n_tokens"] for r in recs), default=0)
    rep["max_branches_seen"] = max((cache[r["id"]]["n_branches"] for r in recs), default=0)
    # s/item bucketed by N (branch count) -- shows the readout does not collapse as N grows.
    by_n: Dict[int, List[float]] = {}
    for r in recs:
        c = cache[r["id"]]
        if "elapsed_s" in c:
            by_n.setdefault(c["n_branches"], []).append(c["elapsed_s"])
    rep["sec_per_item_by_n"] = {str(k): round(sum(v) / len(v), 4) for k, v in sorted(by_n.items())}
    rep["predictions"] = {k: {"type": v["type"], "prediction": metrics.predicted_label(v),
                              "confidence": v["confidence"]} for k, v in preds.items()}
    print(f"[letter_logit:jsonl] {os.path.basename(path)} n={rep['n']} acc={rep['accuracy']:.4f} "
          f"ece={rep['ece']:.4f} brier={rep['brier']:.4f} maxN={rep['max_branches_seen']}", flush=True)
    return rep


def evaluate_twostage(ts: TwoStageScorer, recs: Sequence[Dict[str, Any]], dataset_label: str,
                      T1: float, T2: float, cache1: Dict[str, Dict[str, Any]],
                      cache2: Dict[str, Dict[str, Any]], order_avg: bool = False,
                      k_list: Sequence[int] = (8, 16, 24),
                      all_p: Optional[Dict[str, Dict]] = None,
                      all_g: Optional[Dict[str, Dict]] = None,
                      T1_ret: Optional[float] = None) -> Dict[str, Any]:
    """Two-stage report for a list of records, in the same layout as ``evaluate`` / ``evaluate_jsonl``.
    Adds ``recall_at_k`` (fraction of records whose gold falls in stage-1's top-k, one entry per k
    in ``k_list``) as a FIRST-CLASS output -- the owner's stage-1 recall gate.

    For ``--stage1 union`` ``recall_at_k`` (and ``recall_at_k_by_dataset``) instead carry a
    ``pointwise`` / ``retriever`` / ``union`` sub-key each: the two arms' recall alone (gold in
    that arm's top-kk) and the union picker's recall (gold in the size-kk union shortlist, built
    by ``twostage_union_shortlist`` with the same ``--k-each`` -- so ``union`` at kk==k is exactly
    the shortlist recall the owner gates on).

    Cache/pred keys are a UNIQUE per-record-instance key ``"<dataset_label>#<pos>"`` (NOT the raw
    record id): SalesBench tiers reuse the same ``id`` across records with DIFFERENT candidate sets
    (e.g. the same conversation truncated to N=1 vs N=5), so an id-keyed cache would serve a stale
    stage-1 vector of the wrong length. The unique key also makes ``full_report`` (which pairs
    preds/golds by key) count every record instead of collapsing same-id rows. ``all_p``/``all_g``,
    if given, are populated with the same keys for an aggregate report across sets."""
    preds: Dict[str, Dict] = {}; golds: Dict[str, Dict] = {}
    is_union = getattr(ts, "stage1_kind", "pointwise") == "union"
    arms = ("pointwise", "retriever", "union")
    kk_list = [int(kk) for kk in k_list]
    if is_union:
        recall_counts = {a: {kk: 0 for kk in kk_list} for a in arms}
        ds_counts_u: Dict[str, Dict[str, Dict[int, int]]] = {}
    else:
        recall_counts = {kk: 0 for kk in kk_list}
        ds_counts: Dict[str, Dict[int, int]] = {}
    recall_n = 0
    # per-set (per-dataset) recall is a FIRST-CLASS output: on the high-N sets the owner's gate is
    # per set (banking77/clinc150/lexglue/tasksource), not pooled -- one weak set flips the
    # trained-bi-encoder decision. Grouped by each row's own dataset.
    ds_n: Dict[str, int] = {}
    n_trunc = 0; max_branches = 0; max_tokens = 0
    t0 = time.time()
    for pos, r in enumerate(recs):
        ckey = f"{dataset_label}#{pos}"
        if ckey not in cache1:
            s1, s2 = twostage_score_record(ts, r, order_avg=order_avg)
            cache1[ckey] = s1; cache2[ckey] = s2
        s1 = cache1[ckey]; s2 = cache2[ckey]
        final = twostage_final_probs(r, s1, s2, T1, T2, T1_ret=T1_ret)
        preds[ckey] = pred_record(r, final)
        golds[ckey] = gold_record(r, _row_dataset(r) or dataset_label)
        if all_p is not None:
            all_p[ckey] = preds[ckey]; all_g[ckey] = golds[ckey]
        if is_union:
            labels = [str(x) for x in r["labels"]]; gold = str(r.get("expected"))
            n = len(labels)
            if gold in labels and len(s1["logits_pw"]) == n and len(s1["logits_ret"]) == n:
                gi = labels.index(gold)
                rank_pw = _rank_of_gold(r, s1["logits_pw"])
                rank_ret = _rank_of_gold(r, s1["logits_ret"])
                ds = _row_dataset(r) or dataset_label
                recall_n += 1
                ds_n[ds] = ds_n.get(ds, 0) + 1
                ds_counts_u.setdefault(ds, {a: {kk: 0 for kk in kk_list} for a in arms})
                for kk in kk_list:
                    usl = twostage_union_shortlist(s1["logits_pw"], s1["logits_ret"], n, kk, ts.k_each)
                    hits = {"pointwise": rank_pw < kk, "retriever": rank_ret < kk,
                            "union": gi in usl}
                    for a in arms:
                        if hits[a]:
                            recall_counts[a][kk] += 1
                            ds_counts_u[ds][a][kk] += 1
        else:
            rank = twostage_recall_rank(r, s1)
            if rank is not None:
                ds = _row_dataset(r) or dataset_label
                recall_n += 1
                ds_n[ds] = ds_n.get(ds, 0) + 1
                ds_counts.setdefault(ds, {kk: 0 for kk in kk_list})
                for kk in kk_list:
                    if rank < kk:
                        recall_counts[kk] += 1
                        ds_counts[ds][kk] += 1
        n_trunc += int(bool(s1["truncated"]))
        max_branches = max(max_branches, s1["n_branches"]); max_tokens = max(max_tokens, s1["n_tokens"])
    rep = metrics.full_report(preds, golds, bins=10)
    rep["elapsed_s"] = round(time.time() - t0, 1)
    if is_union:
        rep["recall_at_k"] = {
            a: {str(kk): (recall_counts[a][kk] / recall_n if recall_n else None) for kk in kk_list}
            for a in arms}
        rep["recall_at_k_by_dataset"] = {
            ds: {"n": ds_n[ds],
                 **{a: {str(kk): ds_counts_u[ds][a][kk] / ds_n[ds] for kk in kk_list} for a in arms}}
            for ds in sorted(ds_n)}
        rep["k_each"] = ts.k_each
    else:
        rep["recall_at_k"] = {str(kk): (recall_counts[kk] / recall_n if recall_n else None)
                              for kk in kk_list}
        rep["recall_at_k_by_dataset"] = {
            ds: {"n": ds_n[ds], **{str(kk): ds_counts[ds][kk] / ds_n[ds] for kk in kk_list}}
            for ds in sorted(ds_n)}
    rep["recall_n"] = recall_n
    rep["k"] = ts.k
    rep["n_truncated"] = n_trunc
    rep["max_branches_seen"] = max_branches
    rep["max_tokens_seen"] = max_tokens
    rep["predictions"] = {k: {"type": v["type"], "prediction": metrics.predicted_label(v),
                              "confidence": v["confidence"]} for k, v in preds.items()}
    rvals = rep["recall_at_k"]["union"] if is_union else rep["recall_at_k"]
    rk = " ".join(f"r@{kk}={rvals[str(kk)]:.4f}" if rvals[str(kk)] is not None else f"r@{kk}=n/a"
                  for kk in kk_list)
    print(f"[twostage] {dataset_label} n={rep['n']} acc={rep['accuracy']:.4f} ece={rep['ece']:.4f} "
          f"brier={rep['brier']:.4f} maxN={rep['max_branches_seen']} "
          f"{'union ' if is_union else ''}{rk}", flush=True)
    return rep


def _twostage_calib(ts: TwoStageScorer, path: str, n: int, order_avg: bool = False, seed: int = 0):
    """Calibration items for BOTH temperatures, from one pass over the calib rows:
      items1 = (rec, stage-1 branch log-odds over all N)  -> fits T1 (branch softmax),
      items2 = (shortlist-rec, stage-2 letter logits)      -> fits T2, but ONLY on rows whose gold
               survives into the shortlist (T2 calibrates the re-rank, so a row where stage 1 lost
               the gold carries no learnable stage-2 target).
    For ``--stage1 union`` items1 holds the POINTWISE arm's scores and a third list ``items1_ret``
    (else ``None``) holds the RETRIEVER arm's scores, so one temperature is fit per arm."""
    import random
    is_union = getattr(ts, "stage1_kind", "pointwise") == "union"
    crecs = [r for r in load_records(path) if r.get("expected") is not None and r.get("labels")]
    random.Random(seed).shuffle(crecs)
    crecs = crecs[:n]
    items1: List[Tuple[Dict[str, Any], List[float]]] = []
    items2: List[Tuple[Dict[str, Any], List[float]]] = []
    items1_ret: Optional[List[Tuple[Dict[str, Any], List[float]]]] = [] if is_union else None
    for r in crecs:
        s1, s2 = twostage_score_record(ts, r, order_avg=order_avg)
        if is_union:
            items1.append((r, s1["logits_pw"]))
            items1_ret.append((r, s1["logits_ret"]))
        else:
            items1.append((r, s1["logits"]))
        sl = [str(x) for x in s2["shortlist_labels"]]
        if str(r["expected"]) in sl:
            srec = dict(r); srec["labels"] = s2["shortlist_labels"]
            items2.append((srec, s2["logits"]))
    return items1, items2, items1_ret


def _load_mm_rows(canonical_root: str, dataset: str, split: str, n: int) -> List[Dict[str, Any]]:
    """First ``n`` scorable image rows of a canonical split (``expected`` set, LIST state)."""
    from ..data.dataset import iter_canonical_rows
    shard = os.path.join(canonical_root, dataset, f"{split}.jsonl.zst")
    if not os.path.exists(shard):
        shard = shard[:-4]  # allow uncompressed .jsonl
    if not os.path.exists(shard):
        return []
    out: List[Dict[str, Any]] = []
    for rec in iter_canonical_rows(shard):
        if rec.get("expected") is None or not (0 < len(rec.get("labels", [])) <= len(LETTERS)):
            continue
        if not isinstance(rec.get("state"), list):
            continue  # multimodal eval scores image rows only
        out.append(rec)
        if n and len(out) >= n:
            break
    return out


def evaluate_multimodal(scorer: LetterLogitScorer, canonical_root: str, datasets: Sequence[str],
                        split: str, n: int, T: float = 1.0, n_boot: int = 1000,
                        bins: int = 10) -> Dict[str, Any]:
    """Score each image dataset with AND without images and report the vision value-prop.

    Reuses ``tod.eval.multimodal.bootstrap_delta_acc`` (item bootstrap on Δacc) and the
    "vision earns its keep" criterion (Δacc>0 with CI low bound >0, per dataset)."""
    from .multimodal import bootstrap_delta_acc
    report: Dict[str, Any] = {"split": split, "n_per_dataset": n, "canonical_root": canonical_root,
                              "datasets": list(datasets), "per_dataset": {}}
    for ds in datasets:
        recs = _load_mm_rows(canonical_root, ds, split, n)
        if not recs:
            report["per_dataset"][ds] = {"n": 0, "note": f"no {split} shard / no image rows"}
            print(f"[letter_logit:mm] {ds}: no {split} rows", flush=True)
            continue
        golds: Dict[str, Dict] = {}
        with_p: Dict[str, Dict] = {}
        without_p: Dict[str, Dict] = {}
        t0 = time.time()
        for r in recs:
            gid = r["id"]
            golds[gid] = gold_record(r, f"mm_{ds}")
            sw = scorer.score_mm(r, canonical_root, drop_images=False)
            so = scorer.score_mm(r, canonical_root, drop_images=True)
            with_p[gid] = pred_record(r, probs_from_logits(sw["logits"], T))
            without_p[gid] = pred_record(r, probs_from_logits(so["logits"], T))
        boot = bootstrap_delta_acc(with_p, without_p, golds, n_boot=n_boot)
        rep = {
            "n": len(golds),
            "acc_with": metrics.accuracy(with_p, golds),
            "acc_without": metrics.accuracy(without_p, golds),
            "delta_accuracy": boot["delta"],
            "delta_accuracy_ci": [boot["lo"], boot["hi"]],
            "brier_with": metrics.brier(with_p, golds),
            "brier_without": metrics.brier(without_p, golds),
            "ece_with": metrics.ece(with_p, golds, bins),
            "vision_earns_keep": bool(boot["lo"] > 0.0),
            "elapsed_s": round(time.time() - t0, 1),
        }
        report["per_dataset"][ds] = rep
        print(f"[letter_logit:mm] {ds} n={rep['n']} acc_with={rep['acc_with']:.4f} "
              f"acc_without={rep['acc_without']:.4f} dacc={rep['delta_accuracy']:+.4f} "
              f"ci=[{boot['lo']:+.4f},{boot['hi']:+.4f}] earns_keep={rep['vision_earns_keep']}",
              flush=True)
    scored = [d for d in report["per_dataset"].values() if d.get("n")]
    deltas = [d["delta_accuracy"] for d in scored]
    report["summary"] = {
        "mean_delta_accuracy": (sum(deltas) / len(deltas)) if deltas else 0.0,
        "vision_earns_keep_all": bool(scored) and all(d.get("vision_earns_keep") for d in scored),
    }
    report["vision_earns_keep_all"] = report["summary"]["vision_earns_keep_all"]
    return report


def _pointwise_calib_items(scorer: "PointwiseScorer", path: str, n: int, seed: int = 0):
    """Calibration (rec, branch-log-odds) pairs -- NO <=26 filter (that cap is the whole point
    of the pointwise readout)."""
    import random
    crecs = [r for r in load_records(path) if r.get("expected") is not None and r.get("labels")]
    random.Random(seed).shuffle(crecs)
    crecs = crecs[:n]
    return [(r, scorer.score(r)["logits"]) for r in crecs]


def _main_pointwise(args) -> Dict[str, Any]:
    scorer = PointwiseScorer(args.model, device=args.device, attn=args.attn,
                             max_tokens=args.max_tokens, revision=args.revision,
                             adapter=args.adapter, branch_batch=args.branch_batch,
                             route=args.pw_route, dtype=args.dtype)
    print(f"[letter_logit] loaded {args.model} in {scorer.load_s:.0f}s (pointwise); "
          f"chat_template={scorer.has_chat} route={scorer.route} dtype={args.dtype}", flush=True)

    # self-test BEFORE any numbers. Prefer real calibration rows; fall back to a tier.
    src = args.calib_jsonl or os.path.join(args.jevbench_root, f"{args.tiers[0]}.jsonl")
    pool = [r for r in load_records(src) if r.get("expected") is not None and r.get("labels")]
    # Exactness / IIA are N-independent, and the reference (score_solo) costs ONE full forward
    # PER branch -- so high-N selftest rows are needlessly expensive. Prefer small-N rows, but
    # keep a few with >2 labels so the deletion-invariance check still runs.
    import random
    lown = [r for r in pool if 3 <= len(r["labels"]) <= 8] or pool
    random.Random(1).shuffle(lown)
    st_recs = lown[: max(1, args.selftest)]
    st = pointwise_selftest(scorer, st_recs, tol=0.05)
    if not st["pass"] and scorer.route == "cache":
        print("[pointwise:selftest] cache route not exact -> falling back to solo_batched", flush=True)
        scorer.route = "solo_batched"
        st = pointwise_selftest(scorer, st_recs, tol=0.05)
    if not st["pass"]:
        raise SystemExit("pointwise readout is NOT exact on this model/attention path; refusing to report numbers")

    result: Dict[str, Any] = {
        "model": args.model, "revision": args.revision, "adapter": args.adapter,
        "prompt_version": PW_PROMPT_VERSION, "max_tokens": args.max_tokens,
        "readout": "pointwise.v1", "pw_route": scorer.route, "branch_batch": args.branch_batch,
        "dtype": args.dtype, "selftest": st,
        "readout_desc": ("per-option yes/no log-odds (logsumexp yes-variants minus no-variants) "
                         "under a shared prefix; softmax over branches / T; unbounded N"),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if args.multimodal:
        if scorer.processor is None:
            print(f"[letter_logit] WARNING: {args.model} has no multimodal processor; "
                  f"image rows will score as text (no pixels).", flush=True)
        result["multimodal"] = evaluate_multimodal(
            scorer, args.canonical_root, args.mm_datasets, args.mm_split, args.mm_n,
            T=1.0, n_boot=args.n_boot)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=1)
        print(f"[letter_logit] wrote {args.out} (pointwise multimodal); "
              f"vision_earns_keep_all={result['multimodal']['vision_earns_keep_all']}", flush=True)
        return result

    cache: Dict[str, Dict[str, Any]] = {}
    is_jsonl = bool(args.jsonl)
    if is_jsonl:
        result["jsonl"] = args.jsonl
        result["raw"] = evaluate_jsonl(scorer, args.jsonl, 1.0, cache)
    else:
        result["raw"] = evaluate(scorer, args.jevbench_root, args.tiers, 1.0, cache)
    T = None
    if args.temperature is not None:
        T = float(args.temperature)
        result["temperature"] = {"T": T, "source": "fixed (--temperature)"}
        print(f"[letter_logit] using fixed temperature T={T}", flush=True)
    elif args.calib_jsonl:
        t0 = time.time()
        items = _pointwise_calib_items(scorer, args.calib_jsonl, args.calib_n)
        fit = fit_temperature(items)
        fit["elapsed_s"] = round(time.time() - t0, 1); fit["source"] = args.calib_jsonl
        result["temperature"] = fit
        T = fit["T"]
        print(f"[letter_logit] temperature fit: {fit}", flush=True)
    if T is not None:
        if is_jsonl:
            result["calibrated"] = evaluate_jsonl(scorer, args.jsonl, T, cache)
        else:
            result["calibrated"] = evaluate(scorer, args.jevbench_root, args.tiers, T, cache)
    result["cache"] = cache
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    print(f"[letter_logit] wrote {args.out} (pointwise, route={scorer.route})", flush=True)
    return result


def _main_twostage(args) -> Dict[str, Any]:
    k_list = sorted(set([8, 16, 24, int(args.k)]))
    ts = TwoStageScorer(args.model, device=args.device, dtype=args.dtype, attn=args.attn,
                        max_tokens=args.max_tokens, revision=args.revision, adapter=args.adapter,
                        adapter2=args.adapter2, branch_batch=args.branch_batch,
                        route=args.pw_route, k=int(args.k), stage1_kind=args.stage1,
                        retriever_ckpt=args.retriever_ckpt, retriever_batch=args.retriever_batch,
                        k_each=int(args.k_each))
    is_union = args.stage1 == "union"
    print(f"[letter_logit] loaded {args.model} in {ts.load_s:.0f}s (twostage k={args.k}"
          f"{f' k_each={args.k_each}' if is_union else ''}); "
          f"stage1={args.stage1} stage1_route={ts.stage1.route} "
          f"retriever_ckpt={args.retriever_ckpt} adapter={args.adapter} adapter2={args.adapter2} "
          f"dtype={args.dtype}", flush=True)

    # self-test BEFORE any numbers: stage-1 exactness/IIA + N<=k reduces to the pure letter read.
    src = args.calib_jsonl or args.jsonl or os.path.join(args.jevbench_root, f"{args.tiers[0]}.jsonl")
    pool = [r for r in load_records(src) if r.get("expected") is not None and r.get("labels")]
    import random
    lown = [r for r in pool if 3 <= len(r["labels"]) <= 8] or pool
    random.Random(1).shuffle(lown)
    st_recs = lown[: max(1, args.selftest)]
    st = twostage_selftest(ts, pool, st_recs, tol=0.05)
    if st.get("max_abs_diff", 0.0) > 0.05 and ts.stage1.route == "cache":
        print("[twostage:selftest] stage-1 cache route not exact -> falling back to solo_batched", flush=True)
        ts.stage1.route = "solo_batched"
        st = twostage_selftest(ts, pool, st_recs, tol=0.05)
    if not st["pass"]:
        raise SystemExit("two-stage readout is NOT exact on this model/attention path; refusing to report numbers")

    result: Dict[str, Any] = {
        "model": args.model, "revision": args.revision, "adapter": args.adapter,
        "adapter2": args.adapter2, "readout": "twostage.v1", "k": int(args.k),
        "stage1": args.stage1, "retriever_ckpt": args.retriever_ckpt,
        "prompt_version": {"stage1": PW_PROMPT_VERSION, "stage2": PROMPT_VERSION},
        "max_tokens": args.max_tokens, "pw_route": ts.stage1.route, "dtype": args.dtype,
        "option_style": args.option_style, "order_avg": args.order_avg, "selftest": st,
        "readout_desc": ("stage 1 = pointwise yes/no log-odds over all N (recall); stage 2 = letter "
                         "read over the top-k shortlist (re-rank); final = shortlist stage-1 mass * "
                         "stage-2 softmax on shortlisted labels, stage-1 tail mass on the rest "
                         "(== pure letter read for N<=k)") if not is_union else (
            "stage 1 = UNION of two recall arms (pointwise yes/no log-odds + trained bi-encoder): "
            "shortlist = union of each arm's top-k_each, filled alternately to k in original order; "
            "stage-1 tail mass = average of the two calibrated branch softmaxes (one T1 per arm); "
            "stage 2 = letter re-rank over the shortlist (== pure letter read for N<=k)"),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if is_union:
        result["k_each"] = int(args.k_each)

    cache1: Dict[str, Dict[str, Any]] = {}; cache2: Dict[str, Dict[str, Any]] = {}
    is_jsonl = bool(args.jsonl)

    def _pool_recall(report: Dict[str, Any]) -> Any:
        """Overall recall@k pooled over tiers (weighted by each tier's recall_n). Union keeps the
        pointwise/retriever/union sub-key structure."""
        def pool_flat(get) -> Dict[str, Any]:
            out = {}
            for kk in k_list:
                num = sum(get(t, kk) * report["tiers"][t]["recall_n"] for t in args.tiers
                          if get(t, kk) is not None)
                den = sum(report["tiers"][t]["recall_n"] for t in args.tiers)
                out[str(int(kk))] = (num / den) if den else None
            return out
        if is_union:
            return {a: pool_flat(lambda t, kk, a=a: report["tiers"][t]["recall_at_k"][a][str(int(kk))])
                    for a in ("pointwise", "retriever", "union")}
        return pool_flat(lambda t, kk: report["tiers"][t]["recall_at_k"][str(int(kk))])

    def eval_all(T1: float, T2: float, T1_ret: float = 1.0) -> Dict[str, Any]:
        if is_jsonl:
            recs = [r for r in load_records(args.jsonl)
                    if r.get("expected") is not None and not (r.get("provenance") or {}).get("exclude_reason")]
            label = (_row_dataset(recs[0]) if recs else None) or os.path.basename(args.jsonl)
            return {"jsonl": args.jsonl,
                    "report": evaluate_twostage(ts, recs, label, T1, T2, cache1, cache2,
                                                args.order_avg, k_list, T1_ret=T1_ret)}
        report: Dict[str, Any] = {"tiers": {}}
        all_p: Dict[str, Dict] = {}; all_g: Dict[str, Dict] = {}
        for tier in args.tiers:
            recs = [r for r in load_records(os.path.join(args.jevbench_root, f"{tier}.jsonl"))
                    if r.get("expected") is not None and not (r.get("provenance") or {}).get("exclude_reason")]
            rep = evaluate_twostage(ts, recs, f"jevbench_{tier}", T1, T2, cache1, cache2,
                                    args.order_avg, k_list, all_p=all_p, all_g=all_g, T1_ret=T1_ret)
            report["tiers"][tier] = rep
        ov = metrics.full_report(all_p, all_g, bins=10)
        ov_recall = _pool_recall(report)
        ov["recall_at_k"] = ov_recall
        report["overall"] = ov
        r16 = ov_recall["union"].get("16") if is_union else ov_recall.get("16")
        print(f"[twostage] OVERALL n={ov['n']} acc={ov['accuracy']:.4f} ece={ov['ece']:.4f} "
              f"ece_db={ov['ece_debiased']:.4f} brier={ov['brier']:.4f} "
              f"{'union ' if is_union else ''}r@16={r16}", flush=True)
        return report

    result["raw"] = eval_all(1.0, 1.0, 1.0)

    T1 = T2 = T1_ret = 1.0
    have_fixed = (args.t1 is not None and args.t2 is not None
                  and (args.t1_ret is not None or not is_union))
    if args.calib_jsonl and not have_fixed:
        t0 = time.time()
        items1, items2, items1_ret = _twostage_calib(ts, args.calib_jsonl, args.calib_n, args.order_avg)
        fit1 = fit_temperature(items1)
        fit2 = fit_temperature(items2) if items2 else {"T": 1.0, "n": 0, "note": "no gold-in-shortlist calib rows"}
        T1 = fit1["T"]; T2 = fit2["T"]
        result["temperature"] = {"T1": fit1, "T2": fit2, "source": args.calib_jsonl,
                                 "elapsed_s": round(time.time() - t0, 1)}
        if is_union:
            fit1_ret = fit_temperature(items1_ret)
            T1_ret = fit1_ret["T"]
            result["temperature"]["T1_ret"] = fit1_ret
        print(f"[twostage] temperature fit: T1={T1} T2={T2}"
              f"{f' T1_ret={T1_ret}' if is_union else ''} "
              f"(calib n1={fit1.get('n')} n2={fit2.get('n')})", flush=True)
    if args.t1 is not None:
        T1 = float(args.t1)
    if args.t2 is not None:
        T2 = float(args.t2)
    if args.t1_ret is not None:
        T1_ret = float(args.t1_ret)
    if (args.t1 is not None) or (args.t2 is not None) or (args.t1_ret is not None):
        result.setdefault("temperature", {})["T1_fixed"] = args.t1
        result["temperature"]["T2_fixed"] = args.t2
        if is_union:
            result["temperature"]["T1_ret_fixed"] = args.t1_ret
    if T1 != 1.0 or T2 != 1.0 or T1_ret != 1.0:
        result["calibrated"] = eval_all(T1, T2, T1_ret)
        result["temperature_used"] = {"T1": T1, "T2": T2}
        if is_union:
            result["temperature_used"]["T1_ret"] = T1_ret

    result["stage1"] = cache1
    result["stage2"] = cache2
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    print(f"[letter_logit] wrote {args.out} (twostage k={args.k}, route={ts.stage1.route})", flush=True)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser("tod.eval.letter_logit")
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--adapter", default=None, help="optional PEFT adapter dir")
    ap.add_argument("--jevbench-root", default="/data/raw/jevbench/datasets/public")
    ap.add_argument("--tiers", nargs="+", default=["easy", "original", "hard"])
    ap.add_argument("--calib-jsonl", default=None,
                    help="off-benchmark records (canonical/JevBench schema) to fit the temperature")
    ap.add_argument("--calib-n", type=int, default=600)
    ap.add_argument("--max-tokens", type=int, default=49152,
                    help="prefix budget (middle-truncation, recorded). Owner rule: never cap below "
                         "49152. Eager attention is O(L^2) memory, so prefixes beyond ~13k need "
                         "--attn chunked_eager (query-chunked exact eager) to fit; sdpa is NOT exact "
                         "on this model (measured ~5-8 log-odds drift vs eager, even unpadded).")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--device", default="cuda",
                    help="cuda | cuda:N | cpu | auto (shard the 12B across all visible GPUs with device_map=auto; use when one card cannot hold fp32, e.g. 2x40 GB)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump-prompt", action="store_true", help="print one rendered prompt and exit")
    ap.add_argument("--option-style", choices=["label_desc", "desc"], default="label_desc")
    ap.add_argument("--order-avg", action="store_true", help="average forward + reversed option order")
    # multimodal ("does vision earn its keep") mode -- scores canonical image rows with/without
    # images through the native processor; the public JevBench tiers always stay text-only.
    ap.add_argument("--multimodal", action="store_true",
                    help="score canonical image datasets with/without images instead of the tiers")
    ap.add_argument("--mm-datasets", nargs="+", default=["scienceqa", "ai2d", "chartqa"])
    ap.add_argument("--mm-split", default="val", choices=["val", "test"])
    ap.add_argument("--mm-n", type=int, default=200, help="rows per dataset (0=all)")
    ap.add_argument("--canonical-root", default="/data/canonical")
    ap.add_argument("--n-boot", type=int, default=1000)
    # pointwise (unbounded-N) readout: prefix = state+question (NO option list); each option a
    # branch "description -> yes/no slot"; choice = softmax over per-branch yes/no log-odds / T.
    ap.add_argument("--readout", choices=["letter", "pointwise", "twostage"], default="letter",
                    help="letter (default; reproduces the committed board) | pointwise (unbounded N) "
                         "| twostage (pointwise recall over all N -> letter re-rank of the top-k)")
    ap.add_argument("--k", type=int, default=16,
                    help="twostage: shortlist / beam width for the stage-2 letter re-rank (N never "
                         "capped). N<=k reduces exactly to the pure letter read; set --k>=maxN for "
                         "the full-list-letter-vs-twostage ablation")
    ap.add_argument("--stage1", choices=["pointwise", "retriever", "union"], default="pointwise",
                    help="twostage: stage-1 recall arm. pointwise (default) = the 12B yes/no readout; "
                         "retriever = a trained bi-encoder (tod.train.retriever_sft) -- one query "
                         "encode + cached candidate lookups, shortlist = its top-k in original order, "
                         "stage-1 tail mass = softmax(retriever scores / T1); union = run BOTH arms, "
                         "shortlist = union of each arm's top-k_each filled alternately to k, tail "
                         "mass = average of the two calibrated branch softmaxes (one T1 each). Stage "
                         "2 (letter re-rank) is unchanged and still needs --model.")
    ap.add_argument("--k-each", type=int, default=8,
                    help="twostage --stage1 union: how many top candidates to take from EACH arm's "
                         "ranking for the union (default 8, so <=16=2*k_each before dedup/fill; the "
                         "shortlist is then filled alternately to --k). Also the per-arm depth of the "
                         "reported union recall@k.")
    ap.add_argument("--retriever-ckpt", default=None,
                    help="twostage --stage1 retriever: bi-encoder checkpoint dir from retriever_sft")
    ap.add_argument("--retriever-batch", type=int, default=64,
                    help="twostage --stage1 retriever: candidate encode batch size")
    ap.add_argument("--adapter2", default=None,
                    help="twostage: stage-2 PEFT adapter (default: reuse --adapter / none). If it "
                         "differs from --adapter it is attached as a named 'stage2' adapter on the "
                         "shared base; with no --adapter, stage 1 runs the frozen base (disable_adapter)")
    ap.add_argument("--t1", type=float, default=None, help="twostage: fixed stage-1 (branch softmax) T "
                    "(for --stage1 union this is the POINTWISE arm's T)")
    ap.add_argument("--t1-ret", type=float, default=None,
                    help="twostage --stage1 union: fixed retriever-arm stage-1 T (default: fit on calib)")
    ap.add_argument("--t2", type=float, default=None, help="twostage: fixed stage-2 (letter softmax) T")
    ap.add_argument("--branch-batch", type=int, default=16, help="branches per forward (pointwise)")
    ap.add_argument("--dtype", default="bfloat16",
                    help="pointwise model dtype; float32 is bit-exact under batched padding "
                         "(bf16 batched matmul is non-associative -> ~0.25 log-odds padding noise)")
    ap.add_argument("--pw-route", choices=["cache", "solo_batched", "packed4d", "flat", "auto"], default="cache",
                    help="pointwise exactness route; auto-falls back to solo_batched if selftest fails. "
                         "packed4d = one forward with a 4D block mask (fastest; needs fp32 to be exact)")
    ap.add_argument("--selftest", type=int, default=8,
                    help="pointwise: #records for the packed-vs-solo + deletion self-test (run before numbers)")
    ap.add_argument("--jsonl", default=None,
                    help="pointwise: evaluate an arbitrary canonical-schema jsonl (high-N sets) instead of tiers")
    ap.add_argument("--temperature", type=float, default=None,
                    help="pointwise: use this fixed calibration T (skip fitting; e.g. reuse a T fit once "
                         "on a shared calib set across nested tiers)")
    args = ap.parse_args(argv)
    global OPTION_STYLE
    OPTION_STYLE = args.option_style

    if args.dump_prompt:
        recs = load_records(os.path.join(args.jevbench_root, f"{args.tiers[0]}.jsonl"))
        print(build_user_message(recs[0]))
        return

    if args.readout == "pointwise":
        return _main_pointwise(args)
    if args.readout == "twostage":
        return _main_twostage(args)

    scorer = LetterLogitScorer(args.model, device=args.device, attn=args.attn,
                               max_tokens=args.max_tokens, revision=args.revision,
                               adapter=args.adapter)
    print(f"[letter_logit] loaded {args.model} in {scorer.load_s:.0f}s; chat_template={scorer.has_chat}",
          flush=True)
    result: Dict[str, Any] = {"model": args.model, "revision": args.revision, "adapter": args.adapter,
                              "prompt_version": PROMPT_VERSION, "max_tokens": args.max_tokens,
                              "readout": "restricted softmax over option-letter logits at the last "
                                         "position (logsumexp over 'A'/' A' variants)",
                              "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    result["option_style"] = args.option_style; result["order_avg"] = args.order_avg
    if args.multimodal:
        if scorer.processor is None:
            print(f"[letter_logit] WARNING: {args.model} has no multimodal processor; "
                  f"image rows will score as text (no pixels).", flush=True)
        result["multimodal"] = evaluate_multimodal(
            scorer, args.canonical_root, args.mm_datasets, args.mm_split, args.mm_n,
            T=1.0, n_boot=args.n_boot)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=1)
        print(f"[letter_logit] wrote {args.out} (multimodal); "
              f"vision_earns_keep_all={result['multimodal']['vision_earns_keep_all']}", flush=True)
        return result

    cache: Dict[str, Dict[str, Any]] = {}
    # raw (T=1)
    result["raw"] = evaluate(scorer, args.jevbench_root, args.tiers, 1.0, cache, args.order_avg)
    # calibrated
    if args.calib_jsonl:
        import random
        crecs = [r for r in load_records(args.calib_jsonl)
                 if r.get("expected") is not None and 0 < len(r.get("labels", [])) <= len(LETTERS)]
        random.Random(0).shuffle(crecs)
        crecs = crecs[: args.calib_n]
        t0 = time.time()
        items = [(r, scorer.score(r, order_avg=args.order_avg)["logits"]) for r in crecs]
        fit = fit_temperature(items)
        fit["elapsed_s"] = round(time.time() - t0, 1); fit["source"] = args.calib_jsonl
        result["temperature"] = fit
        print(f"[letter_logit] temperature fit: {fit}", flush=True)
        result["calibrated"] = evaluate(scorer, args.jevbench_root, args.tiers, fit["T"], cache,
                                        args.order_avg)
    result["cache"] = cache  # per-item letter logits: lets anyone re-calibrate without a GPU
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    print(f"[letter_logit] wrote {args.out}", flush=True)
    return result


if __name__ == "__main__":
    main()

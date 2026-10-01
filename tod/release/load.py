"""Reconstruct a two-stage picker from a packaged directory and predict.

``load_picker(dir)`` reads ``picker.json`` and rebuilds a ``tod.eval.letter_logit.TwoStageScorer``
(no prompt code is duplicated -- stage prompts, shortlist and final-probability math all come from
letter_logit). ``Picker.predict(state, options, question=None, images=None)`` returns calibrated
probabilities over ALL N options via exactly the harness's two-stage probability path."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Union


def _abs(base_dir: str, rel: Optional[str]) -> Optional[str]:
    if not rel:
        return None
    return rel if os.path.isabs(rel) else os.path.join(base_dir, rel)


class Picker:
    """A loaded two-stage picker: the reconstructed scorer + its calibrated temperatures."""

    def __init__(self, scorer: Any, manifest: Dict[str, Any]):
        self.scorer = scorer
        self.manifest = manifest
        t = manifest["temperatures"]
        self.T1 = float(t["T1"]); self.T2 = float(t["T2"])
        self.T1_ret = None if t.get("T1_ret") is None else float(t["T1_ret"])

    @staticmethod
    def build_record(state: Any, options: Union[Sequence[str], Dict[str, Any]],
                     question: Union[None, str, Dict[str, Any]] = None,
                     images: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        """Assemble one harness record (the schema ``tod.eval.letter_logit`` consumes) from a raw
        (state, options, question, images) call. ``options`` is a label list or a
        ``{label: description}`` map (its keys become the labels, its values the criteria). ``question``
        is a full question dict, an instructions string, or None (a plain choice). ``images`` (paths)
        route the record through the multimodal content path by turning ``state`` into ordered
        text/image segments; without images ``state`` is passed through untouched (str/dict/list)."""
        if isinstance(options, dict):
            labels = [str(k) for k in options.keys()]
            criteria: Optional[Dict[str, Any]] = {str(k): v for k, v in options.items()}
        else:
            labels = [str(x) for x in options]
            criteria = None
        if isinstance(question, dict):
            q = dict(question)
            q.setdefault("type", "choice")
            q.setdefault("instructions", "Select the single best option.")
            if criteria is not None:
                q.setdefault("criteria", criteria)
        else:
            q = {"type": "choice",
                 "instructions": str(question) if question else "Select the single best option."}
            if criteria is not None:
                q["criteria"] = criteria
        if images:
            if isinstance(state, list):
                segs = list(state)
            else:
                segs = [{"text": state if isinstance(state, str)
                         else json.dumps(state, ensure_ascii=False)}]
            segs += [{"image": str(p)} for p in images]
            rec_state: Any = segs
        else:
            rec_state = state
        return {"id": "predict", "state": rec_state, "question": q, "labels": labels}

    def predict(self, state: Any, options: Union[Sequence[str], Dict[str, Any]],
                question: Union[None, str, Dict[str, Any]] = None,
                images: Optional[Sequence[str]] = None) -> List[float]:
        """Calibrated probability for every one of the N ``options`` (order preserved; N unbounded).

        Runs exactly the harness path: stage-1 recall over all N (pointwise unioned with the
        retriever) -> stage-2 letter re-rank of the top-k shortlist -> ``twostage_final_probs`` with
        the packaged T1/T1_ret/T2. No prompt/shortlist/probability logic is duplicated here. For
        N <= k this is exactly the pure letter read. Returns a list of floats summing to 1."""
        from tod.eval.letter_logit import twostage_score_record, twostage_final_probs
        rec = self.build_record(state, options, question, images)
        s1, s2 = twostage_score_record(self.scorer, rec)
        return twostage_final_probs(rec, s1, s2, self.T1, self.T2, self.T1_ret)


def load_picker(dir: str, *, device: str = "cuda", dtype: Optional[str] = None,
                attn: Optional[str] = None) -> Picker:
    """Rebuild the ``TwoStageScorer`` described by ``<dir>/picker.json``.

    ``dtype``/``attn`` default to the manifest's serve dtype and recorded attention. The base id
    is resolved relative to ``dir`` when it is a bundled ``merged/`` directory, else used verbatim
    (e.g. a HuggingFace hub id). Adapters/retriever paths are resolved relative to ``dir``."""
    with open(os.path.join(dir, "picker.json"), "r", encoding="utf-8") as fh:
        m = json.load(fh)
    rd = m["readout"]
    base = m["base"]["id"]
    base_id = _abs(dir, base) if base == "merged" else base
    from tod.eval.letter_logit import TwoStageScorer
    scorer = TwoStageScorer(
        base_id,
        device=device,
        dtype=dtype or rd["dtype"]["serve"],
        attn=attn or rd["attn"],
        max_tokens=int(rd["max_tokens"]),
        revision=m["base"].get("revision"),
        adapter=_abs(dir, m["adapters"].get("stage1")),
        adapter2=_abs(dir, m["adapters"].get("stage2")),
        route="cache",
        k=int(rd["k"]),
        stage1_kind=rd["stage1_kind"],
        retriever_ckpt=_abs(dir, m.get("retriever")),
        k_each=int(rd["k_each"]),
    )
    return Picker(scorer, m)

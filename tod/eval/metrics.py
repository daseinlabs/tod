"""Evaluation metrics: accuracy, ECE, Brier, gold-distribution fidelity,
paraphrase consistency, majority-class baseline.

A prediction record is the dict emitted by ``TodModel.decode``:
  {type, id, choice|score|noul, probabilities?, confidence}
A gold record is:
  {id, type, expected, probs, group, family, soft_source?}
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


def predicted_label(pred: Dict[str, Any]) -> str:
    t = pred["type"]
    if t == "choice":
        return pred["choice"]
    if t == "noul":
        return "yes" if pred["noul"] >= 0.5 else "no"
    # score: nearest integer level
    return str(int(round(pred["score"])))


def _prob_of(pred: Dict[str, Any], label: str) -> float:
    t = pred["type"]
    if t == "noul":
        return pred["noul"] if label == "yes" else 1.0 - pred["noul"]
    return float(pred.get("probabilities", {}).get(label, 0.0))


def accuracy(preds: Dict[str, Dict], golds: Dict[str, Dict]) -> float:
    ok = tot = 0
    for gid, g in golds.items():
        if gid not in preds:
            continue
        tot += 1
        if predicted_label(preds[gid]) == str(g["expected"]):
            ok += 1
    return ok / tot if tot else 0.0


def ece(preds: Dict[str, Dict], golds: Dict[str, Dict], bins: int = 10) -> float:
    """Expected Calibration Error on the top-label confidence (10-bin)."""
    buckets: List[List[Tuple[float, int]]] = [[] for _ in range(bins)]
    n = 0
    for gid, g in golds.items():
        if gid not in preds:
            continue
        p = preds[gid]
        conf = float(p.get("confidence", 0.0))
        correct = int(predicted_label(p) == str(g["expected"]))
        b = min(bins - 1, int(conf * bins))
        buckets[b].append((conf, correct))
        n += 1
    if n == 0:
        return 0.0
    e = 0.0
    for bk in buckets:
        if not bk:
            continue
        avg_conf = sum(c for c, _ in bk) / len(bk)
        acc = sum(y for _, y in bk) / len(bk)
        e += (len(bk) / n) * abs(avg_conf - acc)
    return e


def _conf_correct(preds, golds):
    """(confidence, correct) pairs; prefers calibrated_confidence when present."""
    out = []
    for gid, g in golds.items():
        if gid not in preds:
            continue
        p = preds[gid]
        conf = p.get("calibrated_confidence")
        if conf is None:
            conf = float(p.get("confidence", 0.0))
        out.append((float(conf), int(predicted_label(p) == str(g["expected"]))))
    return out


def ece_debiased(preds, golds, bins: int = 10) -> float:
    """Bias-corrected binned ECE: subtract each bin's accuracy sampling variance
    acc*(1-acc)/(n-1) (Roelofs et al. 2022) so small bins don't inflate ECE."""
    cc = _conf_correct(preds, golds)
    n = len(cc)
    if n == 0:
        return 0.0
    buckets: List[List[Tuple[float, int]]] = [[] for _ in range(bins)]
    for conf, y in cc:
        buckets[min(bins - 1, int(conf * bins))].append((conf, y))
    e = 0.0
    for bk in buckets:
        if not bk:
            continue
        m = len(bk)
        acc = sum(y for _, y in bk) / m
        conf = sum(c for c, _ in bk) / m
        gap = abs(conf - acc)
        var = acc * (1 - acc) / max(m - 1, 1)
        gap = max(0.0, gap - var ** 0.5)  # subtract ~1 std of sampling noise
        e += (m / n) * gap
    return e


def smooth_ece(preds, golds, sigma: float = 0.1) -> float:
    """Kernel-smoothed calibration error (Blasiok & Nakkiran 2023, approx): Gaussian-
    weighted local accuracy vs confidence over a fine grid; binning-free."""
    import math
    cc = _conf_correct(preds, golds)
    if not cc:
        return 0.0
    grid = [i / 100.0 for i in range(101)]
    num = 0.0
    den = 0.0
    for x in grid:
        w = sum(math.exp(-((c - x) ** 2) / (2 * sigma ** 2)) for c, _ in cc)
        if w <= 0:
            continue
        acc = sum(math.exp(-((c - x) ** 2) / (2 * sigma ** 2)) * y for c, y in cc) / w
        conf = sum(math.exp(-((c - x) ** 2) / (2 * sigma ** 2)) * c for c, _ in cc) / w
        num += w * abs(acc - conf)
        den += w
    return num / den if den else 0.0


def aurc(preds, golds) -> Dict[str, float]:
    """Area under the risk-coverage curve (Geifman & El-Yaniv): lower is better.
    Sort by confidence desc, accumulate risk (error) as coverage grows."""
    cc = sorted(_conf_correct(preds, golds), key=lambda x: -x[0])
    n = len(cc)
    if n == 0:
        return {"aurc": 0.0, "auroc_pcorrect": float("nan"), "n": 0}
    risks = []
    err = 0
    for i, (_, y) in enumerate(cc, 1):
        err += (1 - y)
        risks.append(err / i)
    a = sum(risks) / n
    # AUROC of confidence vs correctness
    pos = [c for c, y in cc if y == 1]
    neg = [c for c, y in cc if y == 0]
    if pos and neg:
        wins = sum(1 for cp in pos for cn in neg if cp > cn)
        ties = sum(1 for cp in pos for cn in neg if cp == cn)
        auroc = (wins + 0.5 * ties) / (len(pos) * len(neg))
    else:
        auroc = float("nan")
    return {"aurc": a, "auroc_pcorrect": auroc, "n": n}


def _card_bucket(k: int) -> str:
    if k <= 2:
        return "2"
    if k <= 4:
        return "3-4"
    if k <= 8:
        return "5-8"
    if k <= 32:
        return "9-32"
    return "33+"


def calibration_by_cardinality(preds, golds, bins: int = 10) -> Dict[str, Dict[str, float]]:
    """ECE + accuracy per choice cardinality bucket (2, 3-4, 5-8, 9-32, 33+)."""
    groups: Dict[str, Tuple[Dict, Dict]] = {}
    for gid, g in golds.items():
        if gid not in preds or g.get("type") != "choice":
            continue
        b = _card_bucket(len(g.get("probs", {})) or 2)
        groups.setdefault(b, ({}, {}))
        groups[b][0][gid] = preds[gid]
        groups[b][1][gid] = g
    return {b: {"n": len(gg), "accuracy": accuracy(pp, gg),
                "ece": ece(pp, gg, bins), "ece_debiased": ece_debiased(pp, gg, bins)}
            for b, (pp, gg) in groups.items()}


def reliability_data(preds, golds, bins: int = 10) -> List[Dict[str, float]]:
    """Reliability-diagram bins: (mean confidence, accuracy, count) per bin."""
    cc = _conf_correct(preds, golds)
    buckets: List[List[Tuple[float, int]]] = [[] for _ in range(bins)]
    for conf, y in cc:
        buckets[min(bins - 1, int(conf * bins))].append((conf, y))
    out = []
    for i, bk in enumerate(buckets):
        if bk:
            out.append({"bin": (i + 0.5) / bins, "count": len(bk),
                        "confidence": sum(c for c, _ in bk) / len(bk),
                        "accuracy": sum(y for _, y in bk) / len(bk)})
    return out


def brier(preds: Dict[str, Dict], golds: Dict[str, Dict]) -> float:
    """Multi-class Brier over the labelled option set (uses gold probs)."""
    tot = 0.0
    n = 0
    for gid, g in golds.items():
        if gid not in preds:
            continue
        labels = list(g.get("probs", {}).keys()) or [g["expected"]]
        gp = g.get("probs", {}) or {g["expected"]: 1.0}
        s = 0.0
        for lb in labels:
            s += (_prob_of(preds[gid], lb) - float(gp.get(lb, 0.0))) ** 2
        tot += s
        n += 1
    return tot / n if n else 0.0


def gold_distribution_fidelity(preds: Dict[str, Dict], golds: Dict[str, Dict]) -> Dict[str, float]:
    """Mean signed / abs gap between predicted and gold prob of the expected label,
    restricted to soft/probability items (soft_source set or family == probability)."""
    signed, absd, n = 0.0, 0.0, 0
    for gid, g in golds.items():
        if gid not in preds:
            continue
        soft = g.get("soft_source") or g.get("family") == "probability"
        if not soft:
            continue
        exp = str(g["expected"])
        gp = float(g.get("probs", {}).get(exp, 1.0))
        pp = _prob_of(preds[gid], exp)
        signed += (pp - gp)
        absd += abs(pp - gp)
        n += 1
    return {"mean_gap": signed / n if n else 0.0,
            "mean_abs_gap": absd / n if n else 0.0, "n": n}


def paraphrase_consistency(preds: Dict[str, Dict], golds: Dict[str, Dict]) -> Dict[str, float]:
    """Fraction of multi-item groups whose predicted labels all agree."""
    groups: Dict[str, List[str]] = defaultdict(list)
    for gid, g in golds.items():
        if gid in preds:
            groups[g.get("group", gid)].append(predicted_label(preds[gid]))
    multi = [v for v in groups.values() if len(v) > 1]
    if not multi:
        return {"consistency": float("nan"), "n_groups": 0}
    consistent = sum(1 for v in multi if len(set(v)) == 1)
    return {"consistency": consistent / len(multi), "n_groups": len(multi)}


def majority_baseline(golds: Dict[str, Dict], key: str = "type") -> float:
    """Accuracy of always predicting the per-(type) most common gold label."""
    by_key: Dict[str, List[str]] = defaultdict(list)
    for g in golds.values():
        by_key[g.get(key, "?")].append(str(g["expected"]))
    ok = tot = 0
    for _, labels in by_key.items():
        if not labels:
            continue
        maj = max(set(labels), key=labels.count)
        ok += sum(1 for x in labels if x == maj)
        tot += len(labels)
    return ok / tot if tot else 0.0


def breakdown(preds: Dict[str, Dict], golds: Dict[str, Dict], key: str) -> Dict[str, Dict[str, float]]:
    """Per-`key` (family / type) accuracy + count."""
    groups: Dict[str, Tuple[int, int]] = defaultdict(lambda: (0, 0))
    for gid, g in golds.items():
        if gid not in preds:
            continue
        k = str(g.get(key, "?"))
        ok, tot = groups[k]
        tot += 1
        if predicted_label(preds[gid]) == str(g["expected"]):
            ok += 1
        groups[k] = (ok, tot)
    return {k: {"acc": ok / tot if tot else 0.0, "n": tot} for k, (ok, tot) in groups.items()}


def selection_score(report: Dict[str, Any]) -> float:
    """Model-selection objective: minimise debiased ECE + Brier (NOT the 10-bin ECE).
    Lower is better; use to pick checkpoints/epochs."""
    return float(report.get("ece_debiased", 0.0)) + float(report.get("brier", 0.0))


def full_report(preds: Dict[str, Dict], golds: Dict[str, Dict], *, bins: int = 10,
                reliability: bool = True) -> Dict[str, Any]:
    rep = {
        "n": len(golds),
        "n_scored": sum(1 for gid in golds if gid in preds),
        "accuracy": accuracy(preds, golds),
        "ece": ece(preds, golds, bins),
        "ece_debiased": ece_debiased(preds, golds, bins),
        "smooth_ece": smooth_ece(preds, golds),
        "brier": brier(preds, golds),
        "aurc": aurc(preds, golds),
        "gold_distribution_fidelity": gold_distribution_fidelity(preds, golds),
        "paraphrase_consistency": paraphrase_consistency(preds, golds),
        "majority_baseline": majority_baseline(golds),
        "by_family": breakdown(preds, golds, "family"),
        "by_type": breakdown(preds, golds, "type"),
        "by_dataset_ood": breakdown(preds, golds, "dataset"),
        "by_cardinality": calibration_by_cardinality(preds, golds, bins),
    }
    if reliability:
        rep["reliability"] = reliability_data(preds, golds, bins)
    rep["selection_score"] = selection_score(rep)
    return rep

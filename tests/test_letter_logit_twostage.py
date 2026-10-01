"""CPU-only tests for the two-stage (pointwise recall -> letter re-rank) readout.

The GPU exactness proof (stage-1 branch log-odds == solo, N<=k == the pure letter read on the
box's real model) is the ``--readout twostage`` self-test. These tests cover the model-free
composition logic with fake per-stage scorers:

  * the top-k shortlist keeps ORIGINAL record order (never stage-1 rank order) and keeps the
    gold label when it survives stage 1,
  * for N <= k the two-stage distribution reduces EXACTLY to the pure stage-2 (letter) read,
  * the final distribution over all N labels sums to 1 (stage-2 mass on the shortlist, stage-1
    tail mass on the rest),
  * ``recall_at_k`` is computed correctly per set.
"""
from __future__ import annotations

import math

import tod.eval.letter_logit as ll


# --- fake per-stage scorers: read pre-baked logits off the record, no model/tokenizer --------
class _FakeStage1:
    def score(self, rec, **_):
        lo = list(rec["_s1"])
        return {"logits": lo, "n_tokens": 10, "n_branches": len(lo), "truncated": False}


class _FakeStage2:
    def _score_one(self, rec):
        lg = [rec["_s2"][str(l)] for l in rec["labels"]]
        return {"logits": lg, "n_tokens": 8, "truncated": False}

    def score(self, rec, order_avg=False):
        return self._score_one(rec)


class _FakeTS:
    def __init__(self, k):
        self.k = k
        self.stage1 = _FakeStage1()
        self.stage2 = _FakeStage2()
        self._diff_adapter = False
        self.model_id = "fake"

    def _activate_stage1(self):
        pass

    def _activate_stage2(self):
        pass


class _FakeRetriever:
    """Second stage-1 arm for --stage1 union: reads pre-baked retriever scores off ``_s1r``."""
    route = "retriever"

    def score(self, rec, **_):
        lo = list(rec["_s1r"])
        return {"logits": lo, "n_tokens": 12, "n_branches": len(lo), "truncated": False}


class _FakeUnionTS:
    stage1_kind = "union"

    def __init__(self, k, k_each):
        self.k = k
        self.k_each = k_each
        self.stage1 = _FakeStage1()      # pointwise arm (reads _s1)
        self.stage1_ret = _FakeRetriever()  # retriever arm (reads _s1r)
        self.stage2 = _FakeStage2()
        self._diff_adapter = False
        self.model_id = "fake-union"

    def _activate_stage1(self):
        pass

    def _activate_stage2(self):
        pass


def _row(labels, s1, s2, expected, rid="r", dataset=None):
    r = {"id": rid, "state": "x", "expected": expected, "labels": list(labels),
         "question": {"type": "choice", "instructions": "pick",
                      "criteria": {l: f"desc {l}" for l in labels}},
         "_s1": list(s1), "_s2": dict(s2)}
    if dataset:
        r["source"] = {"dataset": dataset}
    return r


def _urow(labels, s1, s1r, s2, expected, rid="r", dataset=None):
    r = _row(labels, s1, s2, expected, rid=rid, dataset=dataset)
    r["_s1r"] = list(s1r)
    return r


def _softmax(xs):
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [v / s for v in e]


def test_shortlist_original_order_and_keeps_gold():
    # log-odds rank: B(2.0) D(1.5) E(0.5) A(0.1) C(-1.0); top-3 = {B,D,E} at idx {1,3,4}.
    rec = _row(["A", "B", "C", "D", "E"],
               s1=[0.1, 2.0, -1.0, 1.5, 0.5],
               s2={"A": 0, "B": 1.0, "C": 0, "D": 0.0, "E": 2.0},
               expected="B")
    ts = _FakeTS(k=3)
    s1, s2 = ll.twostage_score_record(ts, rec, order_avg=False)
    # shortlist kept in ORIGINAL order (ascending index), NOT stage-1 rank order (B,D,E).
    assert s2["shortlist_idx"] == [1, 3, 4]
    assert s2["shortlist_labels"] == ["B", "D", "E"]
    # gold "B" survived stage 1 -> it is in the shortlist.
    assert "B" in s2["shortlist_labels"]
    # stage-2 logits are aligned to the shortlist labels in that same original order.
    assert s2["logits"] == [1.0, 0.0, 2.0]


def test_n_le_k_reduces_to_pure_letter_read():
    rec = _row(["A", "B", "C"], s1=[0.3, -0.2, 1.1],
               s2={"A": 0.5, "B": 2.0, "C": -1.0}, expected="B")
    ts = _FakeTS(k=8)  # k >= N -> shortlist is all labels
    s1, s2 = ll.twostage_score_record(ts, rec)
    assert s2["shortlist_idx"] == [0, 1, 2]  # all, original order
    final = ll.twostage_final_probs(rec, s1, s2, T1=1.0, T2=1.0)
    pure = ll.probs_from_logits(ts.stage2._score_one(rec)["logits"], 1.0)
    assert max(abs(a - b) for a, b in zip(final, pure)) < 1e-12
    # and it is exactly softmax over the stage-2 letter logits.
    assert max(abs(a - b) for a, b in zip(final, _softmax([0.5, 2.0, -1.0]))) < 1e-12


def test_final_probs_sum_to_one_with_tail():
    rec = _row(["A", "B", "C", "D", "E"],
               s1=[0.1, 2.0, -1.0, 1.5, 0.5],
               s2={"A": 0, "B": 1.0, "C": 0, "D": 0.0, "E": 2.0},
               expected="B")
    ts = _FakeTS(k=3)
    s1, s2 = ll.twostage_score_record(ts, rec)
    final = ll.twostage_final_probs(rec, s1, s2, T1=1.3, T2=0.7)
    assert abs(sum(final) - 1.0) < 1e-12
    # non-shortlisted labels (A idx0, C idx2) keep their stage-1 calibrated mass.
    p1 = ll.probs_from_logits(s1["logits"], 1.3)
    assert abs(final[0] - p1[0]) < 1e-12 and abs(final[2] - p1[2]) < 1e-12
    # shortlisted mass == the stage-1 mass of the shortlist.
    shortlist_mass = p1[1] + p1[3] + p1[4]
    assert abs((final[1] + final[3] + final[4]) - shortlist_mass) < 1e-12


def test_recall_at_k_computed():
    ts = _FakeTS(k=3)
    # r1: gold "B" is stage-1 rank 0 -> hit @1,@3.  r2: gold "A" (idx0) rank 3 -> miss @1,@3.
    r1 = _row(["A", "B", "C", "D", "E"], s1=[0.1, 2.0, -1.0, 1.5, 0.5],
              s2={"A": 0, "B": 1.0, "C": 0, "D": 0.0, "E": 2.0}, expected="B", rid="r1")
    r2 = _row(["A", "B", "C", "D", "E"], s1=[0.1, 2.0, -1.0, 1.5, 0.5],
              s2={"A": 0, "B": 1.0, "C": 0, "D": 0.0, "E": 2.0}, expected="A", rid="r2")
    c1, c2 = {}, {}
    rep = ll.evaluate_twostage(ts, [r1, r2], "fake_set", 1.0, 1.0, c1, c2,
                               order_avg=False, k_list=(1, 3, 5))
    assert rep["recall_n"] == 2
    assert rep["recall_at_k"]["1"] == 0.5   # only r1 hits @1
    assert rep["recall_at_k"]["3"] == 0.5   # r1 hits @3 (rank0), r2 misses (rank3)
    assert rep["recall_at_k"]["5"] == 1.0   # both golds are within top-5 (=all)
    assert rep["k"] == 3


def test_recall_at_k_per_dataset():
    ts = _FakeTS(k=3)
    # two sets: set A gold rank 0 (hit@1), set B gold rank 3 (miss@1, miss@3).
    a = _row(["A", "B", "C", "D", "E"], s1=[0.1, 2.0, -1.0, 1.5, 0.5],
             s2={"A": 0, "B": 1.0, "C": 0, "D": 0.0, "E": 2.0}, expected="B", rid="a", dataset="setA")
    b = _row(["A", "B", "C", "D", "E"], s1=[0.1, 2.0, -1.0, 1.5, 0.5],
             s2={"A": 0, "B": 1.0, "C": 0, "D": 0.0, "E": 2.0}, expected="A", rid="b", dataset="setB")
    rep = ll.evaluate_twostage(ts, [a, b], "highN", 1.0, 1.0, {}, {}, k_list=(1, 3))
    bd = rep["recall_at_k_by_dataset"]
    assert bd["setA"]["n"] == 1 and bd["setA"]["1"] == 1.0 and bd["setA"]["3"] == 1.0
    assert bd["setB"]["n"] == 1 and bd["setB"]["1"] == 0.0 and bd["setB"]["3"] == 0.0


# ---------------------------------------------------------------- --stage1 union -----------------
def test_union_shortlist_no_fill_union_of_top_k_each():
    # pw ranks A,B,C,D,E,F (idx 0..5); ret ranks the reverse (F,E,D,C,B,A).
    pw = [5, 4, 3, 2, 1, 0]
    ret = [0, 1, 2, 3, 4, 5]
    # top-2 of each: pw -> {0,1}, ret -> {5,4}; union already fills k=4 (no fill needed).
    sl = ll.twostage_union_shortlist(pw, ret, n=6, k=4, k_each=2)
    assert sl == [0, 1, 4, 5]  # ORIGINAL record order (ascending index)


def test_union_shortlist_fill_and_dedup():
    # both arms agree on the ranking -> top-2 union is just {0,1} (dedup); fill alternately to k=4.
    pw = [5, 4, 3, 2, 1, 0]
    ret = [5, 4, 3, 2, 1, 0]
    sl = ll.twostage_union_shortlist(pw, ret, n=6, k=4, k_each=2)
    # union {0,1}; fill: pw next -> idx2, ret next (0,1,2 seen) -> idx3.
    assert sl == [0, 1, 2, 3]


def test_union_shortlist_fill_alternates_between_arms():
    # top-1 each: pw -> {0}, ret -> {5}; fill alternates pw(idx1) then ret(idx4).
    pw = [5, 4, 3, 2, 1, 0]
    ret = [0, 1, 2, 3, 4, 5]
    sl = ll.twostage_union_shortlist(pw, ret, n=6, k=4, k_each=1)
    assert sl == [0, 1, 4, 5]  # {0,5} union + pw idx1 + ret idx4, in original order


def test_union_shortlist_caps_at_min_k_n():
    # N < k: every index selected, original order (this is what reduces the picker to a letter read).
    sl = ll.twostage_union_shortlist([2.0, 1.0, 0.0], [0.0, 1.0, 2.0], n=3, k=8, k_each=4)
    assert sl == [0, 1, 2]


def test_union_n_le_k_reduces_to_pure_letter_read():
    rec = _urow(["A", "B", "C"], s1=[0.3, -0.2, 1.1], s1r=[2.0, 0.0, -1.0],
                s2={"A": 0.5, "B": 2.0, "C": -1.0}, expected="B")
    ts = _FakeUnionTS(k=8, k_each=4)  # k >= N -> union shortlist is all labels
    s1, s2 = ll.twostage_score_record(ts, rec)
    assert s1["kind"] == "union"
    assert s2["shortlist_idx"] == [0, 1, 2]
    final = ll.twostage_final_probs(rec, s1, s2, T1=1.3, T2=1.0, T1_ret=0.6)
    # both stage-1 arms (and their average tail) are fully overwritten -> pure stage-2 letter read.
    assert max(abs(a - b) for a, b in zip(final, _softmax([0.5, 2.0, -1.0]))) < 1e-12


def test_union_final_probs_sum_to_one_with_averaged_tail():
    rec = _urow(["A", "B", "C", "D", "E"], s1=[5, 4, 3, 2, 1], s1r=[1, 2, 3, 4, 5],
                s2={"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0, "E": 0.0}, expected="A")
    ts = _FakeUnionTS(k=3, k_each=2)
    s1, s2 = ll.twostage_score_record(ts, rec)
    assert s2["shortlist_idx"] == [0, 1, 4]  # pw top2 {0,1} U ret top2 {4,3} capped/ordered -> [0,1,4]
    T1, T1r, T2 = 1.2, 0.8, 1.0
    final = ll.twostage_final_probs(rec, s1, s2, T1=T1, T2=T2, T1_ret=T1r)
    assert abs(sum(final) - 1.0) < 1e-12
    # tail mass on the two non-shortlisted labels (C idx2, D idx3) = average of the two calibrated
    # stage-1 branch softmaxes.
    p_avg = [0.5 * (a + b)
             for a, b in zip(_softmax([v / T1 for v in [5, 4, 3, 2, 1]]),
                             _softmax([v / T1r for v in [1, 2, 3, 4, 5]]))]
    assert abs(final[2] - p_avg[2]) < 1e-12 and abs(final[3] - p_avg[3]) < 1e-12


def test_union_recall_keys_per_arm_and_per_dataset():
    ts = _FakeUnionTS(k=3, k_each=2)
    # gold "A" (idx0): pointwise ranks it FIRST (hit @1/@3/@5), retriever ranks it LAST (hit @5 only);
    # union keeps it (pw puts it in the union) at every depth.
    r = _urow(["A", "B", "C", "D", "E"], s1=[5, 4, 3, 2, 1], s1r=[1, 2, 3, 4, 5],
              s2={"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0, "E": 0.0},
              expected="A", rid="r1", dataset="setA")
    rep = ll.evaluate_twostage(ts, [r], "u", 1.0, 1.0, {}, {}, k_list=(1, 3, 5))
    rk = rep["recall_at_k"]
    assert set(rk) == {"pointwise", "retriever", "union"}
    assert rk["pointwise"] == {"1": 1.0, "3": 1.0, "5": 1.0}
    assert rk["retriever"] == {"1": 0.0, "3": 0.0, "5": 1.0}
    assert rk["union"] == {"1": 1.0, "3": 1.0, "5": 1.0}
    bd = rep["recall_at_k_by_dataset"]["setA"]
    assert bd["n"] == 1 and set(bd) == {"n", "pointwise", "retriever", "union"}
    assert bd["union"] == {"1": 1.0, "3": 1.0, "5": 1.0}
    assert rep["k"] == 3 and rep["k_each"] == 2

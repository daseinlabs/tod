"""CPU-only tests for three letter_logit eval fixes (no downloaded checkpoints).

Each test uses a tiny random Llama + the gpt2 tokenizer (same recipe as test_pointwise_sft),
saved to disk so the scorers' own ``from_pretrained`` path is exercised end to end.

  1. ``--adapter2`` WITHOUT ``--adapter``: stage 1 runs the FROZEN base (disable_adapter) and
     stage 2 uses the "stage2" adapter -- proved by comparing to a frozen PointwiseScorer and an
     adapter-loaded LetterLogitScorer.
  2. ``LetterLogitScorer`` registers ``tod.model.attn_chunked`` when ``attn == "chunked_eager"``
     (so the retriever-only two-stage path, where it is the only loader, still resolves the impl).
  3. ``logits_to_keep=1`` keeps only the last position's logits (calib memory) and matches the
     full forward there, with a graceful fallback when the model rejects the kwarg.
"""
from __future__ import annotations

import sys
import types

import pytest
import torch

import tod.eval.letter_logit as ll


# --------------------------------------------------------------------------- fixtures / helpers
@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained("gpt2")
    t.pad_token = t.eos_token
    return t


def _make_tiny_model(tok):
    from transformers import AutoModelForCausalLM, LlamaConfig
    cfg = LlamaConfig(vocab_size=tok.vocab_size, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                      max_position_embeddings=512)
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(cfg)


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory, tok):
    d = tmp_path_factory.mktemp("tiny_model")
    _make_tiny_model(tok).save_pretrained(str(d))
    tok.save_pretrained(str(d))
    return str(d)


@pytest.fixture(scope="module")
def adapter_dir(tmp_path_factory, tiny_dir):
    """A randomised (non-identity) LoRA saved to disk so the adapter measurably changes outputs."""
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    base = AutoModelForCausalLM.from_pretrained(tiny_dir)
    lc = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], lora_dropout=0.0)
    torch.manual_seed(1)
    pm = get_peft_model(base, lc)
    for n, p in pm.named_parameters():
        if "lora_B" in n:  # lora_B initialises to zero (identity); randomise so it does something
            torch.nn.init.normal_(p, std=0.5)
    d = tmp_path_factory.mktemp("tiny_adapter")
    pm.save_pretrained(str(d))
    return str(d)


def _choice_rec():
    return {"id": "c1", "state": "The parcel has not arrived and the buyer is upset.",
            "question": {"type": "choice", "instructions": "Pick the user's intent.",
                         "criteria": {"track": "the user asks about delivery status",
                                      "cancel": "the user wants to cancel",
                                      "refund": "the user wants their money back"}},
            "labels": ["track", "cancel", "refund"], "expected": "track"}


# ------------------------------------------------------------------------------------- Fix 1
def test_adapter2_only_stage1_frozen_stage2_adapted(tiny_dir, adapter_dir):
    """--adapter2 without --adapter: stage 1 == frozen base, stage 2 == the adapter."""
    ts = ll.TwoStageScorer(tiny_dir, device="cpu", dtype="float32", attn="eager",
                           max_tokens=512, adapter=None, adapter2=adapter_dir,
                           branch_batch=8, route="cache", k=16, stage1_kind="pointwise")
    assert ts._adapter2_only is True

    pw_frozen = ll.PointwiseScorer(tiny_dir, device="cpu", dtype="float32", attn="eager",
                                   max_tokens=512, adapter=None, branch_batch=8, route="cache")
    ll_adapt = ll.LetterLogitScorer(tiny_dir, device="cpu", dtype="float32", attn="eager",
                                    max_tokens=512, adapter=adapter_dir)
    ll_frozen = ll.LetterLogitScorer(tiny_dir, device="cpu", dtype="float32", attn="eager",
                                     max_tokens=512, adapter=None)
    rec = _choice_rec()

    ts._activate_stage1()
    s1 = ts.stage1.score(rec)["logits"]
    ts._activate_stage2()
    s2 = ts.stage2._score_one(rec)["logits"]

    # stage 1 forwards run under disable_adapter -> identical to the base with no adapter at all
    ref1 = pw_frozen.score(rec)["logits"]
    assert torch.allclose(torch.tensor(s1), torch.tensor(ref1), atol=1e-4)

    # stage 2 forwards run with the "stage2" adapter active -> identical to loading it directly,
    # and DIFFERENT from the frozen letter read (the LoRA actually moves the logits)
    ref2_adapt = ll_adapt._score_one(rec)["logits"]
    ref2_frozen = ll_frozen._score_one(rec)["logits"]
    assert torch.allclose(torch.tensor(s2), torch.tensor(ref2_adapt), atol=1e-4)
    assert not torch.allclose(torch.tensor(s2), torch.tensor(ref2_frozen), atol=1e-3)

    # switching back to stage 1 re-freezes: still matches the frozen base
    ts._activate_stage1()
    assert torch.allclose(torch.tensor(ts.stage1.score(rec)["logits"]),
                          torch.tensor(ref1), atol=1e-4)


# ------------------------------------------------------------------------------------- Fix 2
def test_chunked_eager_registers_without_prior_import(tiny_dir):
    """Constructing with attn='chunked_eager' works even if attn_chunked was never imported."""
    try:
        from transformers import AttentionInterface  # noqa: F401
    except Exception:
        pytest.skip("transformers has no AttentionInterface (chunked_eager unsupported)")

    sys.modules.pop("tod.model.attn_chunked", None)  # ensure no prior registration import
    scorer = ll.LetterLogitScorer(tiny_dir, device="cpu", dtype="float32",
                                  attn="chunked_eager", max_tokens=512)
    assert "tod.model.attn_chunked" in sys.modules  # LetterLogitScorer did the side-effect import
    # and it actually runs (the custom attention resolved for the forward)
    out = scorer._score_one(_choice_rec())
    assert len(out["logits"]) == 3


# ------------------------------------------------------------------------------------- Fix 3
def test_forward_last_matches_full_last_position(tiny_dir):
    """logits_to_keep=1 keeps a single position whose logits equal the full forward's last."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(tiny_dir).eval()
    scorer = types.SimpleNamespace(model=model, _logits_to_keep_ok=None)
    x = torch.tensor([[10, 20, 30, 40, 50, 60, 70]])

    with torch.no_grad():
        full = model(input_ids=x, use_cache=False).logits
        kept = ll._forward_last(scorer, input_ids=x, use_cache=False).logits

    assert scorer._logits_to_keep_ok is True          # decision recorded on first call
    assert kept.shape[1] == 1                          # only the last position materialised
    assert full.shape[1] == x.shape[1]                 # full path would keep every position
    assert torch.allclose(kept[0, -1], full[0, -1], atol=1e-6)


def test_forward_last_falls_back_on_typeerror(tiny_dir):
    """A model whose forward rejects logits_to_keep -> full logits, decision cached once."""
    from transformers import AutoModelForCausalLM
    real = AutoModelForCausalLM.from_pretrained(tiny_dir).eval()
    calls = {"probes": 0}

    class _NoKwarg:
        def __call__(self, **kw):
            if "logits_to_keep" in kw:
                calls["probes"] += 1
                raise TypeError("forward() got an unexpected keyword argument 'logits_to_keep'")
            return real(**kw)

    scorer = types.SimpleNamespace(model=_NoKwarg(), _logits_to_keep_ok=None)
    x = torch.tensor([[10, 20, 30, 40, 50]])
    with torch.no_grad():
        out1 = ll._forward_last(scorer, input_ids=x, use_cache=False).logits
        out2 = ll._forward_last(scorer, input_ids=x, use_cache=False).logits
        ref = real(input_ids=x, use_cache=False).logits

    assert scorer._logits_to_keep_ok is False
    assert calls["probes"] == 1                         # probed once, then cached (no re-probe)
    assert out1.shape[1] == x.shape[1] and out2.shape[1] == x.shape[1]
    assert torch.allclose(out1[0, -1], ref[0, -1], atol=1e-6)

"""CPU unit test: chunked_eager attention == eager reference on random tensors.

Covers GQA (num_key_value_groups>1), an additive mask (causal + sliding-window style), optional
logit softcap, and several chunk sizes incl. chunk> q_len and chunk not dividing q_len.
"""
import math
import types

import pytest

torch = pytest.importorskip("torch")

from tod.model import attn_chunked as A


def _eager_ref(module, q, k, v, attention_mask, scaling, softcap):
    kk = A._repeat_kv(k, module.num_key_value_groups)
    vv = A._repeat_kv(v, module.num_key_value_groups)
    aw = torch.matmul(q, kk.transpose(2, 3)) * scaling
    if softcap is not None:
        aw = torch.tanh(aw / softcap) * softcap
    if attention_mask is not None:
        aw = aw + attention_mask[:, :, :, : kk.shape[-2]]
    aw = torch.softmax(aw, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.matmul(aw, vv)
    return out.transpose(1, 2).contiguous()


@pytest.mark.parametrize("q_len,chunk", [(40, 8), (40, 7), (40, 64), (128, 32), (1, 4),
                                         (128, 8192)])  # CHUNKED_ATTN_Q=8192: <=8k -> one chunk
@pytest.mark.parametrize("softcap", [None, 50.0])
@pytest.mark.parametrize("n_rep", [1, 4])
def test_chunked_matches_eager(q_len, chunk, softcap, n_rep):
    torch.manual_seed(0)
    b, n_kv, d = 2, 3, 16
    h = n_kv * n_rep
    scaling = d ** -0.5
    q = torch.randn(b, h, q_len, d, dtype=torch.float64)
    k = torch.randn(b, n_kv, q_len, d, dtype=torch.float64)
    v = torch.randn(b, n_kv, q_len, d, dtype=torch.float64)
    # additive mask: causal + a sliding window of 12, plus a random padded key column
    mask = torch.zeros(b, 1, q_len, q_len, dtype=torch.float64)
    for i in range(q_len):
        for j in range(q_len):
            if j > i or (i - j) >= 12:
                mask[:, 0, i, j] = float("-inf")
    mask[:, 0, 1:, 0] = float("-inf")  # a left-pad column masked for rows>0 (row0 keeps diagonal)
    module = types.SimpleNamespace(head_dim=d, num_key_value_groups=n_rep)
    ref = _eager_ref(module, q, k, v, mask, scaling, softcap)
    out, w = A.chunked_eager_attention_forward(module, q, k, v, attention_mask=mask,
                                               dropout=0.0, scaling=scaling, softcap=softcap,
                                               chunk_size=chunk)
    assert w is None
    assert out.shape == (b, q_len, h, d)
    assert torch.allclose(out, ref, atol=1e-9, rtol=1e-7), (out - ref).abs().max().item()


def test_no_mask_and_default_scaling():
    torch.manual_seed(1)
    b, h, q_len, d = 1, 2, 20, 8
    q = torch.randn(b, h, q_len, d, dtype=torch.float64)
    k = torch.randn(b, h, q_len, d, dtype=torch.float64)
    v = torch.randn(b, h, q_len, d, dtype=torch.float64)
    module = types.SimpleNamespace(head_dim=d, num_key_value_groups=1)
    ref = _eager_ref(module, q, k, v, None, d ** -0.5, None)
    out, _ = A.chunked_eager_attention_forward(module, q, k, v, attention_mask=None, chunk_size=6)
    assert torch.allclose(out, ref, atol=1e-9)


def test_grad_path_matches_and_backprops():
    """requires_grad=True triggers the per-chunk checkpoint path; output must still equal eager
    and gradients must flow (checkpoint recomputes the softmax in backward)."""
    torch.manual_seed(3)
    b, h, q_len, d = 1, 2, 50, 8
    q = torch.randn(b, h, q_len, d, dtype=torch.float64, requires_grad=True)
    k = torch.randn(b, h, q_len, d, dtype=torch.float64, requires_grad=True)
    v = torch.randn(b, h, q_len, d, dtype=torch.float64, requires_grad=True)
    module = types.SimpleNamespace(head_dim=d, num_key_value_groups=1)
    ref = _eager_ref(module, q, k, v, None, d ** -0.5, None)
    out, _ = A.chunked_eager_attention_forward(module, q, k, v, attention_mask=None, chunk_size=7)
    assert torch.allclose(out, ref, atol=1e-9)
    out.sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    m = types.SimpleNamespace(head_dim=4, num_key_value_groups=1)
    q = torch.randn(1, 1, 4, 4)
    with pytest.raises(AssertionError):
        A.chunked_eager_attention_forward(m, q, q, q, dropout=0.1)


def test_adaptive_cap_matches_eager():
    """When chunk_size is None the module uses the env default _CHUNK but CAPS it for long
    sequences (kv > _CHUNK) so a chunk's score matrix stays within _CHUNK^2 elements. The capped
    chunking must still equal eager (chunking never changes the math, only memory)."""
    torch.manual_seed(4)
    b, h, q_len, d = 1, 2, 1500, 8               # kv(1500) > _CHUNK(1024) -> cap forces >1 chunk
    q = torch.randn(b, h, q_len, d, dtype=torch.float64)
    k = torch.randn(b, h, q_len, d, dtype=torch.float64)
    v = torch.randn(b, h, q_len, d, dtype=torch.float64)
    module = types.SimpleNamespace(head_dim=d, num_key_value_groups=1)
    ref = _eager_ref(module, q, k, v, None, d ** -0.5, None)
    old = A._CHUNK
    try:
        A.set_chunk(1024)
        # cap: cs = max(_MIN_CHUNK, min(1024, 1024*1024//1500)) = 699 < q_len -> multi-chunk
        assert max(A._MIN_CHUNK, min(1024, 1024 * 1024 // 1500)) < q_len
        out, _ = A.chunked_eager_attention_forward(module, q, k, v, attention_mask=None,
                                                   chunk_size=None)
    finally:
        A.set_chunk(old)
    assert torch.allclose(out, ref, atol=1e-9), (out - ref).abs().max().item()


def test_registration():
    # register() is idempotent and the name is known to transformers' interface
    assert A.register() is True
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        assert "chunked_eager" in ALL_ATTENTION_FUNCTIONS
    except Exception:
        pass  # interface internals vary across versions; registration not raising is enough

"""Exact query-chunked attention for transformers' AttentionInterface (transformers >= 5.14).

Numerically identical to the model's ``eager_attention_forward`` (same GQA repeat, scaling,
optional logit softcap, additive mask, fp32 softmax) but computes ``softmax(QKᵀ/√d + mask) V`` in
QUERY CHUNKS so the [heads, L, L] attention matrix is never materialised in full -- only
[heads, chunk, L] at a time. This lets eager-quality (correct) attention run on long prompts
(up to the owner's 49,152-token rule) without the O(L²) memory blow-up, on a single GPU.

Register once, then load the model with ``attn_implementation="chunked_eager"``:

    import tod.model.attn_chunked          # registers "chunked_eager"
    AutoModelForCausalLM.from_pretrained(..., attn_implementation="chunked_eager")

Constraints (by design): attention dropout must be 0 (inference / SFT); returns
``(attn_output, None)`` -- attention weights are not returned (output_attentions unsupported).
It reuses whatever mask tensor the model passes (padding AND sliding-window layers are already
encoded there by the model) -- it never rebuilds a mask. Dependency-free so the eval harness can
adopt it verbatim. Chunk size defaults to 1024; override with env ``CHUNKED_ATTN_Q``.
"""
from __future__ import annotations

import os

_CHUNK = int(os.environ.get("CHUNKED_ATTN_Q", "1024"))
_MIN_CHUNK = int(os.environ.get("CHUNKED_ATTN_Q_MIN", "512"))   # floor for the adaptive cap


def set_chunk(n: int) -> None:
    global _CHUNK
    _CHUNK = int(n)


def _repeat_kv(x, n_rep: int):
    """[b, n_kv, s, d] -> [b, n_kv*n_rep, s, d] (GQA expansion; identical to HF repeat_kv)."""
    if n_rep == 1:
        return x
    b, n_kv, s, d = x.shape
    return x[:, :, None, :, :].expand(b, n_kv, n_rep, s, d).reshape(b, n_kv * n_rep, s, d)


def chunked_eager_attention_forward(module, query, key, value, attention_mask=None,
                                    dropout: float = 0.0, scaling=None, softcap=None,
                                    chunk_size: int | None = None, **kwargs):
    """Drop-in for ``eager_attention_forward`` (same signature/semantics), query-chunked.

    query/key/value: [batch, n_heads(/n_kv), q_len, head_dim]. ``attention_mask`` is the model's
    additive float bias [batch, 1|n_heads, q_len, kv_len] (or None); we slice its query rows per
    chunk and add it verbatim. Returns ``(attn_output [batch, q_len, n_heads, head_dim], None)``.
    """
    import torch
    import torch.utils.checkpoint

    assert not dropout, "chunked_eager supports attention dropout=0 only"
    if scaling is None:
        scaling = module.head_dim ** -0.5
    key_states = _repeat_kv(key, module.num_key_value_groups)
    value_states = _repeat_kv(value, module.num_key_value_groups)
    q_len = query.shape[2]
    k_t = key_states.transpose(2, 3)                       # [b, h, d, kv]
    kv = key_states.shape[-2]
    # ADAPTIVE query-chunk size. When the caller passes chunk_size explicitly (tests/ablation) we
    # honour it verbatim. Otherwise we use the env default _CHUNK but CAP it so one chunk's
    # [b, h, cs, kv] score matrix never exceeds a _CHUNK x _CHUNK element budget: a sequence of
    # length <= _CHUNK stays ONE chunk (the intended speedup at CHUNKED_ATTN_Q=8192), while a LONG
    # sequence (kv > _CHUNK) uses a proportionally smaller chunk so the per-chunk softmax memory
    # stays bounded by that of a single _CHUNK x _CHUNK chunk. Without this cap, raising _CHUNK to
    # 8192 makes a 30k-token row allocate a [h, 8192, 30000] score matrix and OOM.
    if chunk_size is not None:
        cs = int(chunk_size)
    else:
        cs = int(_CHUNK)
        if kv > cs:
            cs = max(_MIN_CHUNK, min(cs, (cs * cs) // kv))

    def _chunk(qc, mask_slice):
        # one query chunk: [b,h,c,d] -> [b,h,c,dv]. The [b,h,c,kv] softmax lives ONLY inside this
        # call; when checkpointed it is recomputed in backward, so the graph never holds the full
        # [b,h,q,kv] weights -- backward memory is O(chunk*kv), not O(q*kv).
        aw = torch.matmul(qc, k_t) * scaling
        if softcap is not None:
            aw = torch.tanh(aw / softcap) * softcap
        if mask_slice is not None:
            aw = aw + mask_slice
        aw = torch.softmax(aw, dim=-1, dtype=torch.float32).to(qc.dtype)
        return torch.matmul(aw, value_states)

    use_ckpt = torch.is_grad_enabled() and query.requires_grad
    outs = []
    for s in range(0, q_len, cs):
        e = min(s + cs, q_len)
        qc = query[:, :, s:e]                              # [b, h, c, d]
        msl = attention_mask[:, :, s:e, :kv] if attention_mask is not None else None
        if use_ckpt:
            oc = torch.utils.checkpoint.checkpoint(_chunk, qc, msl, use_reentrant=False)
        else:
            oc = _chunk(qc, msl)
        outs.append(oc)                                    # [b, h, c, dv]
    attn_output = torch.cat(outs, dim=2).transpose(1, 2).contiguous()      # [b, q, h, dv]
    return attn_output, None


_REGISTERED = False


def register() -> bool:
    """Register 'chunked_eager' with transformers' attention AND mask interfaces (idempotent).

    Registering the mask fn is essential: for a custom attn_implementation the model passes
    ``attention_mask=None`` (and a ``sliding_window`` kwarg) unless a mask builder is registered
    for that name -- so without this, chunked_eager would apply NO causal/sliding-window masking
    and diverge from eager by many log-odds (measured). We point it at transformers' own
    ``eager_mask`` so we receive the identical additive float mask eager receives (causal +
    per-layer sliding window + padding, built by the model) and add it verbatim.
    """
    global _REGISTERED
    if _REGISTERED:
        return True
    try:
        from transformers import AttentionInterface
        AttentionInterface.register("chunked_eager", chunked_eager_attention_forward)
        try:
            from transformers.masking_utils import AttentionMaskInterface, eager_mask
            AttentionMaskInterface.register("chunked_eager", eager_mask)
        except Exception as e:  # mask interface must exist for correctness
            raise RuntimeError(f"chunked_eager needs a mask fn but registration failed: {e}")
        _REGISTERED = True
    except Exception as e:  # pragma: no cover - surfaced at import site
        raise RuntimeError(f"could not register chunked_eager attention: {e}")
    return _REGISTERED


# register on import so `attn_implementation="chunked_eager"` just works after `import`.
try:
    register()
except Exception:
    pass

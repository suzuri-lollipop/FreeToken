"""Exact rejection sampling for the MTP spec step under non-greedy decoding.

Greedy verification compares the draft id against the target's argmax. A sampled request
needs the rejection rule instead: keep the drafted token with probability
``min(1, q(y)/p(y))`` and, when it is refused, replace it with a draw from the residual
``(q - p)+``. That pair is what makes the emitted distribution exactly ``q`` (the target's)
regardless of how good or bad the draft head is -- an "accept when it looks likely"
heuristic would silently bias every sampled response, so the rule lives here, in one module
the eager sampled spec step calls.

``q`` and ``p`` must be the SAME distribution family on both sides: both get the request's
temperature, then top-k, then top-p, then renormalization. Truncating one side only would
push the acceptance ratio above 1 on the trimmed tail and re-introduce the bias.

Everything is plain torch over dense ``[B, V]`` rows. The uniforms arrive as arguments
rather than being drawn here, and the engine draws them on the host from a seeded
generator: every TP rank decides on the same ones (a device RNG would advance at a
different offset per rank and the ranks would emit different tokens), and the verify
trace prints the draws next to the verdict, so a traced step is reproducible from its log.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from freetoken.core import SamplingParams

# Temperature floor, matching Sampler.prepare: T=0 is the greedy path, never this one.
MIN_TEMPERATURE = 1e-6
# Top-p bounds, matching Sampler.prepare's min(max(top_p, MIN_P), 1.0): the API hands the
# knob over unvalidated, and an unclamped p <= 0 renorms to an all-zero row -- the draw then
# comes back as the vocab's last token instead of the target's top one.
MIN_TOP_P = 1e-6
# Guards a divide by an all-zero residual (q == p exactly, or a fully truncated row).
_EPS = 1e-20

SAMPLING_ENV = "FREETOKEN_MTP_SAMPLING"
DEBUG_ENV = "FREETOKEN_MTP_DEBUG"


def debug_traced() -> bool:
    """Whether the per-step MTP trace is on.

    ``0`` means off, like every other MTP switch: the trace reads device values and, for the
    timing line, synchronizes the device on each spec step -- so a setting that only LOOKED
    disabled would cost the very throughput it is there to measure.
    """
    return os.environ.get(DEBUG_ENV, "0") not in ("", "0")


def sampling_path_enabled() -> bool:
    """Whether sampled drafting may run at all, independent of any one request.

    ``FREETOKEN_MTP_SAMPLING=0`` restores the pre-P4 behaviour of declining every sampled
    request; it is read per call rather than at import so an A/B run does not need a code
    change, and the scheduler's startup line quotes the same switch.
    """
    return os.environ.get(SAMPLING_ENV, "1") != "0"


def sampling_enabled(params: "SamplingParams") -> bool:
    """Whether this request's params may run the sampled spec step.

    Greedy requests keep the argmax path (cheaper, and its exact match against non-spec
    greedy output is a shipped invariant), so this only answers for non-greedy ones.
    """
    if params.is_greedy:
        return False
    return sampling_path_enabled()


def _param(value, device, dtype, *, floor=None, ceiling=None, rows=1):
    """One sampling knob as the per-row device tensor the kernels index by row.

    The kernels read a knob per row, so a scalar broadcast to ``rows`` rather than being
    handed over as a single element: a shorter array is read out of bounds and comes back as
    NaN, which is why the row count is passed in rather than assumed to be one.
    """
    if isinstance(value, torch.Tensor):
        out = value.reshape(-1).to(device=device, dtype=dtype)
        if out.numel() == 1 and rows > 1:
            out = out.expand(rows)
        out = out.contiguous()
        if floor is not None or ceiling is not None:
            out.clamp_(float(floor) if floor is not None else -torch.inf,
                       float(ceiling) if ceiling is not None else torch.inf)
        return out
    out = float(value)
    if floor is not None:
        out = max(out, float(floor))
    if ceiling is not None:
        out = min(out, float(ceiling))
    return torch.full((rows,), out, dtype=dtype, device=device)


def spec_supported(params: "SamplingParams") -> bool:
    """Whether a request may draft at all: greedy compares ids, sampled compares densities."""
    return params.is_greedy or sampling_enabled(params)


def trunc_renorm_probs(
    logits: torch.Tensor,
    temperature,
    top_k=-1,
    top_p=1.0,
) -> torch.Tensor:
    """Sampled distribution of each logit row: temperature, then top-k, then top-p.

    Runs the SAME kernels :func:`freetoken.engine.sample.sample_impl` drives, so the ``q``
    the rejection test compares against is the distribution a non-speculative step samples
    from -- an independently written truncation would agree to within a boundary token and
    silently drift every sampled response.

    Both filters run whenever the row needs them and are skipped when the host value says it
    is a no-op (``top_k >= vocab`` / ``top_p == 1``), which is what the sampler itself does;
    a knob that arrives as a device tensor (a per-request override riding in on the payload)
    cannot be inspected here, so its filter runs with the value as given. Degenerate knobs
    are clamped to the same bounds Sampler.prepare applies (top_p into [MIN_TOP_P, 1.0]), so
    a raw ``top_p <= 0`` truncates to the target's top token here exactly as it does there.

    CPU tensors take a plain-torch reference path with the same rules -- it is what the unit
    tests pin the semantics with, and the fallback if a renorm kernel ever cannot be used.
    """
    x = logits.float()
    vocab = x.shape[-1]
    if x.device.type == "cpu":
        return _trunc_renorm_torch(x, temperature, top_k, top_p)
    from freetoken.kernel.backend import is_flashinfer_installed

    if is_flashinfer_installed():
        import flashinfer.sampling as sampling
    else:
        import freetoken.kernel.triton.sampling as sampling

    probs = sampling.softmax(
        x, _param(temperature, x.device, torch.float32, floor=MIN_TEMPERATURE, rows=x.shape[0]),
        enable_pdl=False,
    )
    skip_k = not isinstance(top_k, torch.Tensor) and (top_k is None or not 1 < top_k < vocab)
    skip_p = not isinstance(top_p, torch.Tensor) and (top_p is None or top_p >= 1.0)
    if not skip_k:
        probs = sampling.top_k_renorm_probs(
            probs, _param(top_k, x.device, torch.int32, floor=1, ceiling=vocab, rows=x.shape[0])
        )
    if not skip_p:
        probs = sampling.top_p_renorm_probs(
            probs, _param(top_p, x.device, torch.float32, floor=MIN_TOP_P, ceiling=1.0,
                          rows=x.shape[0])
        )
    return probs


def _as_host(value):
    return value.item() if isinstance(value, torch.Tensor) else value


def _trunc_renorm_torch(x, temperature, top_k, top_p):
    """Reference implementation of the two renorm kernels (CPU, no kernels available)."""
    vocab = x.shape[-1]
    t = max(_as_host(temperature), MIN_TEMPERATURE)
    k = int(_as_host(top_k))
    p = min(max(float(_as_host(top_p)), MIN_TOP_P), 1.0)
    k = vocab if k <= 0 or k > vocab else k
    values, indices = (x / t).topk(k, dim=-1)  # descending; a no-op sort when k == vocab
    probs = torch.softmax(values, dim=-1)
    if p < 1.0:
        probs = probs.masked_fill((probs.cumsum(-1) - probs) > p, 0.0)
        probs = probs / probs.sum(-1, keepdim=True).clamp_min(_EPS)
    return torch.zeros_like(x).scatter_(-1, indices, probs)


def draw_probs(probs: torch.Tensor, uniform: torch.Tensor) -> torch.Tensor:
    """Inverse-CDF draw of one token per row (``uniform`` is ``[rows]`` in [0, 1)).

    A single uniform broadcasts to every row; the engine hands one uniform per draw, its
    sampled step being a single row.
    """
    idx = (probs.cumsum(-1) < uniform.reshape(-1, 1)).sum(-1)
    return idx.clamp_(max=probs.shape[-1] - 1)


def rejection_sample(
    q_probs: torch.Tensor,
    p_probs: torch.Tensor,
    draft: torch.Tensor,
    uniform: torch.Tensor,
    residual_uniform: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One rejection step: keep the drafted token or replace it from the residual.

    ``q_probs`` is the target's distribution at the drafted position, ``p_probs`` the draft
    head's own distribution that produced ``draft``, and ``draft`` the id under test. The
    ratio clamps at 1 (a token the target likes more than the draft always passes), and a
    ``p`` of zero rejects instead of dividing: the residual then supplies the token.

    Returns ``(emit, accept, ratio)``, all ``[rows]`` tensors -- ``emit`` is the token to
    place at the verified position, which is the drafted id on acceptance and the residual
    draw on rejection.
    """
    q_at = q_probs.gather(-1, draft.reshape(-1, 1)).squeeze(-1)
    p_at = p_probs.gather(-1, draft.reshape(-1, 1)).squeeze(-1)
    # a p of zero means the draft head would never have proposed this id, so the ratio is
    # zero rather than the huge number the division would give against a clamped floor
    ratio = torch.where(
        p_at > 0, (q_at / p_at.clamp_min(_EPS)).clamp_(max=1.0), torch.zeros_like(p_at)
    )
    accept = uniform < ratio
    residual = (q_probs - p_probs).clamp_min_(0.0)
    residual = residual / residual.sum(-1, keepdim=True).clamp_min(_EPS)
    return torch.where(accept, draft, draw_probs(residual, residual_uniform)), accept, ratio


def expected_acceptance(q_probs: torch.Tensor, p_probs: torch.Tensor) -> torch.Tensor:
    """``sum(min(q, p))`` per row: the acceptance probability of a perfectly drawn draft.

    Debug-only -- it says what the acceptance rate COULD reach at these params, which is how
    a low measured rate is told apart from a broken draft head.
    """
    return torch.minimum(q_probs, p_probs).sum(-1)

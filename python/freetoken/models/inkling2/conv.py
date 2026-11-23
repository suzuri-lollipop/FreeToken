"""Inkling-2 short convolutions: depthwise kernel-4 causal convs with a per-request history.

Every visit runs four of them (K, V, attention output, FFN output) over one channel slab
of the ``inkling2_conv`` slot state. The full-attention layers' K/V convs add per-token
taps projected from the conv input, shared within each KV head. Math follows the
transformers inkling2 reference: fp32 compute, ``(conv(x) + x) / sqrt((dynamic?2:1)*K + 1)``,
history = the last K-1 conv inputs. The static weight is stored in LAG order (row 0 weights
the current token); the loader flips the checkpoint's conv1d ``[C, 1, K]`` form into it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, LinearReplicated
from freetoken.layers.linear import LinearColParallelMerged
from freetoken.distributed import get_tp_info
from freetoken.models.config import ModelConfig

from .args import CONV_STATE, Inkling2Args


@dataclass
class ConvMeta:
    """Per-forward conv inputs, built once and shared by every visit (the PLE metadata pattern).

    slots      [B] int64 device -- linear-state slot per request
    cu_seqlens [B+1] int64 device -- query indptr (decode: arange(B+1))
    seq_lens   host per-request token counts
    fresh      [B] bool device or None -- request starts a new sequence (zero history)
    is_decode  one token per request
    is_verify / rows_per_req -- a speculative verify batch: B requests x rows_per_req causal
               rows; the convs read the live state and stash every row's window for the commit
    track_dst / track_boundary_row -- hybrid-radix snapshot target (None on the naive cache)
    """

    slots: torch.Tensor
    cu_seqlens: torch.Tensor
    seq_lens: Sequence[int]
    fresh: torch.Tensor | None
    is_decode: bool
    track_dst: torch.Tensor | None = None
    track_boundary_row: torch.Tensor | None = None
    is_verify: bool = False
    rows_per_req: int = 1


def build_conv_meta(batch, device: torch.device) -> ConvMeta:
    verify = getattr(batch, "phase", None) == "verify"
    rows = getattr(batch, "rows_per_req", None) or 1
    fla = getattr(batch, "fla_metadata", None)
    if fla is not None:
        slots = fla.cache_indices.long()
        if batch.is_decode:
            bs = slots.numel()
            return ConvMeta(
                slots=slots,
                cu_seqlens=torch.arange(bs + 1, dtype=torch.int64, device=device),
                seq_lens=(1,) * bs,
                fresh=None,
                is_decode=True,
            )
        fresh = ~fla.has_initial_state if fla.has_initial_state is not None else None
        return ConvMeta(
            slots=slots,
            cu_seqlens=fla.cu_seqlens.long(),
            seq_lens=tuple(r.extend_len for r in batch.padded_reqs),
            fresh=fresh,
            is_decode=False,
            track_dst=fla.track_dst,
            track_boundary_row=fla.track_boundary_row,
            is_verify=verify,
            rows_per_req=rows,
        )
    # direct-op callers (tests) with no scheduler metadata
    reqs = batch.padded_reqs
    lens = [r.extend_len for r in reqs]
    pin = {"device": "cpu", "pin_memory": torch.cuda.is_available()}
    cu = torch.tensor([0, *lens], dtype=torch.int64, **pin).cumsum_(0).to(device, non_blocking=True)
    slots = torch.tensor([r.table_idx for r in reqs], dtype=torch.int64, **pin).to(device, non_blocking=True)
    fresh = torch.tensor([r.cached_len == 0 for r in reqs], dtype=torch.bool, **pin).to(device, non_blocking=True)
    return ConvMeta(slots=slots, cu_seqlens=cu, seq_lens=tuple(lens), fresh=fresh, is_decode=batch.is_decode,
                    is_verify=verify, rows_per_req=rows)


def conv_state_slab(visit: int, channels: int, offset: int) -> torch.Tensor:
    """This visit's conv history channels: ``[num_slots, channels, K-1]`` fp32."""
    pool = get_global_ctx().linear_state_pool
    assert pool is not None and pool.has_slot_state(CONV_STATE), "inkling2 needs the conv slot state (ModelConfig.slot_states)"
    return pool.slot_state(CONV_STATE, visit)[:, offset : offset + channels, :]


class Inkling2Conv(BaseOP):
    """One depthwise short conv; ``num_heads > 0`` adds the full-attention dynamic taps."""

    def __init__(self, config: ModelConfig, channels: int, *, num_heads: int = 0, prefix: str = ""):
        args: Inkling2Args = config.inkling2_args
        self.kernel = args.conv_kernel
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads if num_heads else 0
        # lag order, fp32: the reference runs every conv module in fp32
        self.weight = torch.empty(channels, self.kernel, dtype=torch.float32)
        self.out_scale = 1.0 / math.sqrt((2 if num_heads else 1) * self.kernel + 1)
        if num_heads:
            tp = get_tp_info().size
            self.conv_proj = LinearColParallelMerged(
                config.hidden_size, [num_heads * tp * self.kernel],
                has_bias=False, quant_config=config.quant, prefix=f"{prefix}.conv_proj",
            )

    def _taps(self, dyn: torch.Tensor | None, tokens: int) -> torch.Tensor:
        """``[T, C, K]`` fp32 taps in lag order (static, plus per-token head-shared dynamic)."""
        w = self.weight.unsqueeze(0)
        if dyn is None:
            return w.expand(tokens, -1, -1)
        dyn = dyn.view(tokens, self.num_heads, self.kernel)
        dyn = dyn.unsqueeze(2).expand(-1, -1, self.head_dim, -1).reshape(tokens, self.channels, self.kernel)
        return w + dyn

    def forward(self, x: torch.Tensor, meta: ConvMeta, states: torch.Tensor, dyn_input: torch.Tensor | None = None,
                *, verify_key: tuple[int, int] | None = None) -> torch.Tensor:
        dyn = self.conv_proj.forward(dyn_input).float() if self.num_heads else None
        if meta.is_verify:
            assert verify_key is not None, "a verify conv needs its (visit, channel offset) slab key"
            out = self._verify(x, meta, states, dyn, verify_key)
        elif meta.is_decode:
            if x.is_cuda:
                from freetoken.kernel.triton.ik2_conv import conv_decode_fused

                # the dynamic taps are head-shared: expand [T, heads*K] to [T, C, K] as _taps does
                dyn_e = None
                if dyn is not None:
                    t = x.shape[0]
                    dyn_e = dyn.view(t, self.num_heads, self.kernel).unsqueeze(2).expand(
                        -1, -1, self.head_dim, -1).reshape(t, self.channels, self.kernel)
                # one kernel: window . taps, history shift, residual + out_scale
                return conv_decode_fused(x, states, meta.slots, self.weight, dyn_e, meta.fresh, self.out_scale)
            out = self._decode(x, meta, states, dyn)
        else:
            out = self._prefill(x, meta, states, dyn)
        return ((out + x.float()) * self.out_scale).to(x.dtype)

    def _verify(self, x: torch.Tensor, meta: ConvMeta, states: torch.Tensor, dyn: torch.Tensor | None,
                verify_key: tuple[int, int]) -> torch.Tensor:
        """Every verify row's causal window over [live history | the block's inputs], the live
        state read-only; the stash lets the commit roll the history to the accepted rows only.
        Device ops only, so a verify graph captures it (the PLE conv verify pattern)."""
        visit, offset = verify_key
        bs, rows = meta.slots.numel(), meta.rows_per_req
        handler = get_global_ctx().linear_state_pool.verify_handler(CONV_STATE)
        history = handler.history[visit][:bs, offset : offset + self.channels]
        history[..., : self.kernel - 1].copy_(self._read_state(meta, states))
        history[..., self.kernel - 1 :].copy_(x.view(bs, rows, self.channels).transpose(1, 2))
        # window[r, lag] = packed column (K-1) + r - lag: lag 0 the row itself, lag K-1 the oldest
        cols = (self.kernel - 1) + torch.arange(rows, device=x.device).unsqueeze(1) - torch.arange(self.kernel, device=x.device)
        packed = history.gather(2, cols.reshape(1, 1, -1).expand(bs, self.channels, -1))
        windows = packed.view(bs, self.channels, rows, self.kernel).permute(0, 2, 1, 3).reshape(bs * rows, self.channels, self.kernel)
        taps = self._taps(dyn, x.shape[0])
        return (windows * taps).sum(-1)

    def _read_state(self, meta: ConvMeta, states: torch.Tensor) -> torch.Tensor:
        state = states.index_select(0, meta.slots)
        if meta.fresh is not None:
            state = torch.where(meta.fresh.view(-1, 1, 1), torch.zeros_like(state), state)
        return state

    def _decode(self, x: torch.Tensor, meta: ConvMeta, states: torch.Tensor, dyn: torch.Tensor | None) -> torch.Tensor:
        """Batched tap read, device ops only so a decode graph captures it."""
        state = self._read_state(meta, states)
        # windows[..., lag]: lag 0 is the current token, lag K-1 the oldest history column
        windows = torch.cat([x.float().unsqueeze(-1), state.flip(-1)], dim=-1)
        taps = self._taps(dyn, x.shape[0])
        out = (windows * taps).sum(-1)
        new_state = torch.cat([state[..., 1:], x.float().unsqueeze(-1)], dim=-1)
        states.index_copy_(0, meta.slots, new_state)
        return out

    def _prefill(self, x: torch.Tensor, meta: ConvMeta, states: torch.Tensor, dyn: torch.Tensor | None) -> torch.Tensor:
        """One packed conv over ``[state_0 | chunk_0 | state_1 | chunk_1 | ...]`` (the PLE packing)."""
        lens = list(meta.seq_lens)
        km1 = self.kernel - 1
        num_reqs, width, tokens = len(lens), self.channels, x.shape[0]
        out_index, state_index, next_state_index, window_index = self._prefill_indices(lens, tokens, x.device)

        state = self._read_state(meta, states)
        history = x.float().new_empty(width, tokens + num_reqs * km1)
        history.index_copy_(1, state_index, state.permute(1, 0, 2).reshape(width, -1))
        history.index_copy_(1, out_index + km1, x.float().transpose(0, 1).contiguous())

        # window_index[t, lag] is the packed-history column of token t-lag
        windows = history.index_select(1, window_index.reshape(-1)).view(width, tokens, self.kernel).permute(1, 0, 2)
        taps = self._taps(dyn, tokens)
        out = (windows * taps).sum(-1)

        new_state = history.index_select(1, next_state_index).view(width, num_reqs, km1).permute(1, 0, 2)
        states.index_copy_(0, meta.slots, new_state.contiguous())
        if meta.track_dst is not None and meta.track_boundary_row is not None:
            self._write_track_snapshot(x, meta, states)
        return out

    def _write_track_snapshot(self, x: torch.Tensor, meta: ConvMeta, states: torch.Tensor) -> None:
        km1 = self.kernel - 1
        src = meta.track_boundary_row.unsqueeze(1) + torch.arange(-km1, 0, device=x.device)
        window = x.float()[src].transpose(-1, -2).contiguous()
        states.index_copy_(0, meta.track_dst, window)

    def _prefill_indices(self, lens: List[int], tokens: int, device: torch.device) -> Tuple[torch.Tensor, ...]:
        km1 = self.kernel - 1
        cpu = torch.device("cpu")
        counts = torch.tensor(lens, dtype=torch.int64, device=cpu)
        cu = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
        pad = torch.arange(len(lens), dtype=torch.int64, device=cpu) * km1
        base = cu[:-1] + pad
        out_index = torch.arange(tokens, device=cpu) + torch.repeat_interleave(pad, counts)
        span = torch.arange(km1, dtype=torch.int64, device=cpu)
        # per-token window columns: token t of req i at local pos p reads base_i + km1 + p - lag
        local_pos = torch.arange(tokens, device=cpu) - torch.repeat_interleave(cu[:-1], counts)
        req_base = torch.repeat_interleave(base, counts)
        lags = torch.arange(self.kernel, dtype=torch.int64, device=cpu)
        window_index = (req_base + km1 + local_pos).unsqueeze(1) - lags.unsqueeze(0)
        packed = torch.cat(
            [
                out_index,
                (base.unsqueeze(1) + span).reshape(-1),
                ((base + counts).unsqueeze(1) + span).reshape(-1),
                window_index.reshape(-1),
            ]
        )
        if torch.cuda.is_available():
            packed = packed.pin_memory()
        packed = packed.to(device, non_blocking=True)
        n_out, n_state, n_win = out_index.numel(), len(lens) * km1, window_index.numel()
        return (
            packed[:n_out],
            packed[n_out : n_out + n_state],
            packed[n_out + n_state : n_out + 2 * n_state],
            packed[n_out + 2 * n_state : n_out + 2 * n_state + n_win].view(tokens, self.kernel),
        )


def short_conv_reference(
    x: torch.Tensor,
    seq_lens: Sequence[int],
    state: torch.Tensor,
    weight_lag: torch.Tensor,
    kernel: int,
    dynamic: torch.Tensor | None = None,
    num_heads: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-request F.conv1d transcription of the HF conv; the oracle the packed paths are diffed against.

    ``x`` [T, C] bf16, ``state`` [B, C, K-1] fp32 (oldest first), ``weight_lag`` [C, K] fp32
    (lag order); returns (output [T, C] in x's dtype, new state [B, C, K-1]).
    """
    input_dtype = x.dtype
    xf = x.float()
    outs, new_states = [], []
    offset = 0
    for i, n in enumerate(seq_lens):
        chunk = xf[offset : offset + n].transpose(0, 1).unsqueeze(0)
        history = torch.cat([state[i : i + 1], chunk], dim=-1)
        if dynamic is None:
            # conv1d order: weight[..., k] weights the token k steps from the window start (oldest)
            w = weight_lag.flip(-1).view(-1, 1, kernel)
            out = F.conv1d(history, w, groups=history.shape[1])
        else:
            head_dim = history.shape[1] // num_heads
            dyn = dynamic[offset : offset + n].view(n, num_heads, kernel)
            taps = weight_lag.view(num_heads, head_dim, kernel).unsqueeze(0) + dyn.unsqueeze(2)
            windows = history.unfold(-1, kernel, 1).flip(-1)  # [1, C, n, K] lag order
            windows = windows.view(1, num_heads, head_dim, n, kernel).permute(3, 1, 2, 0, 4).reshape(n, num_heads, head_dim, kernel)
            out = (windows * taps).sum(-1).reshape(n, -1).transpose(0, 1).unsqueeze(0)
        outs.append(out.squeeze(0).transpose(0, 1))
        new_states.append(history[0, :, -(kernel - 1) :])
        offset += n
    scale = 1.0 / math.sqrt((2 if dynamic is not None else 1) * kernel + 1)
    out = (torch.cat(outs, dim=0) + xf) * scale
    return out.to(input_dtype), torch.stack(new_states, dim=0)


class Inkling2ConvVerify:
    """``SlotVerify`` of the conv history: per visit and verify row, the live history then
    every verify conv input (the PLEConvVerify pattern; the four convs of a visit share the
    slab, each stashing its own channels)."""

    def __init__(self, spec, max_bs: int, rows_per_req: int, device: torch.device, dtype: torch.dtype) -> None:
        self.name = spec.name
        width, self.state_len = spec.shape
        self.history = {
            layer_id: torch.zeros(max_bs, width, self.state_len + rows_per_req, dtype=dtype, device=device)
            for layer_id in spec.layer_ids
        }

    @staticmethod
    def nbytes(spec, max_bs: int, rows_per_req: int, dtype: torch.dtype) -> int:
        width, state_len = spec.shape
        return len(spec.layer_ids) * max_bs * width * (state_len + rows_per_req) * dtype.itemsize

    def commit(self, pool, slots: torch.Tensor, counts: torch.Tensor) -> None:
        # after n accepted rows the history is columns n .. n + state_len - 1
        cols = (counts.unsqueeze(1) + torch.arange(self.state_len, device=counts.device)).unsqueeze(1)
        for layer_id, history in self.history.items():
            states = pool.slot_state(self.name, layer_id)
            window = history[: slots.numel()].gather(2, cols.expand(-1, history.shape[1], -1))
            states.index_copy_(0, slots, window.to(states.dtype))


__all__ = ["ConvMeta", "Inkling2Conv", "Inkling2ConvVerify", "build_conv_meta", "conv_state_slab", "short_conv_reference"]

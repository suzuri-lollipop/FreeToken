"""The expert profile: a ranking of every (layer, expert) pair of the routed MoE.

The RAM tier of a disk-split boot is only as good as the experts it pins. Whole-layer
placement treats every expert of a layer alike, so a uniform router gets the same
coverage either way; real routing is heavy-tailed, and a profile measured on texts
like the workload's concentrates the same bytes on the experts the tokens actually
route to. The ranking is the whole file: the engine takes pairs from the top while
the host RAM budget lasts and leaves the rest on the memory-mapped banks.

The format is ours (FTXP): a 16-byte header (magic, version, num_layers,
num_experts, num_pairs) followed by ``num_pairs`` little-endian (layer, expert)
uint32 pairs, hottest first. A profile may rank fewer pairs than exist; the
unranked tail is implicitly coldest. Write one with ``ft-profile`` or
:func:`write_profile`; build the ranking from routing counts with
:func:`rank_by_counts`, or bootstrap with :func:`round_robin` (max coverage under
a uniform prior, no measurement behind it).
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

_MAGIC = b"FTXP"
_VERSION = 1
_HEADER = struct.Struct("<4sIIII")
_PAIR = struct.Struct("<II")


def round_robin(num_layers: int, num_experts: int, *, first_layer: int = 0) -> list[tuple[int, int]]:
    """Every pair, expert-major then layer-ascending, so any prefix spreads evenly.

    Without a trace this is the best a profile can be: holding the first ``k``
    pairs pins ``ceil(k / num_layers)`` experts of EVERY layer rather than whole
    layers, and coverage under a uniform router is the same either way. A measured
    ranking (see :func:`rank_by_counts`) is what makes the profile pay.
    """
    if num_layers < 1 or num_experts < 1:
        raise ValueError(f"empty profile: {num_layers} layers x {num_experts} experts")
    if not 0 <= first_layer < num_layers:
        raise ValueError(f"first_layer {first_layer} outside 0..{num_layers - 1}")
    return [(l, e) for e in range(num_experts) for l in range(first_layer, num_layers)]


def rank_by_counts(counts: list[list[int]], *, first_layer: int = 0) -> list[tuple[int, int]]:
    """Rank every pair by descending routing count (stable: layer, expert order).

    ``counts[layer][expert]`` is how often that expert was routed (the engine's
    routing-statistics dump, or any offline tally). Pairs never routed keep their
    relative order at the tail -- they are the complement's cold end.
    """
    ranked = []
    for l, layer_counts in enumerate(counts):
        if l < first_layer:
            continue
        for e, c in enumerate(layer_counts):
            ranked.append((-int(c), l, e))
    ranked.sort()
    return [(l, e) for _, l, e in ranked]


def write_profile(path: str | Path, ranked: list[tuple[int, int]], *, num_layers: int, num_experts: int) -> None:
    """Write ``ranked`` (hottest first; duplicates dropped) as an FTXP profile."""
    seen: set[tuple[int, int]] = set()
    pairs = []
    for l, e in ranked:
        if not 0 <= l < num_layers or not 0 <= e < num_experts:
            raise ValueError(f"pair ({l}, {e}) outside {num_layers} layers x {num_experts} experts")
        if (l, e) in seen:
            continue
        seen.add((l, e))
        pairs.append((l, e))
    Path(path).write_bytes(
        _HEADER.pack(_MAGIC, _VERSION, num_layers, num_experts, len(pairs))
        + b"".join(_PAIR.pack(l, e) for l, e in pairs)
    )


def load_profile(path: str | Path) -> tuple[list[tuple[int, int]], int, int]:
    """Return (ranked pairs, num_layers, num_experts) of an FTXP profile file."""
    data = Path(path).read_bytes()
    if len(data) < _HEADER.size:
        raise ValueError(f"{path}: {len(data)} bytes is shorter than an FTXP header")
    magic, version, num_layers, num_experts, num_pairs = _HEADER.unpack_from(data)
    if magic != _MAGIC:
        raise ValueError(f"{path}: not an FTXP profile (magic {magic!r})")
    if version != _VERSION:
        raise ValueError(f"{path}: FTXP version {version} (this build reads {_VERSION})")
    if num_layers < 1 or num_experts < 1:
        raise ValueError(f"{path}: {num_layers} layers x {num_experts} experts")
    off = _HEADER.size
    if len(data) < off + num_pairs * _PAIR.size:
        raise ValueError(f"{path}: header promises {num_pairs} pairs, the file holds fewer")
    pairs = [_PAIR.unpack_from(data, off + i * _PAIR.size) for i in range(num_pairs)]
    seen = set(pairs)
    if len(seen) != len(pairs):
        raise ValueError(f"{path}: {len(pairs) - len(seen)} duplicate (layer, expert) pairs")
    return pairs, num_layers, num_experts


def take_prefix_until_budget(
    ranked: list[tuple[int, int]], *, bytes_per_expert: int, budget: int
) -> list[tuple[int, int]]:
    """The hottest pairs whose expert bytes fit ``budget``, in rank order.

    ``bytes_per_expert`` is the per-expert bank size of one layer; a profile that
    spans layers of differing sizes pays the same for every pair here (the GLM
    banks are uniform), and the caller sizes the budget accordingly.
    """
    if bytes_per_expert <= 0:
        raise ValueError(f"bytes_per_expert {bytes_per_expert} must be positive")
    taken: list[tuple[int, int]] = []
    used = 0
    for l, e in ranked:
        if used + bytes_per_expert > budget:
            break
        taken.append((l, e))
        used += bytes_per_expert
    return taken


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ft-profile", description="Write or inspect an FTXP expert profile."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("write-round-robin", help="bootstrap profile: all pairs, evenly spread")
    mk.add_argument("out")
    mk.add_argument("--num-layers", type=int, required=True, help="MoE layer count of the model")
    mk.add_argument("--num-experts", type=int, required=True)
    mk.add_argument("--first-layer", type=int, default=0, help="skip dense-replaced layers below this")
    rd = sub.add_parser("inspect", help="print a profile's header and hottest pairs")
    rd.add_argument("path")
    rd.add_argument("--top", type=int, default=10)
    args = parser.parse_args(argv)

    if args.cmd == "write-round-robin":
        ranked = round_robin(args.num_layers, args.num_experts, first_layer=args.first_layer)
        write_profile(args.out, ranked, num_layers=args.num_layers, num_experts=args.num_experts)
        print(f"wrote {args.out}: {len(ranked)} pairs of {args.num_layers} x {args.num_experts}")
    else:
        ranked, num_layers, num_experts = load_profile(args.path)
        print(f"{args.path}: {num_layers} layers x {num_experts} experts, {len(ranked)} pairs ranked")
        for l, e in ranked[: args.top]:
            print(f"  rank {(l, e)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

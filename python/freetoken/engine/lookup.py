"""Prompt-lookup (suffix) drafting: the no-weights drafter Strata ships as ``--suffix-draft``.

When the text being written repeats an earlier stretch of the context -- a code edit quoting the
file, a refactor, a quoted patch -- the next token is usually the one that followed the same
suffix last time. A trigram index over the history (prompt + accepted output) finds the most
recent occurrence of the current suffix, extends the match backwards, and proposes the tokens
that followed it. ``--speculative lookup`` feeds the proposal to the same extend verify the MTP
head uses, widened to the proposal's length; a full hit emits every draft plus the target's next
token in one pass, and every emitted token is the target's own argmax. A partial hit keeps only
the verified first token (the GDN state after row 0 is the one snapshot the verify saves).

This is a direct port of Strata's ``strata::spec::SuffixDrafter``
(``include/strata/spec/suffix_drafter.hpp`` / ``src/spec/suffix_drafter.cpp``): every trigram
maps to its ``WAYS`` most recent end positions in a bounded open-addressing table, a proposal
extends each candidate backwards up to ``max_match`` and takes the longest (newest on ties),
and matches shorter than ``min_match`` propose nothing. The Python table is a dict rather than a
fixed slot array; ``capacity_tokens`` keeps the same memory bound by refusing to index new
trigrams once full (existing trigrams keep updating, so a long history still drafts on repeats
of anything it has already seen).

``LookupState`` is the per-request wrapper the scheduler holds: it mirrors committed tokens into
the index incrementally (``sync``), proposes up to ``max_draft`` tokens, and applies Strata's
acceptance/cost gate -- a request whose drafts keep missing stops paying for the wider verify.
"""

from __future__ import annotations

import os
from typing import Iterable, Sequence

# Trigrams are the unit Strata indexes. Shorter suffixes match too often to be evidence.
MIN_MATCH = 3
# Do not draft on a short history: below this many placed tokens the decode step is cheap and
# the wider verify's fixed cost dominates. Measured on a 32-token haiku prompt: drafting there
# cost ~7%; with the floor the request decodes at baseline speed. Copy workloads (275+ tokens)
# are unaffected.
MIN_CONTEXT = 256
# Proposal width. Depth 1 verifies two target rows for one draft and cannot pay for itself on
# GDN decodes (measured 2.13 decode steps per verify, so even acceptance 1.0 loses); a wider
# extend amortizes the fixed cost of the step. Four drafts measured +38% on a copy-heavy code
# edit and +31% on prose at acceptance ~0.85, with five close behind.
MAX_DRAFT = 4
# Candidates kept per trigram, newest first (Strata WAYS = 4).
WAYS = 4
# Bounded index: ~one dict entry per history token up to this many unique trigrams.
CAPACITY = 1 << 17
# Acceptance gate. With K=4 and a ~2.4-step verify, the all-or-nothing payout is
# 1 + 4*p^4 tokens per verify, so break-even is p ~= 0.77; 0.8 keeps a margin. Below the floor
# the request returns to plain decoding (the index is cheap to rebuild if it ever qualifies).
MIN_ACCEPTANCE = 0.8
ACCEPT_WINDOW = 128

_ENV = "FREETOKEN_LOOKUP_"


def lookup_debug() -> bool:
    """Whether the per-step lookup trace is on. Reads truthiness, so ``0`` is off."""
    return os.environ.get(_ENV + "DEBUG", "0") not in ("", "0")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(_ENV + name, "")
    try:
        return int(raw) if raw.strip() else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(_ENV + name, "")
    try:
        return float(raw) if raw.strip() else default
    except ValueError:
        return default


class SuffixDrafter:
    """Strata's suffix-lookup drafter: trigram index over the history, no model, no GPU."""

    def __init__(
        self,
        min_match: int = MIN_MATCH,
        max_match: int = 32,
        capacity_tokens: int = CAPACITY,
    ) -> None:
        # Strata clamps min_match to 3 and max_match to at least min_match.
        self.min_match = max(MIN_MATCH, int(min_match))
        self.max_match = max(self.min_match, int(max_match))
        self.capacity = max(1, int(capacity_tokens))
        self._hist: list[int] = []
        self._table: dict[int, list[int]] = {}
        self._last_match = 0

    # ------------------------------------------------------------------ index

    def __len__(self) -> int:
        return len(self._hist)

    @property
    def last_match(self) -> int:
        """Length of the match behind the last proposal (0 when none)."""
        return self._last_match

    @property
    def history(self) -> Sequence[int]:
        return self._hist

    @staticmethod
    def _key_at(hist: list[int], end: int) -> int:
        """A packed trigram key. Token ids are < 2**18 today; 18 bits each keeps it exact."""
        a, b, c = hist[end - 2], hist[end - 1], hist[end]
        return (a << 36) | (b << 18) | c

    def append(self, tokens: Iterable[int]) -> None:
        """Add tokens to the history (the prompt, then every accepted token)."""
        for token in tokens:
            self._hist.append(int(token))
            end = len(self._hist) - 1
            if end < 2:
                continue
            key = self._key_at(self._hist, end)
            positions = self._table.get(key)
            if positions is None:
                if len(self._table) >= self.capacity:
                    # The table is full: keep serving the trigrams it already knows, but do not
                    # grow it. Strata's fixed table behaves the same way (find_slot returns
                    # nullptr once full), which only costs proposals on never-seen suffixes.
                    continue
                self._table[key] = [end]
                continue
            positions.insert(0, end)
            if len(positions) > WAYS:
                del positions[WAYS:]

    # -------------------------------------------------------------- proposal

    def propose(self, max_k: int) -> list[int]:
        """Up to ``max_k`` tokens that followed the longest earlier occurrence of the suffix."""
        self._last_match = 0
        n = len(self._hist)
        if n < 4 or max_k <= 0:
            return []
        cur = n - 1
        positions = self._table.get(self._key_at(self._hist, cur))
        if not positions:
            return []
        best_end = 0
        best_len = 0
        # Newest first, so a tie on match length keeps the more recent occurrence (Strata).
        for p in positions:
            if p >= cur:
                continue  # the current suffix itself
            length = 0
            while (
                length < self.max_match
                and length <= p
                and self._hist[p - length] == self._hist[cur - length]
            ):
                length += 1
            if length > best_len:
                best_len = length
                best_end = p
        if best_len < self.min_match:
            return []
        self._last_match = best_len
        # The continuation may run into the current suffix (periodic text); reading history up
        # to `cur` is valid, exactly as Strata's propose() does.
        out: list[int] = []
        for q in range(best_end + 1, cur + 1):
            if len(out) >= max_k:
                break
            out.append(self._hist[q])
        return out

    def propose_after(self, extra: Iterable[int], max_k: int) -> list[int]:
        """``propose`` with ``extra`` appended to the suffix but not to the index.

        Strata's ``--lookup-chain``: continue a lookup where another drafter's proposal ends.
        The history itself is unchanged.
        """
        extra = list(extra)
        if not extra:
            return self.propose(max_k)
        n0 = len(self._hist)
        self._hist.extend(int(t) for t in extra)
        try:
            return self.propose(max_k)
        finally:
            del self._hist[n0:]


class LookupState:
    """Per-request drafter + the acceptance gate the scheduler consults."""

    def __init__(
        self,
        *,
        min_match: int = MIN_MATCH,
        max_match: int = 32,
        capacity_tokens: int = CAPACITY,
        max_draft: int = MAX_DRAFT,
        accept_window: int = ACCEPT_WINDOW,
        min_acceptance: float = MIN_ACCEPTANCE,
        min_context: int = MIN_CONTEXT,
    ) -> None:
        self.drafter = SuffixDrafter(min_match, max_match, capacity_tokens)
        self.max_draft = max(1, int(max_draft))
        self.min_context = max(0, int(min_context))
        self.accept_window = max(8, int(accept_window))
        self.min_acceptance = max(0.0, float(min_acceptance))
        self._indexed = 0
        self.off = False
        self.off_reason = ""
        self.offered = 0
        self.accepted = 0
        self.window_offered = 0
        self.window_accepted = 0

    @classmethod
    def from_env(cls) -> "LookupState":
        return cls(
            min_match=_env_int("MIN_MATCH", MIN_MATCH),
            max_match=_env_int("MAX_MATCH", 32),
            capacity_tokens=_env_int("CAPACITY", CAPACITY),
            max_draft=_env_int("MAX_DRAFT", MAX_DRAFT),
            accept_window=_env_int("ACCEPT_WINDOW", ACCEPT_WINDOW),
            min_acceptance=_env_float("MIN_ACCEPTANCE", MIN_ACCEPTANCE),
            min_context=_env_int("MIN_CONTEXT", MIN_CONTEXT),
        )

    # ----------------------------------------------------------------- index

    def sync(self, tokens, n: int) -> None:
        """Index committed tokens up to ``n`` (the request's device length)."""
        n = min(int(n), len(tokens))
        if n <= self._indexed:
            return
        chunk = tokens[self._indexed : n]
        if hasattr(chunk, "tolist"):
            chunk = chunk.tolist()
        self.drafter.append(chunk)
        self._indexed = n

    def propose(self) -> int | None:
        """One draft token, or None when the request is off / the suffix does not match."""
        out = self.propose_many(1)
        return out[0] if out else None

    def propose_many(self, max_draft: int | None = None) -> list[int]:
        """Up to ``max_draft`` draft tokens (all-or-nothing at the verify)."""
        if self.off:
            return []
        return self.drafter.propose(self.max_draft if max_draft is None else max_draft)

    @property
    def last_match(self) -> int:
        return self.drafter.last_match

    @property
    def acceptance(self) -> float | None:
        return self.accepted / self.offered if self.offered else None

    # ------------------------------------------------------------- acceptance

    def record(self, accepted: bool) -> bool:
        """Fold one single-draft verify into the gate."""
        return self.record_many(int(accepted), 1)

    def record_many(self, accepted: int, drafts: int) -> bool:
        """Fold ``accepted`` of ``drafts`` verified drafts into the gate.

        Returns False when drafting has just switched off. A multi-draft verify counts every
        draft here, so the floor keeps meaning "fraction of drafts that landed" rather than
        "fraction of verify batches that fully landed"."""
        if self.off:
            return False
        self.offered += drafts
        self.window_offered += drafts
        self.accepted += accepted
        self.window_accepted += accepted
        if self.window_offered < self.accept_window:
            return True
        rate = self.window_accepted / self.window_offered
        self.window_offered = 0
        self.window_accepted = 0
        if self.min_acceptance > 0 and rate < self.min_acceptance:
            self.off = True
            self.off_reason = f"low_acceptance({rate:.2f})"
            return False
        return True
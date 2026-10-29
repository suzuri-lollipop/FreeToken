"""Strata's prompt-lookup drafter, ported case-for-case.

The unit cases come from ``module/strata/src/spec/suffix_drafter_test.cpp`` so the Python
port and the C++ source answer the same questions: a repeated passage is proposed verbatim, a
longer older match beats a newer shorter one, ``min_match`` is honoured, periodic text may
continue into its own suffix, a full table keeps serving what it indexed, and
``propose_after`` extends a match without mutating the history.

``LookupState`` adds the scheduler-facing pieces FreeToken needs: incremental indexing from a
request's committed tokens, a single-draft proposal, and the acceptance gate that stops paying
for a second target row when the drafts keep missing.
"""

from __future__ import annotations

import torch

from freetoken.engine.lookup import LookupState, SuffixDrafter


def test_repeated_passage_is_proposed_verbatim():
    d = SuffixDrafter()
    d.append([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    d.append([10, 11, 12])
    assert d.propose(5) == [13, 14, 15, 16, 17]
    assert d.last_match == 3


def test_no_earlier_occurrence_proposes_nothing():
    d = SuffixDrafter()
    d.append([1, 2, 3, 4, 5, 6, 7])
    assert d.propose(4) == []
    assert d.last_match == 0


def test_longest_match_wins_over_the_most_recent():
    d = SuffixDrafter()
    d.append([7, 8, 1, 2, 3, 100, 101, 9, 1, 2, 3, 200, 201, 50, 7, 8, 1, 2, 3])
    assert d.propose(2) == [100, 101]
    assert d.last_match == 5


def test_min_match_is_respected():
    d = SuffixDrafter(min_match=4)
    d.append([1, 2, 3, 9, 5, 2, 3])
    assert d.propose(2) == []


def test_periodic_text_may_continue_into_the_current_suffix():
    d = SuffixDrafter()
    d.append([1, 2, 3, 1, 2, 3, 1, 2, 3])
    out = d.propose(6)
    assert len(out) >= 3 and out[:3] == [1, 2, 3]


def test_a_full_table_still_proposes_from_what_it_indexed():
    d = SuffixDrafter(capacity_tokens=1024)
    d.append([i % 997 for i in range(5000)])
    assert d.propose(4)


def test_propose_after_extends_without_mutating_history():
    d = SuffixDrafter(capacity_tokens=256)
    hist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 50, 51, 52, 1, 2]
    d.append(hist)
    out = d.propose_after([3, 4], 3)
    assert out == [5, 6, 7]
    assert d.last_match >= 4
    assert len(d) == len(hist)
    assert d.propose_after([99, 98, 97], 3) == []
    assert d.propose(3) == []  # the suffix (52, 1, 2) has no earlier occurrence
    assert d.propose_after([], 3) == d.propose(3)


def test_state_indexes_incrementally_and_proposes_the_repeat():
    state = LookupState()
    tokens = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3], dtype=torch.int32)
    state.sync(tokens, 12)
    assert state.propose() == 4
    assert state.last_match == 3
    # Nothing new to index, and no repeated suffix yet for the tail (1, 2, 3) twice in a row.
    state.drafter.append([4, 5, 6])
    assert state.propose() == 7


def test_state_gate_turns_off_below_the_acceptance_floor():
    state = LookupState(accept_window=8, min_acceptance=0.5)
    for _ in range(7):
        assert state.record(False)
    assert not state.off
    assert not state.record(False)  # 8 offers, 0 accepted -> off
    assert state.off and "low_acceptance" in state.off_reason
    assert state.propose() is None
    assert state.offered == 8 and state.accepted == 0


def test_state_gate_stays_on_above_the_floor_and_reports():
    state = LookupState(accept_window=8, min_acceptance=0.5)
    for i in range(8):
        assert state.record(i < 5)  # 5/8 = 0.625
    assert not state.off
    assert state.acceptance == 5 / 8
    # the window resets: another 4 accepts are enough for the next 8-offer window
    for _ in range(4):
        assert state.record(True)
    assert not state.off


def test_state_off_short_circuits_indexing_and_recording():
    state = LookupState(accept_window=8, min_acceptance=1.0)
    for _ in range(8):
        state.record(False)
    assert state.off
    assert not state.record(True)  # no window to fold into once off
    assert state.offered == 8

def test_state_wrapper_propose_after_respects_the_off_switch():
    """The chain calls the STATE wrapper, not the drafter: a gate that ended drafting for the
    request must also stop its continuations from widening an MTP verify."""
    from freetoken.engine.lookup import LookupState

    st = LookupState(min_match=3, max_match=8, capacity_tokens=64, max_draft=3,
                     accept_window=8, min_acceptance=0.0, min_context=0)
    st.sync([5, 6, 7, 30, 31, 5, 6, 7], 8)
    # (6,7,30) repeats at index 1..3 -> the continuation of [...,7]+[30] is 31,5
    assert st.propose_after([30], 2) == [31, 5]
    st.off = True
    assert st.propose_after([30], 2) == []

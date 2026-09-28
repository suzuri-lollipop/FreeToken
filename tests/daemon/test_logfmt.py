"""LineAssembler byte-stream normalization and ANSI stripping (daemon/logfmt.py).

The assembler is the pure core of the Logs view: a wrong \r/\n decision shows up
as phantom blank rows or a tqdm bar frozen on screen, and a broken ANSI regex
leaks escape garbage into display lines. Both are CPU-only and deterministic, so
the expectations here come from the byte-sequence contract in the docstring,
not from how the code branches.
"""

from __future__ import annotations

import pytest

from freetoken.daemon.logfmt import LineAssembler, strip_ansi


def _feed(text: str) -> list[tuple[str, str]]:
    a = LineAssembler()
    out = a.feed(text)
    out.extend(a.flush())
    return out


def test_plain_lines_split_on_newline():
    assert _feed("one\ntwo\nthree") == [
        ("line", "one"),
        ("line", "two"),
        ("line", "three"),
    ]


def test_empty_input_produces_nothing():
    assert _feed("") == []
    assert _feed("\n") == [("line", "")]


def test_crlf_is_a_single_newline():
    assert _feed("first\r\nsecond") == [("line", "first"), ("line", "second")]


def test_windows_cr_crlf_collapses_to_crlf():
    # Child text-mode stdout adds a CR to output already ending in CRLF.
    assert _feed("first\r\r\nsecond") == [("line", "first"), ("line", "second")]


def test_bare_cr_run_then_text_is_a_partial_redraw():
    assert _feed("100%\r200%\n") == [
        ("partial", "100%"),
        ("line", "200%"),
    ]


def test_bare_cr_run_collapses_to_one_redraw():
    assert _feed("progress\r\rdone\n") == [("partial", "progress"), ("line", "done")]


def test_state_survives_chunk_boundaries():
    a = LineAssembler()
    assert a.feed("he") == []
    assert a.feed("llo\nwor") == [("line", "hello")]
    assert a.feed("ld") == []
    assert a.flush() == [("line", "world")]


def test_cr_pending_across_chunks_decides_on_next_char():
    a = LineAssembler()
    assert a.feed("chunk1\r") == []
    # the deciding char arrives in the next chunk
    assert a.feed("chunk2\n") == [("partial", "chunk1"), ("line", "chunk2")]


def test_trailing_bare_cr_flushes_as_partial():
    a = LineAssembler()
    assert a.feed("stalled\r") == []
    assert a.flush() == [("partial", "stalled")]


def test_flush_without_trailing_newline_is_a_line():
    a = LineAssembler()
    assert a.feed("no newline") == []
    assert a.flush() == [("line", "no newline")]


def test_flush_of_empty_assembler_is_empty():
    assert LineAssembler().flush() == []


def test_strip_ansi_removes_csi_sequences():
    assert strip_ansi("\x1b[32mgreen\x1b[0m") == "green"
    assert strip_ansi("plain \x1b[1;32mtext") == "plain text"


def test_strip_ansi_kills_tqdm_progress_escapes():
    assert strip_ansi("\x1b[?25l\x1b[2K 50%\x1b[?25h\x1b[?25l") == " 50%"


def test_strip_ansi_leaves_plain_text_untouched():
    assert strip_ansi("no escapes at all") == "no escapes at all"
    assert strip_ansi("") == ""


def test_strip_ansi_clears_erase_line_and_leaves_text():
    assert strip_ansi("trailing \x1b[K") == "trailing "


def test_strip_ansi_removes_standalone_two_char_escapes():
    # ESC ] starts an OSC title sequence; the escape opener is stripped with it.
    assert strip_ansi("\x1b]0;title\x07done").endswith("done")
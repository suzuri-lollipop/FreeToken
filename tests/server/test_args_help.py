"""`ft serve --help` formats every flag in one pass, so a single help string argparse cannot
interpolate kills the whole usage listing, not just its own line.

argparse runs each help string through ``help % params`` (the ``%(default)s`` machinery), so a
literal percent sign has to be written ``%%``. Nothing else in the suite renders the parser, so a
flag that trips it stays invisible until someone asks the CLI for help.
"""

from __future__ import annotations

import pytest

from freetoken.server.args import parse_args


@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_serve_help_renders_and_exits_zero(flag, capsys):
    with pytest.raises(SystemExit) as excinfo:
        parse_args([flag])

    assert excinfo.value.code == 0
    # The percent in this one's help is what used to raise out of the formatter.
    out = capsys.readouterr().out
    assert "-85% multi-turn" in out

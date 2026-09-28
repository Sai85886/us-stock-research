"""Tests for the unified stock-research CLI."""

from __future__ import annotations

from stock_research.cli import COMMANDS, main


def test_cli_help_exits_zero():
    assert main(["--help"]) == 0


def test_cli_version_exits_zero(capsys):
    assert main(["--version"]) == 0
    assert "stock-research" in capsys.readouterr().out


def test_cli_unknown_command():
    assert main(["nope"]) == 2


def test_cli_dispatches_status(monkeypatch):
    called = {"yes": False}

    def fake_status() -> int:
        called["yes"] = True
        return 0

    monkeypatch.setitem(COMMANDS, "status", ("Show local setup / data status", fake_status))
    assert main(["status"]) == 0
    assert called["yes"] is True


def test_cli_forwards_args(monkeypatch):
    seen: dict[str, list[str]] = {}

    def fake_quote() -> int:
        import sys

        seen["argv"] = list(sys.argv)
        return 0

    monkeypatch.setitem(COMMANDS, "quote", ("quote", fake_quote))
    assert main(["quote", "AAPL", "--fundamentals"]) == 0
    assert seen["argv"][0].endswith("quote")
    assert seen["argv"][1:] == ["AAPL", "--fundamentals"]

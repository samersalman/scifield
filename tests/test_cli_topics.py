"""Smoke test for the `scifield topics` subcommand (V1-S06)."""

from __future__ import annotations

from typer.testing import CliRunner

from scifield.cli import app

runner = CliRunner()


def test_topics_help() -> None:
    result = runner.invoke(app, ["topics", "--help"])
    assert result.exit_code == 0
    assert "--config" in result.stdout
    assert "--skip-sweep" in result.stdout

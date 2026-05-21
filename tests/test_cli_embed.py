"""Smoke tests for the `scifield embed` and `scifield faiss-build` subcommands (V1-S05)."""

from __future__ import annotations

from typer.testing import CliRunner

from scifield.cli import app

runner = CliRunner()


def test_embed_help() -> None:
    result = runner.invoke(app, ["embed", "--help"])
    assert result.exit_code == 0
    assert "--config" in result.stdout
    assert "--limit" in result.stdout


def test_faiss_build_help() -> None:
    result = runner.invoke(app, ["faiss-build", "--help"])
    assert result.exit_code == 0
    assert "--embeddings" in result.stdout
    assert "--out" in result.stdout

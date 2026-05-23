"""Tests for V1-S07 Claude Code subprocess extractor.

Covers the contract :func:`scifield.epistemic.pilot.run_pilot` relies on:

* Happy-path parse + validate yields a fully-populated
  :class:`EpistemicExtraction`.
* Markdown-fenced JSON is salvageable.
* Single retry fires on both JSON parse failure and Pydantic validation
  failure, and the retry path is gated by
  :attr:`ExtractConfig.retry_on_parse_failure`.
* Persistent failure raises :class:`ExtractionError` with enough context
  for the failures-parquet writer.
* :class:`ExtractConfig` fields (``prompt_version``, ``claude_cmd``)
  thread through to both the persisted row and the subprocess argv.
* :attr:`EpistemicExtraction.raw_response` is preserved verbatim — no
  strip, no unfence — so forensic replay can rebuild what the model
  actually emitted.

All tests mock :func:`subprocess.run`; the real ``claude`` binary is
never invoked.
"""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from scifield.epistemic.extract import (
    ExtractConfig,
    ExtractionError,
    extract_one,
)
from scifield.epistemic.schema import EpistemicExtraction, EpistemicLabel

_VALID_LABEL_DICT: dict = {
    "study_design": "RCT",
    "sample_size": 480,
    "has_control": True,
    "effect_direction": "positive",
    "statistical_claim_present": True,
    "coi_disclosed_in_abstract": False,
}
_VALID_LABEL_JSON: str = json.dumps(_VALID_LABEL_DICT)


def _make_run_result(stdout: str, returncode: int = 0) -> MagicMock:
    """Build a :class:`subprocess.CompletedProcess`-shaped Mock.

    Only the attributes the extractor actually reads are populated:
    ``stdout``, ``stderr``, ``returncode``.
    """
    proc = MagicMock(spec=subprocess.CompletedProcess)
    proc.stdout = stdout
    proc.stderr = ""
    proc.returncode = returncode
    return proc


def test_extract_one_happy_path() -> None:
    """Valid JSON on first try -> fully populated EpistemicExtraction."""
    with patch(
        "scifield.epistemic.extract.subprocess.run",
        return_value=_make_run_result(_VALID_LABEL_JSON),
    ) as mock_run:
        result = extract_one(abstract="some abstract text", pmid=12345)
    assert mock_run.call_count == 1
    assert isinstance(result, EpistemicExtraction)
    assert result.pmid == 12345
    assert result.prompt_version == "v0.1"
    assert result.model_id == "claude-via-claude-code"
    assert isinstance(result.label, EpistemicLabel)
    assert result.label.study_design == "RCT"
    assert result.label.sample_size == 480


def test_extract_one_strips_fenced_json() -> None:
    """```json ...``` fence is tolerated by the unfence helper."""
    fenced = f"```json\n{_VALID_LABEL_JSON}\n```"
    with patch(
        "scifield.epistemic.extract.subprocess.run",
        return_value=_make_run_result(fenced),
    ):
        result = extract_one(abstract="abc", pmid=1)
    assert result.label.study_design == "RCT"
    # raw_response is the verbatim fenced string, not the cleaned form.
    assert result.raw_response == fenced


def test_extract_one_retry_on_parse_failure() -> None:
    """First call returns garbage; retry returns valid JSON; we succeed."""
    fail_proc = _make_run_result("I cannot do this")
    ok_proc = _make_run_result(_VALID_LABEL_JSON)
    with patch(
        "scifield.epistemic.extract.subprocess.run",
        side_effect=[fail_proc, ok_proc],
    ) as mock_run:
        result = extract_one(abstract="abc", pmid=7)
    assert mock_run.call_count == 2
    assert result.label.study_design == "RCT"
    # raw_response reflects the *successful* (retry) attempt's stdout.
    assert result.raw_response == _VALID_LABEL_JSON


def test_extract_one_retry_disabled_raises() -> None:
    """With retry disabled, the first parse failure raises immediately."""
    fail_proc = _make_run_result("not json at all")
    cfg = ExtractConfig(retry_on_parse_failure=False)
    with (
        patch(
            "scifield.epistemic.extract.subprocess.run",
            return_value=fail_proc,
        ) as mock_run,
        pytest.raises(ExtractionError) as exc_info,
    ):
        extract_one(abstract="abc", pmid=9, cfg=cfg)
    assert mock_run.call_count == 1
    assert exc_info.value.pmid == 9
    assert exc_info.value.raw_response == "not json at all"


def test_extract_one_persistent_failure_raises() -> None:
    """Both attempts return un-parseable garbage -> ExtractionError."""
    bad1 = _make_run_result("still not json")
    bad2 = _make_run_result("also not json")
    with (
        patch(
            "scifield.epistemic.extract.subprocess.run",
            side_effect=[bad1, bad2],
        ) as mock_run,
        pytest.raises(ExtractionError) as exc_info,
    ):
        extract_one(abstract="abc", pmid=11)
    assert mock_run.call_count == 2
    err = exc_info.value
    assert err.pmid == 11
    assert "json" in err.reason.lower()
    assert err.raw_response is not None
    # The raw_response captured on the exception is the last (retry) stdout.
    assert err.raw_response == "also not json"


def test_extract_one_validation_error_triggers_retry() -> None:
    """Bad enum on first try -> Pydantic ValidationError -> retry fires."""
    bad_label = dict(_VALID_LABEL_DICT)
    bad_label["study_design"] = "experimental"  # not in closed enum
    bad_proc = _make_run_result(json.dumps(bad_label))
    ok_proc = _make_run_result(_VALID_LABEL_JSON)
    with patch(
        "scifield.epistemic.extract.subprocess.run",
        side_effect=[bad_proc, ok_proc],
    ) as mock_run:
        result = extract_one(abstract="abc", pmid=21)
    assert mock_run.call_count == 2
    assert result.label.study_design == "RCT"


def test_extract_one_uses_prompt_version_from_config() -> None:
    """Custom prompt_version threads through to the persisted row."""
    cfg = ExtractConfig(prompt_version="v0.1.99")
    with patch(
        "scifield.epistemic.extract.subprocess.run",
        return_value=_make_run_result(_VALID_LABEL_JSON),
    ):
        result = extract_one(abstract="abc", pmid=33, cfg=cfg)
    assert result.prompt_version == "v0.1.99"


def test_extract_one_uses_claude_cmd_from_config() -> None:
    """Custom argv is passed to subprocess.run as the first positional arg."""
    custom_cmd = ("/path/to/claude", "--print", "--model", "sonnet")
    cfg = ExtractConfig(claude_cmd=custom_cmd)
    with patch(
        "scifield.epistemic.extract.subprocess.run",
        return_value=_make_run_result(_VALID_LABEL_JSON),
    ) as mock_run:
        extract_one(abstract="abc", pmid=42, cfg=cfg)
    # subprocess.run was called with list(custom_cmd) as args[0].
    call_args, call_kwargs = mock_run.call_args
    assert call_args[0] == list(custom_cmd)
    # And the prompt was passed via stdin.
    assert "input" in call_kwargs
    assert isinstance(call_kwargs["input"], str)
    assert call_kwargs["input"].endswith("JSON:")


def test_raw_response_preserved_on_success() -> None:
    """raw_response equals the un-stripped, un-unfenced stdout verbatim."""
    # Pad with whitespace + wrap in fence to make sure neither is stripped
    # before we hand the string to EpistemicExtraction.
    fenced_padded = f"   ```json\n{_VALID_LABEL_JSON}\n```   \n"
    with patch(
        "scifield.epistemic.extract.subprocess.run",
        return_value=_make_run_result(fenced_padded),
    ):
        result = extract_one(abstract="abc", pmid=55)
    assert result.raw_response == fenced_padded


def test_extraction_error_str_contains_pmid() -> None:
    """The exception's str() representation surfaces the offending pmid."""
    err = ExtractionError(pmid=42, reason="bad", raw_response="x")
    assert "42" in str(err)

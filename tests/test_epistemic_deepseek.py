"""Unit tests for the DeepSeek HTTP transport (V1-S08).

All tests use :class:`httpx.MockTransport` to intercept calls — no real
network, no real API key. Includes a leakage guard that the API key
value never ends up in any serialized sidecar / parquet field.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import httpx
import pytest

from scifield.epistemic.deepseek_extract import (
    DEEPSEEK_FLASH_PRICING,
    build_system_content,
    build_user_content,
    estimate_corpus_cost,
    extract_one_deepseek,
)
from scifield.epistemic.extract import ExtractConfig, ExtractionError

_VALID_LABEL_JSON = json.dumps(
    {
        "study_design": "RCT",
        "sample_size": 200,
        "has_control": True,
        "effect_direction": "positive",
        "statistical_claim_present": True,
        "coi_disclosed_in_abstract": False,
    }
)


def _mock_response(content: str, usage: dict[str, int] | None = None) -> dict[str, Any]:
    """Build a minimal chat-completions response body."""
    return {
        "id": "test-id",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
        "usage": usage
        or {
            "prompt_tokens": 3100,
            "completion_tokens": 80,
            "total_tokens": 3180,
            "prompt_cache_hit_tokens": 3000,
        },
    }


def _make_client(
    handler,
) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.fixture
def cfg_deepseek() -> ExtractConfig:
    return ExtractConfig(
        transport="deepseek",
        deepseek_model="deepseek-v4-flash",
        deepseek_base_url="https://api.deepseek.com",
        deepseek_api_key_env="TEST_DEEPSEEK_KEY",
    )


@pytest.fixture(autouse=True)
def _clear_test_env(monkeypatch):
    """Ensure no real key leaks from the developer's env into tests."""
    monkeypatch.delenv("TEST_DEEPSEEK_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)


def test_happy_path_round_trips_label(monkeypatch, cfg_deepseek):
    """A valid 200 response yields a populated EpistemicExtraction."""
    monkeypatch.setenv("TEST_DEEPSEEK_KEY", "sk-test-fake")
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json=_mock_response(_VALID_LABEL_JSON))

    client = _make_client(handler)
    extraction = extract_one_deepseek(
        abstract="A test abstract.", pmid=12345, cfg=cfg_deepseek, client=client
    )

    assert extraction.pmid == 12345
    assert extraction.label.study_design == "RCT"
    assert extraction.label.sample_size == 200
    assert extraction.model_id == "deepseek-v4-flash"
    # Bearer auth must be sent.
    assert captured["headers"]["authorization"] == "Bearer sk-test-fake"
    # System content carries the few-shots and is byte-identical
    # (cacheable) across calls.
    assert captured["body"]["messages"][0]["role"] == "system"
    assert "study_design" in captured["body"]["messages"][0]["content"]
    # JSON mode + temperature 0 set.
    assert captured["body"]["response_format"] == {"type": "json_object"}
    assert captured["body"]["temperature"] == 0.0


def test_missing_api_key_raises_without_network(cfg_deepseek):
    """Empty/missing key → ExtractionError, no HTTP call."""
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        call_count["n"] += 1
        return httpx.Response(200, json={})

    client = _make_client(handler)
    with pytest.raises(ExtractionError) as excinfo:
        extract_one_deepseek(abstract="abs", pmid=99, cfg=cfg_deepseek, client=client)
    assert "api key not set" in excinfo.value.reason
    assert call_count["n"] == 0


def test_http_5xx_raises_extraction_error(monkeypatch, cfg_deepseek):
    monkeypatch.setenv("TEST_DEEPSEEK_KEY", "sk-test-fake")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server error")

    client = _make_client(handler)
    with pytest.raises(ExtractionError) as excinfo:
        extract_one_deepseek(abstract="abs", pmid=99, cfg=cfg_deepseek, client=client)
    assert "http error" in excinfo.value.reason.lower()


def test_malformed_json_then_valid_retry(monkeypatch, cfg_deepseek):
    """First response is garbage; retry returns valid JSON."""
    monkeypatch.setenv("TEST_DEEPSEEK_KEY", "sk-test-fake")
    call_idx = {"i": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_idx["i"] += 1
        if call_idx["i"] == 1:
            return httpx.Response(200, json=_mock_response("not json at all", usage={}))
        body = json.loads(request.content.decode())
        # Retry must append the strict-JSON suffix to the user message.
        assert "IMPORTANT" in body["messages"][1]["content"]
        return httpx.Response(200, json=_mock_response(_VALID_LABEL_JSON))

    client = _make_client(handler)
    extraction = extract_one_deepseek(abstract="abs", pmid=42, cfg=cfg_deepseek, client=client)
    assert call_idx["i"] == 2
    assert extraction.label.study_design == "RCT"


def test_both_attempts_invalid_raises(monkeypatch, cfg_deepseek):
    monkeypatch.setenv("TEST_DEEPSEEK_KEY", "sk-test-fake")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_mock_response("still not json"))

    client = _make_client(handler)
    with pytest.raises(ExtractionError) as excinfo:
        extract_one_deepseek(abstract="abs", pmid=42, cfg=cfg_deepseek, client=client)
    assert "after retry" in excinfo.value.reason


def test_raw_response_envelope_carries_usage(monkeypatch, cfg_deepseek):
    """The parquet `raw_response` field embeds the usage dict for the cost ledger."""
    monkeypatch.setenv("TEST_DEEPSEEK_KEY", "sk-test-fake")
    usage = {
        "prompt_tokens": 3200,
        "completion_tokens": 75,
        "total_tokens": 3275,
        "prompt_cache_hit_tokens": 3000,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_mock_response(_VALID_LABEL_JSON, usage))

    client = _make_client(handler)
    extraction = extract_one_deepseek(abstract="abs", pmid=7, cfg=cfg_deepseek, client=client)
    envelope = json.loads(extraction.raw_response or "{}")
    assert envelope["usage"] == usage
    assert envelope["content"] == _VALID_LABEL_JSON


# ---------------------------------------------------------------------------
# Leakage guard: the API key VALUE must never end up in serialized state.
# ---------------------------------------------------------------------------


def test_api_key_value_never_serialized(monkeypatch, cfg_deepseek):
    """asdict(ExtractConfig) and EpistemicExtraction must not carry the key."""
    monkeypatch.setenv("TEST_DEEPSEEK_KEY", "sk-VERY-SECRET-12345")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_mock_response(_VALID_LABEL_JSON))

    client = _make_client(handler)
    extraction = extract_one_deepseek(abstract="abs", pmid=11, cfg=cfg_deepseek, client=client)

    # 1) The dataclass dict (which `pilot._extract_cfg_to_dict` serializes
    #    into the .run.json sidecar) must not contain the key value.
    cfg_dict = asdict(cfg_deepseek)
    assert "sk-VERY-SECRET-12345" not in json.dumps(cfg_dict)
    # Only the env var NAME should be present.
    assert cfg_dict["deepseek_api_key_env"] == "TEST_DEEPSEEK_KEY"

    # 2) The extraction's raw_response (which lands in the parquet) must
    #    not echo the key.
    assert "sk-VERY-SECRET-12345" not in (extraction.raw_response or "")

    # 3) The model_dump (any future serialization path) must not echo it.
    assert "sk-VERY-SECRET-12345" not in json.dumps(extraction.model_dump())


# ---------------------------------------------------------------------------
# Cost estimator
# ---------------------------------------------------------------------------


def test_estimate_corpus_cost_zero_abstracts():
    est = estimate_corpus_cost(0)
    assert est["n_abstracts"] == 0
    assert est["estimated_usd"] == 0.0


def test_estimate_corpus_cost_full_corpus_is_under_budget():
    """89k abstracts on deepseek-v4-flash should land well under $20."""
    est = estimate_corpus_cost(89_230)
    assert isinstance(est["estimated_usd"], float)
    # Sanity bounds — should be a few dollars, not hundreds.
    assert 1.0 < est["estimated_usd"] < 20.0
    # Cache-hit input tokens must dominate cache-miss input tokens for
    # the prefix path; otherwise the model is misconfigured.
    assert est["prefix_hit_tokens"] > est["prefix_miss_tokens"] * 100


def test_pricing_table_keys_present():
    """Guard against the pricing dict drifting out of shape."""
    for k in ("input_cache_hit_per_mtok", "input_cache_miss_per_mtok", "output_per_mtok"):
        assert k in DEEPSEEK_FLASH_PRICING


# ---------------------------------------------------------------------------
# Prompt builder structure
# ---------------------------------------------------------------------------


def test_system_content_is_identical_across_calls():
    """Cache-friendliness invariant: the system content is pure / static."""
    a = build_system_content()
    b = build_system_content()
    assert a == b
    assert "study_design" in a
    assert "JSON: " in a  # few-shots present


def test_user_content_is_minimal():
    """User content is just abstract + JSON cue (keeps the miss tail short)."""
    u = build_user_content("hello world")
    assert u == "Abstract: hello world\nJSON:"

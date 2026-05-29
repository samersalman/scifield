r"""DeepSeek HTTP transport for V1-S08 epistemic extraction.

Drop-in alternative to :mod:`scifield.epistemic.extract`'s
``claude --print`` subprocess path. Selected via ``ExtractConfig.transport
= "deepseek"``; the existing ``"claude-code"`` value preserves the
original subprocess behavior unchanged.

Why a separate module
---------------------

The Claude Code subprocess path and the DeepSeek HTTP path share
nothing operationally — different failure modes (rate limits vs HTTP
errors), different cost model (zero marginal vs metered), different
provenance string (model id resolved at call time vs explicit API
model id). Splitting the modules keeps each path readable and lets the
test suite exercise the HTTP path with a mocked transport without
touching the subprocess path's mock surface.

Key-handling discipline (mandatory, do not relax)
-------------------------------------------------

The API key is read from ``os.environ[cfg.deepseek_api_key_env]`` at
**call time only** and is never:

* stored on :class:`ExtractConfig`,
* written to the ``.run.json`` sidecar (the sidecar serializes
  ``ExtractConfig`` via :func:`scifield.epistemic.pilot._extract_cfg_to_dict`,
  which only sees the env var *name*, not the value),
* echoed in the parquet ``raw_response`` column,
* logged on any error path.

If the env var is missing, :func:`extract_one_deepseek` raises
:class:`scifield.epistemic.extract.ExtractionError` *before* touching the
network, so an unconfigured key cannot produce a stray HTTP request.

Cost model & prompt caching
---------------------------

Per the DeepSeek pricing docs (April 2026), ``deepseek-v4-flash`` is
billed at $0.14 / 1M input tokens on cache miss and $0.0028 / 1M on
cache hit — a 50× reduction. DeepSeek's prompt cache is byte-prefix
based and automatic: identical prefixes across calls hit the cache.

This module places the **fixed system prompt and few-shot block** in
the ``system`` message and the **abstract-specific text** in the
``user`` message. With ~89k abstracts, only the first call pays the
miss rate on the ~3000-token prefix; all subsequent calls hit. The
per-call variable cost is dominated by the ~150-token abstract (miss)
plus the ~80-token JSON output. See :func:`estimate_corpus_cost` for
the projected total.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
from pydantic import ValidationError

from scifield.epistemic.extract import ExtractConfig, ExtractionError
from scifield.epistemic.prompt import FEW_SHOT_EXAMPLES, SYSTEM_PROMPT_V0_1
from scifield.epistemic.schema import EpistemicExtraction, EpistemicLabel

__all__ = [
    "DEEPSEEK_FLASH_PRICING",
    "build_system_content",
    "build_user_content",
    "estimate_corpus_cost",
    "extract_one_deepseek",
]


DEEPSEEK_FLASH_PRICING: dict[str, float] = {
    "input_cache_hit_per_mtok": 0.0028,
    "input_cache_miss_per_mtok": 0.14,
    "output_per_mtok": 0.28,
}
"""Per-1M-token USD pricing for ``deepseek-v4-flash`` (April 2026).

Source: https://api-docs.deepseek.com/quick_start/pricing. Update
both this constant and the projection in
:func:`estimate_corpus_cost` if DeepSeek revises pricing."""


_STRICT_JSON_SUFFIX: str = (
    "\n\nIMPORTANT: respond with valid JSON only — no prose, no markdown fences."
)


def build_system_content() -> str:
    """Compose the cacheable system message: instructions + few-shots.

    Everything in this string is byte-identical across every call,
    which is exactly the prefix DeepSeek's cache keys on. Putting the
    six exemplars in the system message rather than the user message
    is the difference between paying the cache-hit rate on the bulk
    of input tokens vs the cache-miss rate.

    Returns:
        The full system-message string the API call should use.
    """
    parts: list[str] = [SYSTEM_PROMPT_V0_1, ""]
    for ex in FEW_SHOT_EXAMPLES:
        parts.append(f"Abstract: {ex['abstract']}")
        parts.append(f"JSON: {json.dumps(ex['label'])}")
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def build_user_content(abstract: str) -> str:
    """Compose the per-call user message containing the target abstract.

    Kept minimal — only the abstract and the trailing ``JSON:`` cue —
    so the cache-miss tail is as short as possible.
    """
    return f"Abstract: {abstract}\nJSON:"


def _post_chat(
    base_url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout_s: float,
    client: httpx.Client | None = None,
) -> httpx.Response:
    """POST one chat-completions request, with an optional injected client.

    ``client`` exists for tests: pass an :class:`httpx.Client` built on a
    :class:`respx.MockTransport` or :class:`httpx.MockTransport` to
    intercept the call without monkey-patching. Production callers pass
    ``None`` and get a fresh :func:`httpx.post` per call.
    """
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if client is not None:
        return client.post(url, json=payload, headers=headers, timeout=timeout_s)
    return httpx.post(url, json=payload, headers=headers, timeout=timeout_s)


def _extract_content_and_usage(body: dict[str, Any]) -> tuple[str, dict[str, int]]:
    """Pull the assistant text and usage dict out of a chat-completions response.

    Raises a plain :class:`KeyError` / :class:`IndexError` on shape
    failure so the caller can re-raise as a typed
    :class:`ExtractionError` with the offending body included.
    """
    content = body["choices"][0]["message"]["content"]
    usage = body.get("usage", {})
    # Coerce usage values to int and pick only the keys we record. The
    # full DeepSeek usage dict includes some non-standard fields; we
    # keep just the four we'll log to the sidecar.
    usage_clean = {
        k: int(usage[k])
        for k in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "prompt_cache_hit_tokens",
        )
        if k in usage
    }
    return str(content), usage_clean


def extract_one_deepseek(
    abstract: str,
    pmid: int,
    cfg: ExtractConfig,
    *,
    client: httpx.Client | None = None,
) -> EpistemicExtraction:
    """Run one DeepSeek extraction; mirror ``extract_one``'s contract.

    Behavior:

    * Read the API key from ``os.environ[cfg.deepseek_api_key_env]``;
      raise :class:`ExtractionError` (no network call) if absent.
    * Build a two-message conversation: system (instructions +
      few-shots, cache-friendly) + user (abstract).
    * POST to ``{cfg.deepseek_base_url}/chat/completions`` with
      ``response_format={"type": "json_object"}`` and ``temperature=0``.
    * Parse the assistant content as JSON, validate against
      :class:`EpistemicLabel`.
    * On parse or validation failure: optionally retry once with the
      strict-JSON suffix appended to the user message (controlled by
      :attr:`ExtractConfig.retry_on_parse_failure`).
    * On persistent failure: raise :class:`ExtractionError` with the
      reason and the raw assistant content (NOT the key).

    The returned :class:`EpistemicExtraction`'s ``model_id`` is set to
    ``cfg.deepseek_model`` (e.g. ``"deepseek-v4-flash"``), distinguishing
    it from the Claude Code path's ``"claude-via-claude-code"`` so the
    parquet can be partitioned by transport later.

    The ``raw_response`` field carries a JSON-encoded envelope
    ``{"content": <assistant text>, "usage": <usage dict>}`` so the
    parquet preserves per-call token counts for the cost ledger.

    Args:
        abstract: PubMed abstract text.
        pmid: PubMed ID, stamped onto the result and any
            :class:`ExtractionError`.
        cfg: :class:`ExtractConfig`; only the ``deepseek_*``,
            ``timeout_s``, ``prompt_version``, and
            ``retry_on_parse_failure`` fields are consulted.
        client: Optional injected :class:`httpx.Client` for tests.

    Returns:
        A validated :class:`EpistemicExtraction`.

    Raises:
        ExtractionError: On missing key, HTTP failure, malformed
            response, or persistent parse / validation failure.
    """
    api_key = os.environ.get(cfg.deepseek_api_key_env, "")
    if not api_key:
        raise ExtractionError(
            pmid=pmid,
            reason=(
                f"deepseek api key not set: env var "
                f"{cfg.deepseek_api_key_env} is missing or empty"
            ),
            raw_response=None,
        )

    system_content = build_system_content()
    user_content = build_user_content(abstract)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    payload: dict[str, Any] = {
        "model": cfg.deepseek_model,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "temperature": 0.0,
        "max_tokens": 512,
        # deepseek-v4-flash defaults to thinking mode, which consumes
        # reasoning tokens against ``max_tokens`` before emitting the
        # JSON. For fixed-schema labeling we don't need any chain of
        # thought, so disable thinking explicitly. Without this, ~70%
        # of calls returned empty or mid-JSON-truncated output in the
        # 100-abstract smoke run.
        "thinking": {"type": "disabled"},
    }

    try:
        resp = _post_chat(
            base_url=cfg.deepseek_base_url,
            api_key=api_key,
            payload=payload,
            timeout_s=cfg.timeout_s,
            client=client,
        )
        resp.raise_for_status()
        body = resp.json()
    except httpx.HTTPError as err:
        raise ExtractionError(
            pmid=pmid,
            reason=f"http error: {err.__class__.__name__}: {err}",
            raw_response=None,
        ) from err

    try:
        raw_content, usage = _extract_content_and_usage(body)
    except (KeyError, IndexError, TypeError) as err:
        raise ExtractionError(
            pmid=pmid,
            reason=f"malformed response shape: {err}",
            raw_response=json.dumps(body)[:2000],
        ) from err

    try:
        parsed = json.loads(raw_content)
        label = EpistemicLabel.model_validate(parsed)
        return EpistemicExtraction(
            pmid=pmid,
            label=label,
            model_id=cfg.deepseek_model,
            prompt_version=cfg.prompt_version,
            raw_response=json.dumps({"content": raw_content, "usage": usage}),
        )
    except (json.JSONDecodeError, ValidationError) as first_err:
        if not cfg.retry_on_parse_failure:
            raise ExtractionError(
                pmid=pmid,
                reason=f"json/validation failure (no retry): {first_err}",
                raw_response=raw_content,
            ) from first_err

    # ---- retry path ----
    retry_messages = [
        messages[0],
        {"role": "user", "content": user_content + _STRICT_JSON_SUFFIX},
    ]
    retry_payload = {**payload, "messages": retry_messages}
    try:
        resp_retry = _post_chat(
            base_url=cfg.deepseek_base_url,
            api_key=api_key,
            payload=retry_payload,
            timeout_s=cfg.timeout_s,
            client=client,
        )
        resp_retry.raise_for_status()
        body_retry = resp_retry.json()
        raw_retry, usage_retry = _extract_content_and_usage(body_retry)
        parsed_retry = json.loads(raw_retry)
        label_retry = EpistemicLabel.model_validate(parsed_retry)
    except (
        httpx.HTTPError,
        KeyError,
        IndexError,
        TypeError,
        json.JSONDecodeError,
        ValidationError,
    ) as retry_err:
        raise ExtractionError(
            pmid=pmid,
            reason=f"json/validation failure after retry: {retry_err}",
            raw_response=raw_content,
        ) from retry_err

    return EpistemicExtraction(
        pmid=pmid,
        label=label_retry,
        model_id=cfg.deepseek_model,
        prompt_version=cfg.prompt_version,
        raw_response=json.dumps({"content": raw_retry, "usage": usage_retry}),
    )


def estimate_corpus_cost(
    n_abstracts: int,
    avg_abstract_chars: int = 1500,
    prefix_tokens: int = 3000,
    output_tokens_per_call: int = 80,
    pricing: dict[str, float] | None = None,
) -> dict[str, float | int]:
    """Project total USD cost for an N-abstract DeepSeek run.

    Cost model:

    * The system prompt + few-shot prefix (≈ ``prefix_tokens``) misses
      the cache on call 1 and hits on every subsequent call.
    * The per-call abstract text (≈ ``avg_abstract_chars / 4`` tokens)
      always cache-misses (it varies per call).
    * The JSON output is ≈ ``output_tokens_per_call`` per call.

    Token approximation: 4 chars ≈ 1 token (English heuristic).

    Args:
        n_abstracts: Number of abstracts to extract.
        avg_abstract_chars: Mean character length of the abstracts in
            the run. PubMed abstracts average ~1500 chars.
        prefix_tokens: Size of the cached prefix (system + few-shots).
            Default 3000 based on a one-shot tokenization of the
            current ``v0.1`` prompt + the six exemplars.
        output_tokens_per_call: Expected JSON output size per call.
        pricing: Override the pricing dict (defaults to
            :data:`DEEPSEEK_FLASH_PRICING`).

    Returns:
        Breakdown dict with token counts and ``estimated_usd``.
    """
    p = pricing if pricing is not None else DEEPSEEK_FLASH_PRICING
    if n_abstracts <= 0:
        return {
            "n_abstracts": 0,
            "prefix_miss_tokens": 0,
            "prefix_hit_tokens": 0,
            "abstract_miss_tokens": 0,
            "output_tokens": 0,
            "estimated_usd": 0.0,
        }
    abstract_tokens_per_call = max(1, avg_abstract_chars // 4)
    prefix_miss = prefix_tokens  # call 1 misses
    prefix_hit = prefix_tokens * (n_abstracts - 1)
    abstract_miss = n_abstracts * abstract_tokens_per_call
    output_total = n_abstracts * output_tokens_per_call
    cost = (
        prefix_hit * p["input_cache_hit_per_mtok"] / 1_000_000
        + prefix_miss * p["input_cache_miss_per_mtok"] / 1_000_000
        + abstract_miss * p["input_cache_miss_per_mtok"] / 1_000_000
        + output_total * p["output_per_mtok"] / 1_000_000
    )
    return {
        "n_abstracts": n_abstracts,
        "prefix_miss_tokens": prefix_miss,
        "prefix_hit_tokens": prefix_hit,
        "abstract_miss_tokens": abstract_miss,
        "output_tokens": output_total,
        "estimated_usd": round(cost, 4),
    }

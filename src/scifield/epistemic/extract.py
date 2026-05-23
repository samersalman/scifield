r"""V1-S07 Claude Code subprocess extractor for epistemic-quality labels.

This module is the model-facing edge of the V1-S07 pilot. Per plan §D, we
deliberately commit to invoking the local ``claude --print`` CLI binary
via :func:`subprocess.run` rather than the Anthropic HTTP API. Rationale,
preserved here so a later reader does not "fix" it back to HTTP:

* Zero API spend during V1-S07 — only the local Claude Code subscription
  is hit.
* The exact model resolved by the CLI at call time is captured verbatim
  in :attr:`EpistemicExtraction.raw_response` for forensic replay.
* :class:`ExtractConfig` parameterizes everything (binary path, model id
  label, prompt version, timeout, retry behavior) so V1-S08 can promote
  this to the Batch API by either swapping ``cfg.claude_cmd`` for an
  equivalent shell shim or substituting a new implementation behind the
  same :func:`extract_one` signature.

The :func:`extract_one` function is the single entrypoint:

1. Compose the prompt via :func:`scifield.epistemic.prompt.build_prompt`.
2. Run the configured subprocess with the prompt on stdin.
3. Strip whitespace and any ``\`\`\`json ... \`\`\``` markdown fences off
   the stdout.
4. Parse JSON, validate against :class:`EpistemicLabel`.
5. On parse OR validation failure, optionally retry once with a stricter
   "JSON only" suffix appended to the prompt.
6. On persistent failure, raise :class:`ExtractionError` so
   :func:`scifield.epistemic.pilot.run_pilot` can route the row into the
   failures parquet rather than poisoning the success parquet.

The :attr:`EpistemicExtraction.raw_response` field always holds the
**original** stdout from the first successful response (un-stripped,
un-unfenced), so downstream forensics can replay what the model actually
sent on the wire — not what we cleaned up before parsing.

.. note::

   Per the V1-S07 plan risk note: this code MUST NOT be invoked on real
   corpus data outside the 50-abstract pilot in
   :mod:`scifield.epistemic.pilot`. V1-S08 (the full 200k batch) requires
   OSF pre-registration #1 to be filed first and is explicitly gated on
   that link landing in ``docs/preregistrations/PR1_epistemic_extraction.md``.

All :func:`subprocess.run` invocations live inside the function body —
nothing at module import or class-construction time touches the
``claude`` binary, so tests can safely mock ``subprocess.run`` via
:func:`unittest.mock.patch`.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import cast

from pydantic import ValidationError

from scifield.epistemic.prompt import PROMPT_VERSION, build_prompt
from scifield.epistemic.schema import EpistemicExtraction, EpistemicLabel

__all__ = [
    "ExtractConfig",
    "ExtractionError",
    "extract_one",
]


_STRICT_JSON_SUFFIX: str = (
    "\n\nIMPORTANT: respond with valid JSON only — no prose, no markdown fences."
)
"""Appended to the prompt on the retry attempt. Kept as a module-level
constant so the test suite can reference the exact text if it ever needs
to assert the retry prompt shape."""


@dataclass(frozen=True)
class ExtractConfig:
    """Configuration for a single :func:`extract_one` call.

    All fields have sensible defaults so callers can pass ``None`` and
    get the V1-S07 pilot behavior. V1-S08 will likely override at least
    :attr:`claude_cmd` and :attr:`model_id` once it promotes the pipeline.

    Attributes:
        claude_cmd: ``argv`` tuple handed to :func:`subprocess.run`. The
            default ``("claude", "--print")`` matches
            ``conf/epistemic/v1.yaml`` ``pilot.claude_cmd``.
        model_id: Opaque identifier stamped onto every
            :class:`EpistemicExtraction.model_id`. This is a *label*, not
            the real underlying model id resolved by the CLI; the real
            model fingerprint (if any) lives in :attr:`raw_response`.
        prompt_version: Stamped onto every
            :class:`EpistemicExtraction.prompt_version`. Defaults to the
            module-level :data:`scifield.epistemic.prompt.PROMPT_VERSION`
            so bumping the prompt version in one place propagates here.
        timeout_s: Subprocess wall-clock timeout in seconds. Default 120s
            comfortably exceeds typical Claude Code latency on a single
            abstract.
        retry_on_parse_failure: If True (default), a single retry is
            attempted on JSON parse failure OR Pydantic validation
            failure, with a stricter "JSON only" suffix appended to the
            prompt. If False, the first failure raises immediately.
    """

    claude_cmd: tuple[str, ...] = ("claude", "--print")
    model_id: str = "claude-via-claude-code"
    prompt_version: str = PROMPT_VERSION
    timeout_s: float = 120.0
    retry_on_parse_failure: bool = True


class ExtractionError(Exception):
    """Raised by :func:`extract_one` when extraction cannot be salvaged.

    Carries enough context for :func:`scifield.epistemic.pilot.run_pilot`
    to write a row to the failures parquet without needing to crack open
    the exception type — :attr:`pmid`, :attr:`reason`, and (when we got
    any bytes back from the subprocess at all) :attr:`raw_response`.
    """

    def __init__(self, pmid: int, reason: str, raw_response: str | None) -> None:
        self.pmid = pmid
        self.reason = reason
        self.raw_response = raw_response
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        snippet = ""
        if self.raw_response is not None:
            preview = self.raw_response.strip().replace("\n", " ")
            if len(preview) > 200:
                preview = preview[:200] + "..."
            snippet = f" raw_response={preview!r}"
        return f"ExtractionError(pmid={self.pmid}): {self.reason}.{snippet}"

    def __str__(self) -> str:
        return self._format_message()


def _unfence(text: str) -> str:
    r"""Strip a single surrounding markdown code fence, if present.

    Handles the three common shapes the model emits when it ignores the
    "no fences" instruction:

    * ``\`\`\`json\\n{...}\\n\`\`\``
    * ``\`\`\`\\n{...}\\n\`\`\``
    * ``\`\`\`{...}\`\`\``` (single-line, no language tag, no newline)

    If no fence is detected, returns ``text`` unchanged. This is a
    forgiving helper, not a strict parser — we still rely on
    :func:`json.loads` to catch genuinely malformed payloads.

    Args:
        text: Already-``strip()``-ed stdout from the subprocess.

    Returns:
        The contents between the fences (themselves ``strip()``-ed), or
        the input untouched if no fence pattern matched.
    """
    if not text.startswith("```"):
        return text
    # Drop the leading fence (and optional language tag on the same line).
    body = text[3:]
    newline = body.find("\n")
    if newline != -1:
        # Inspect the substring between the opening ``` and the first
        # newline. If it's a short language tag (alphanumerics only) we
        # consume it; otherwise we keep it as part of the body.
        lang = body[:newline].strip()
        if lang.isalnum() or lang == "":
            body = body[newline + 1 :]
    # Strip a trailing fence if present.
    if body.endswith("```"):
        body = body[:-3]
    return body.strip()


def _run_subprocess(prompt: str, cfg: ExtractConfig) -> subprocess.CompletedProcess[str]:
    """Single :func:`subprocess.run` invocation, isolated for testability.

    Kept as a thin wrapper so the test suite can patch one symbol
    (``scifield.epistemic.extract.subprocess.run``) and exercise both
    the happy and retry paths without ever touching the real binary.
    """
    return subprocess.run(
        list(cfg.claude_cmd),
        input=prompt,
        capture_output=True,
        text=True,
        timeout=cfg.timeout_s,
        check=False,
    )


def _try_parse(stdout: str) -> EpistemicLabel:
    """Parse + validate one stdout response, or raise.

    Raises :exc:`json.JSONDecodeError` (from :func:`json.loads`) or
    :exc:`pydantic.ValidationError` (from
    :meth:`EpistemicLabel.model_validate`) on failure. Callers
    distinguish those two error types in their retry-decision logic.
    """
    cleaned = _unfence(stdout.strip())
    parsed = json.loads(cleaned)
    return cast(EpistemicLabel, EpistemicLabel.model_validate(parsed))


def extract_one(
    abstract: str,
    pmid: int,
    cfg: ExtractConfig | None = None,
) -> EpistemicExtraction:
    """Extract one :class:`EpistemicExtraction` from one abstract.

    The full lifecycle is:

    1. Resolve config (default :class:`ExtractConfig` if ``cfg is None``).
    2. Build the prompt via :func:`build_prompt`.
    3. Run the subprocess; capture stdout.
    4. Attempt JSON parse + Pydantic validation on the (un-fenced) stdout.
    5. On any failure with :attr:`ExtractConfig.retry_on_parse_failure`
       enabled, re-run the subprocess once with
       :data:`_STRICT_JSON_SUFFIX` appended to the original prompt and
       attempt parse + validation again.
    6. On persistent failure, raise :class:`ExtractionError` with the
       last stdout seen (so the pilot driver can write a failures-parquet
       row without losing the raw response).

    On success, :attr:`EpistemicExtraction.raw_response` is set to the
    **first** successful subprocess's stdout *verbatim* — no strip, no
    unfence. The cleaned, parsed form is reconstructable from
    :attr:`EpistemicExtraction.label`; the raw form is what we keep for
    forensics.

    Args:
        abstract: The PubMed abstract text to extract from.
        pmid: PubMed ID, stamped onto the returned :class:`EpistemicExtraction`.
        cfg: Optional :class:`ExtractConfig`; defaults to ``ExtractConfig()``.

    Returns:
        A fully validated :class:`EpistemicExtraction` carrying the
        label, provenance, and verbatim raw response.

    Raises:
        ExtractionError: When parse + validation fail on the first
            attempt and either retry is disabled or the retry attempt
            also fails. The exception carries ``pmid``, a human-readable
            ``reason``, and (when available) the last ``raw_response``
            received from the subprocess.
    """
    cfg = cfg if cfg is not None else ExtractConfig()
    prompt = build_prompt(abstract)

    # ---- First attempt ----
    proc = _run_subprocess(prompt, cfg)
    first_stdout = proc.stdout if proc.stdout is not None else ""
    try:
        label = _try_parse(first_stdout)
        return EpistemicExtraction(
            pmid=pmid,
            label=label,
            model_id=cfg.model_id,
            prompt_version=cfg.prompt_version,
            raw_response=first_stdout,
        )
    except (json.JSONDecodeError, ValidationError) as first_err:
        if not cfg.retry_on_parse_failure:
            raise ExtractionError(
                pmid=pmid,
                reason=f"json/validation failure on first attempt: {first_err}",
                raw_response=first_stdout,
            ) from first_err

    # ---- Retry attempt ----
    strict_prompt = prompt + _STRICT_JSON_SUFFIX
    retry_proc = _run_subprocess(strict_prompt, cfg)
    retry_stdout = retry_proc.stdout if retry_proc.stdout is not None else ""
    try:
        label = _try_parse(retry_stdout)
        return EpistemicExtraction(
            pmid=pmid,
            label=label,
            model_id=cfg.model_id,
            prompt_version=cfg.prompt_version,
            raw_response=retry_stdout,
        )
    except (json.JSONDecodeError, ValidationError) as retry_err:
        raise ExtractionError(
            pmid=pmid,
            reason=f"json/validation failure after retry: {retry_err}",
            raw_response=retry_stdout,
        ) from retry_err

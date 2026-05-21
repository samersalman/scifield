"""Top-level enrichment orchestrator (V1-S04).

Sequences the four enrichment sources — OpenAlex, Authors, ROR, Semantic
Scholar — into a single ``enrich_corpus()`` async entrypoint. Each source is
idempotent: re-running is safe and manifest-aware (OpenAlex), cache-backed
(ROR), or schema-empty when unconfigured (Semantic Scholar).

The orchestrator owns no httpx state; each source module spins up its own
``httpx.AsyncClient``. It owns:
- per-source ``RateLimiter`` instances (different rate budgets per API)
- the ``authorships_staging`` sidecar that bridges OpenAlex → Authors/ROR for
  ``--only authors`` / ``--only ror`` re-runs (so the OpenAlex fetch is not
  repeated)
- ``record_run()`` provenance sidecars next to every output Parquet
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scifield.corpus.authors import AuthorsConfig, disambiguate_authorships
from scifield.corpus.enrich_store import write_enrichment_parquet
from scifield.corpus.openalex import OpenAlexConfig, enrich_openalex
from scifield.corpus.pubmed import RateLimiter
from scifield.corpus.ror import RORConfig, RORMatcher, build_institution_tables
from scifield.corpus.semantic_scholar import SemanticScholarConfig, enrich_semantic_scholar
from scifield.repro import record_run

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

ENRICH_SOURCES: tuple[str, ...] = ("openalex", "authors", "ror", "semantic_scholar")

_STAGING_FILENAME = "staging.json"


@dataclass(slots=True)
class EnrichmentPaths:
    """Filesystem layout for an enrichment run."""

    enrichment_dir: Path
    cache_dir: Path
    manifest_dir: Path
    log_dir: Path


@dataclass(slots=True)
class EnrichmentReport:
    """Summary counters returned from :func:`enrich_corpus`."""

    n_pmids_input: int = 0
    n_openalex_ok: int = 0
    n_openalex_failed: int = 0
    n_authorships: int = 0
    n_institutions: int = 0
    n_paper_institutions: int = 0
    n_references: int = 0
    n_ss_papers: int = 0
    ss_skipped: bool = False
    elapsed_s: float = 0.0
    sources_run: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# PMID loading
# ---------------------------------------------------------------------------


def load_pmids_from_corpus(
    duckdb_path: Path | str,
    limit: int | None = None,
) -> list[str]:
    """Read distinct PMIDs from the V1-S03 ``papers`` view.

    Returns the PMIDs sorted lexically (deterministic). ``limit`` is applied
    AFTER the sort so a ``--limit 200`` smoke is the same 200 every run.
    """
    import duckdb

    conn = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        rows = conn.execute(
            "SELECT DISTINCT pmid FROM papers WHERE pmid IS NOT NULL AND pmid <> ''"
        ).fetchall()
    finally:
        conn.close()
    pmids = sorted(str(r[0]) for r in rows if r[0])
    if limit is not None:
        pmids = pmids[:limit]
    return pmids


# ---------------------------------------------------------------------------
# Staging sidecar (OpenAlex authorships staging) — JSON list-of-dicts
# ---------------------------------------------------------------------------


def _staging_path(paths: EnrichmentPaths) -> Path:
    return paths.cache_dir / "openalex" / _STAGING_FILENAME


def _write_staging(paths: EnrichmentPaths, staging_rows: list[dict[str, Any]]) -> None:
    path = _staging_path(paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(staging_rows, ensure_ascii=False))
    tmp.replace(path)


def _read_staging(paths: EnrichmentPaths) -> list[dict[str, Any]] | None:
    path = _staging_path(paths)
    if not path.exists():
        return None
    raw = path.read_text()
    if not raw.strip():
        return []
    data = json.loads(raw)
    if not isinstance(data, list):
        return None
    return data


# ---------------------------------------------------------------------------
# Sidecar helper
# ---------------------------------------------------------------------------


def _record(parquet_path: Path, source: str, config_hash_input: dict[str, Any] | None) -> None:
    record_run(
        artifact_path=parquet_path,
        inputs={},
        config={
            "source": source,
            "config_hash_input": config_hash_input or {},
        },
    )


# ---------------------------------------------------------------------------
# Per-source runners
# ---------------------------------------------------------------------------


async def _run_openalex(
    *,
    pmids: list[str],
    paths: EnrichmentPaths,
    cfg: OpenAlexConfig,
    config_hash_input: dict[str, Any] | None,
    report: EnrichmentReport,
) -> list[dict[str, Any]]:
    rate_limiter = RateLimiter(cfg.rate_limit)
    result = await enrich_openalex(pmids=pmids, cfg=cfg, rate_limiter=rate_limiter)

    works_rows = result["works"]
    references_rows = result["references"]
    failed_rows = result["failed"]
    staging_rows = result["authorships_staging"]

    works_path = write_enrichment_parquet(works_rows, "openalex_works", paths.enrichment_dir)
    refs_path = write_enrichment_parquet(references_rows, "references_out", paths.enrichment_dir)
    failed_path = write_enrichment_parquet(failed_rows, "enrichment_failed", paths.enrichment_dir)

    _record(works_path, "openalex", config_hash_input)
    _record(refs_path, "openalex", config_hash_input)
    _record(failed_path, "openalex", config_hash_input)

    _write_staging(paths, staging_rows)

    report.n_openalex_ok = len(works_rows)
    report.n_openalex_failed = len(failed_rows)
    report.n_references = len(references_rows)
    report.sources_run.append("openalex")
    return list(staging_rows)


async def _run_authors(
    *,
    staging_rows: list[dict[str, Any]],
    paths: EnrichmentPaths,
    cfg: AuthorsConfig,
    config_hash_input: dict[str, Any] | None,
    report: EnrichmentReport,
) -> None:
    # CPU-bound + small; run in a worker thread so it doesn't block ROR's I/O.
    rows = await asyncio.to_thread(disambiguate_authorships, staging_rows, cfg)
    out_path = write_enrichment_parquet(rows, "authorships", paths.enrichment_dir)
    _record(out_path, "authors", config_hash_input)
    report.n_authorships = len(rows)
    report.sources_run.append("authors")


async def _run_ror(
    *,
    staging_rows: list[dict[str, Any]],
    paths: EnrichmentPaths,
    cfg: RORConfig,
    config_hash_input: dict[str, Any] | None,
    report: EnrichmentReport,
) -> None:
    rate_limiter = RateLimiter(cfg.rate_limit)
    matcher = RORMatcher(cfg, rate_limiter)
    try:
        institutions_rows, paper_institutions_rows = await build_institution_tables(
            staging_rows=staging_rows,
            matcher=matcher,
        )
        await matcher.flush_cache()
    finally:
        await matcher.aclose()

    inst_path = write_enrichment_parquet(institutions_rows, "institutions", paths.enrichment_dir)
    paper_inst_path = write_enrichment_parquet(
        paper_institutions_rows, "paper_institutions", paths.enrichment_dir
    )
    _record(inst_path, "ror", config_hash_input)
    _record(paper_inst_path, "ror", config_hash_input)
    report.n_institutions = len(institutions_rows)
    report.n_paper_institutions = len(paper_institutions_rows)
    report.sources_run.append("ror")


async def _run_semantic_scholar(
    *,
    pmids: list[str],
    paths: EnrichmentPaths,
    cfg: SemanticScholarConfig,
    config_hash_input: dict[str, Any] | None,
    report: EnrichmentReport,
) -> None:
    rate_limiter = RateLimiter(cfg.rate_limit)
    result = await enrich_semantic_scholar(pmids=pmids, cfg=cfg, rate_limiter=rate_limiter)
    papers = result["papers"]
    intents = result["intents"]
    skipped = bool(result.get("skipped"))

    papers_path = write_enrichment_parquet(papers, "semantic_scholar", paths.enrichment_dir)
    intents_path = write_enrichment_parquet(intents, "citation_intents", paths.enrichment_dir)
    _record(papers_path, "semantic_scholar", config_hash_input)
    _record(intents_path, "semantic_scholar", config_hash_input)

    report.n_ss_papers = len(papers)
    report.ss_skipped = skipped
    if skipped:
        logger.info("semantic_scholar: empty parquet written (skipped — no API key)")
    report.sources_run.append("semantic_scholar")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


async def enrich_corpus(
    *,
    pmids: list[str],
    paths: EnrichmentPaths,
    openalex_cfg: OpenAlexConfig,
    authors_cfg: AuthorsConfig,
    ror_cfg: RORConfig,
    ss_cfg: SemanticScholarConfig,
    only: str | None = None,
    skip: set[str] | None = None,
    config_hash_input: dict[str, Any] | None = None,
) -> EnrichmentReport:
    """Drive the four-source V1-S04 enrichment pipeline.

    Sequence (each source idempotent; re-running is safe):

    1. **OpenAlex** — must run first; produces the ``authorships_staging`` list
       that Authors + ROR consume. Persisted to
       ``paths.cache_dir/openalex/staging.json`` so ``--only authors`` /
       ``--only ror`` re-runs don't re-fetch OpenAlex.
    2. **Authors** — three-layer disambiguation over the staging list.
    3. **ROR** — un-matched-affiliation fill-in over the same staging list.
       Runs *concurrently* with Authors (different output files; no shared
       mutable state).
    4. **SemanticScholar** — independent of OpenAlex; logs a warning and
       writes empty schema-only Parquets when the API key is missing.

    Parameters
    ----------
    pmids:
        Input PMIDs to enrich. ``--only authors`` / ``--only ror`` ignore this
        in favor of the cached OpenAlex staging.
    paths:
        Filesystem layout (enrichment outputs, on-disk caches, manifests, logs).
    openalex_cfg, authors_cfg, ror_cfg, ss_cfg:
        Per-source configs. The orchestrator wires up ``RateLimiter`` instances
        from each ``rate_limit`` field.
    only:
        If set, run ONLY that source. For ``"authors"`` or ``"ror"``, the prior
        OpenAlex staging is loaded from ``paths.cache_dir/openalex/staging.json``;
        if missing, the run logs a warning and skips that source.
    skip:
        Sources to bypass entirely (a no-op for any source already excluded by
        ``only``).
    config_hash_input:
        Passed verbatim into every ``record_run`` sidecar's config dict (under
        ``config_hash_input``) so output provenance is tied to the caller's
        config.

    Returns
    -------
    EnrichmentReport
        Counters + elapsed wall-clock + ordered list of sources that actually
        ran.

    Notes
    -----
    View registration over ``papers.duckdb`` is NOT done here — the CLI layer
    calls :func:`scifield.corpus.enrich_store.register_enrichment_views`
    once all sources have completed.
    """
    skip = skip or set()
    started = time.monotonic()
    report = EnrichmentReport(n_pmids_input=len(pmids))

    def should_run(source: str) -> bool:
        if only is not None and source != only:
            return False
        return source not in skip

    paths.enrichment_dir.mkdir(parents=True, exist_ok=True)
    paths.cache_dir.mkdir(parents=True, exist_ok=True)
    paths.manifest_dir.mkdir(parents=True, exist_ok=True)
    paths.log_dir.mkdir(parents=True, exist_ok=True)

    # ----- OpenAlex (must run first if Authors/ROR are running and we have no
    # staging cached) ------------------------------------------------------
    staging_rows: list[dict[str, Any]] | None = None
    if should_run("openalex"):
        staging_rows = await _run_openalex(
            pmids=pmids,
            paths=paths,
            cfg=openalex_cfg,
            config_hash_input=config_hash_input,
            report=report,
        )

    # ----- Authors + ROR (concurrent — both read staging, write distinct
    # output files, no shared state) ---------------------------------------
    needs_staging_for_downstream = should_run("authors") or should_run("ror")
    if needs_staging_for_downstream and staging_rows is None:
        staging_rows = _read_staging(paths)
        if staging_rows is None:
            logger.warning(
                "authors/ror requested but no OpenAlex staging found at %s — "
                "run --only openalex first or drop --only/--skip filters",
                _staging_path(paths),
            )

    concurrent_tasks: list[asyncio.Task[None]] = []
    if should_run("authors") and staging_rows is not None:
        concurrent_tasks.append(
            asyncio.create_task(
                _run_authors(
                    staging_rows=staging_rows,
                    paths=paths,
                    cfg=authors_cfg,
                    config_hash_input=config_hash_input,
                    report=report,
                )
            )
        )
    if should_run("ror") and staging_rows is not None:
        concurrent_tasks.append(
            asyncio.create_task(
                _run_ror(
                    staging_rows=staging_rows,
                    paths=paths,
                    cfg=ror_cfg,
                    config_hash_input=config_hash_input,
                    report=report,
                )
            )
        )
    if concurrent_tasks:
        await asyncio.gather(*concurrent_tasks)

    # ----- Semantic Scholar (independent) ---------------------------------
    if should_run("semantic_scholar"):
        await _run_semantic_scholar(
            pmids=pmids,
            paths=paths,
            cfg=ss_cfg,
            config_hash_input=config_hash_input,
            report=report,
        )

    report.elapsed_s = time.monotonic() - started
    return report

"""Command-line interface for scifield."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from statistics import mean
from typing import Any, cast

import hydra
import pyarrow as pa
import pyarrow.parquet as pq
import typer
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from scifield.corpus import (
    AuthorsConfig,
    EnrichmentPaths,
    EntrezClient,
    EntrezConfig,
    HarvestConfig,
    HarvestReport,
    JournalSpec,
    OpenAlexConfig,
    OutputConfig,
    RateLimiter,
    RORConfig,
    SemanticScholarConfig,
    build_duckdb,
    enrich_corpus,
    harvest_corpus,
    load_pmids_from_corpus,
    register_enrichment_views,
)
from scifield.corpus.pubmed_demo import fetch_demo_papers
from scifield.repro import record_run
from scifield.thematic import (
    build_faiss_hnsw,
    make_embedder,
    write_index,
    write_pmid_map,
)

app = typer.Typer(
    name="scifield",
    help="SciField — multi-axis framework for monitoring scientific field health.",
    no_args_is_help=True,
)

epistemic_app = typer.Typer(
    name="epistemic",
    help="V1-S07 epistemic-quality extraction (schema, sampling, labeling, pilot).",
    no_args_is_help=True,
)
app.add_typer(epistemic_app, name="epistemic")


def _load_config(name: str) -> DictConfig:
    """Compose a Hydra config from the repo's `conf/` directory."""
    conf_dir = Path(__file__).resolve().parents[2] / "conf"
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(version_base="1.3", config_dir=str(conf_dir)):
        cfg = hydra.compose(config_name=name)
    return cast(DictConfig, cfg)


def _load_topics_config(path: Path | None = None) -> DictConfig:
    """Load conf/thematic/topics.yaml directly via OmegaConf."""
    if path is None:
        path = Path(__file__).resolve().parents[2] / "conf" / "thematic" / "topics.yaml"
    cfg = OmegaConf.load(path)
    return cast(DictConfig, cfg)


def _load_epistemic_config(name: str = "v1") -> DictConfig:
    """Load conf/epistemic/<name>.yaml directly via OmegaConf (flat — no Hydra group nesting)."""
    path = Path(__file__).resolve().parents[2] / "conf" / "epistemic" / f"{name}.yaml"
    cfg = OmegaConf.load(path)
    return cast(DictConfig, cfg)


@app.callback()
def _root() -> None:
    """SciField — multi-axis framework for monitoring scientific field health."""


@app.command()
def demo() -> None:
    """Run the end-to-end demo on a toy corpus."""
    cfg = _load_config("demo")
    y0, y1 = cfg.demo.year_range
    rows = fetch_demo_papers(
        journal=cfg.demo.journal,
        year_range=(int(y0), int(y1)),
        max_papers=int(cfg.demo.max_papers),
        email=cfg.demo.email,
    )

    if not rows:
        typer.echo("no papers found")
        raise typer.Exit(code=1)

    out_path = Path(cfg.demo.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out_path)

    config_dict = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
    record_run(artifact_path=out_path, inputs={}, config=config_dict)

    mean_len = mean(len(cast(str, r.get("abstract", "")) or "") for r in rows)
    typer.echo(f"n_papers={len(rows)}  mean_abstract_chars={mean_len:.0f}")


@app.command()
def harvest(
    config: str = typer.Option(
        "v1",
        "--config",
        "-c",
        help="Hydra config name under conf/corpus/",
    ),
    journal: str | None = typer.Option(
        None,
        "--journal",
        help="Limit to one journal slug",
    ),
    year: int | None = typer.Option(
        None,
        "--year",
        help="Limit to one year",
    ),
    refresh: bool = typer.Option(
        False,
        "--refresh",
        help="Re-fetch even if bucket Parquet exists",
    ),
    max_papers_per_bucket: int | None = typer.Option(
        None,
        "--max-papers-per-bucket",
        help="Smoke-test cap; harvests at most N PMIDs per (journal, year).",
    ),
) -> None:
    """Harvest the configured PubMed corpus into Parquet + DuckDB."""
    corpus_dir = Path(__file__).resolve().parents[2] / "conf" / "corpus"
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(version_base="1.3", config_dir=str(corpus_dir)):
        cfg = cast(DictConfig, hydra.compose(config_name=config))

    journals: list[JournalSpec] = [
        JournalSpec(
            slug=str(j.slug),
            display=str(j.display),
            ta_terms=[str(t) for t in j.ta_terms],
        )
        for j in cfg.journals
    ]
    if not journals:
        typer.echo("no journals configured")
        raise typer.Exit(code=1)

    year_range = (int(cfg.year_range[0]), int(cfg.year_range[1]))

    api_key = os.environ.get("NCBI_API_KEY")

    entrez_config = EntrezConfig(
        email=str(cfg.entrez.email),
        base_url=str(cfg.entrez.base_url),
        api_key=api_key,
        request_timeout_s=float(cfg.entrez.request_timeout_s),
        max_retries=int(cfg.entrez.max_retries),
    )

    rate_limit = (
        float(cfg.harvest.rate_limit_with_key) if api_key else float(cfg.harvest.rate_limit_no_key)
    )
    harvest_cfg = HarvestConfig(
        batch_size=int(cfg.harvest.batch_size),
        rate_limit=rate_limit,
        max_papers_per_bucket=max_papers_per_bucket,
    )

    output = OutputConfig(
        parquet_dir=Path(str(cfg.output.parquet_dir)),
        duckdb_path=Path(str(cfg.output.duckdb_path)),
        manifest_dir=Path(str(cfg.output.manifest_dir)),
        log_dir=Path(str(cfg.output.log_dir)),
    )

    rate_limiter = RateLimiter(rate=harvest_cfg.rate_limit)
    entrez = EntrezClient(entrez_config, rate_limiter=rate_limiter)

    async def _run() -> HarvestReport:
        try:
            return await harvest_corpus(
                journals=journals,
                year_range=year_range,
                entrez=entrez,
                output=output,
                harvest_cfg=harvest_cfg,
                refresh=refresh,
                only_journal=journal,
                only_year=year,
            )
        finally:
            await entrez.aclose()

    report = asyncio.run(_run())

    build_duckdb(
        parquet_dir=output.parquet_dir,
        duckdb_path=output.duckdb_path,
        config={
            "config_name": config,
            "only_journal": journal,
            "only_year": year,
            "refresh": refresh,
            "max_papers_per_bucket": max_papers_per_bucket,
        },
    )

    n_errors = sum(1 for b in report.buckets if b.error)
    summary = (
        f"n_papers={report.total_papers}  "
        f"n_buckets={len(report.buckets)}  "
        f"elapsed={report.elapsed_s:.1f}s"
    )
    if n_errors:
        summary += f"  n_errors={n_errors}"
    typer.echo(summary)


@app.command()
def enrich(
    config: str = typer.Option(
        "v1",
        "--config",
        "-c",
        help="Hydra config name under conf/corpus/",
    ),
    only: str | None = typer.Option(
        None,
        "--only",
        help="Run only this source (openalex|semantic_scholar|ror|authors).",
    ),
    skip: list[str] = typer.Option(  # noqa: B008
        [],
        "--skip",
        help="Source(s) to skip. Repeatable.",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        help="Smoke-test cap: process only the first N PMIDs.",
    ),
) -> None:
    """Enrich the corpus with OpenAlex + Semantic Scholar + ROR + author IDs."""
    corpus_dir = Path(__file__).resolve().parents[2] / "conf" / "corpus"
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(version_base="1.3", config_dir=str(corpus_dir)):
        cfg = cast(DictConfig, hydra.compose(config_name=config))

    valid_sources = {"openalex", "semantic_scholar", "ror", "authors"}
    if only is not None and only not in valid_sources:
        typer.echo(f"--only must be one of {sorted(valid_sources)}; got {only!r}")
        raise typer.Exit(code=2)
    skip_set = set(skip)
    bad_skip = skip_set - valid_sources
    if bad_skip:
        typer.echo(f"--skip got unknown sources: {sorted(bad_skip)}")
        raise typer.Exit(code=2)

    enrich_cfg = cfg.enrichment

    openalex_email = os.environ.get("OPENALEX_EMAIL")
    if not openalex_email:
        typer.echo("OPENALEX_EMAIL is required (polite pool); set it in env.")
        raise typer.Exit(code=2)

    enrichment_dir = Path(str(enrich_cfg.output.enrichment_dir))
    cache_dir = Path(str(enrich_cfg.output.cache_dir))
    manifest_dir = Path(str(enrich_cfg.output.manifest_dir))
    log_dir = Path(str(enrich_cfg.output.log_dir))

    paths = EnrichmentPaths(
        enrichment_dir=enrichment_dir,
        cache_dir=cache_dir,
        manifest_dir=manifest_dir,
        log_dir=log_dir,
    )

    openalex_cfg = OpenAlexConfig(
        email=openalex_email,
        api_key=os.environ.get("OPENALEX_API_KEY") or None,
        base_url=str(enrich_cfg.openalex.base_url),
        batch_size=int(enrich_cfg.openalex.batch_size),
        rate_limit=float(enrich_cfg.openalex.rate_limit),
        request_timeout_s=float(enrich_cfg.openalex.request_timeout_s),
        max_retries=int(enrich_cfg.openalex.max_retries),
        cache_dir=cache_dir / "openalex" / "raw",
        manifest_path=manifest_dir / "openalex.parquet",
    )

    ss_api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    ss_rate = (
        float(enrich_cfg.semantic_scholar.rate_limit_with_key)
        if ss_api_key
        else float(enrich_cfg.semantic_scholar.rate_limit_no_key)
    )
    ss_cfg = SemanticScholarConfig(
        base_url=str(enrich_cfg.semantic_scholar.base_url),
        api_key=ss_api_key,
        batch_size=int(enrich_cfg.semantic_scholar.batch_size),
        rate_limit=ss_rate,
        request_timeout_s=float(enrich_cfg.semantic_scholar.request_timeout_s),
        max_retries=int(enrich_cfg.semantic_scholar.max_retries),
    )

    ror_cfg = RORConfig(
        base_url=str(enrich_cfg.ror.base_url),
        rate_limit=float(enrich_cfg.ror.rate_limit),
        request_timeout_s=float(enrich_cfg.ror.request_timeout_s),
        max_retries=int(enrich_cfg.ror.max_retries),
        min_match_score=float(enrich_cfg.ror.min_match_score),
        cache_path=cache_dir / "ror" / "affiliations.parquet",
    )

    authors_cfg = AuthorsConfig(
        heuristic_salt=str(enrich_cfg.authors.heuristic_salt),
    )

    duckdb_path = Path(str(cfg.output.duckdb_path))
    if not duckdb_path.exists():
        typer.echo(f"papers DuckDB not found at {duckdb_path}; run `scifield harvest` first.")
        raise typer.Exit(code=1)

    pmids = load_pmids_from_corpus(duckdb_path, limit=limit)
    if not pmids:
        typer.echo("no PMIDs found in corpus")
        raise typer.Exit(code=1)

    config_hash_input = {
        "config_name": config,
        "only": only,
        "skip": sorted(skip_set),
        "limit": limit,
    }

    async def _run() -> Any:
        return await enrich_corpus(
            pmids=pmids,
            paths=paths,
            openalex_cfg=openalex_cfg,
            authors_cfg=authors_cfg,
            ror_cfg=ror_cfg,
            ss_cfg=ss_cfg,
            only=only,
            skip=skip_set or None,
            config_hash_input=config_hash_input,
        )

    report = asyncio.run(_run())

    registered = register_enrichment_views(
        duckdb_path=duckdb_path,
        enrichment_dir=enrichment_dir,
    )

    summary = (
        f"n_pmids={report.n_pmids_input}  "
        f"oa_ok={report.n_openalex_ok}  oa_failed={report.n_openalex_failed}  "
        f"refs={report.n_references}  authors={report.n_authorships}  "
        f"insts={report.n_institutions}  paper_insts={report.n_paper_institutions}  "
        f"ss_papers={report.n_ss_papers}{' (skipped)' if report.ss_skipped else ''}  "
        f"elapsed={report.elapsed_s:.1f}s  "
        f"sources={','.join(report.sources_run) or '-'}  "
        f"views={','.join(registered) or '-'}"
    )
    typer.echo(summary)


@app.command()
def embed(
    config: str = typer.Option(
        "config",
        "--config",
        "-c",
        help="Hydra config name (composition root); reads cfg.thematic.*",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        help="Smoke-test cap: encode only the first N abstract-bearing rows.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the row count that would be encoded and exit without loading the model.",
    ),
    device: str | None = typer.Option(
        None,
        "--device",
        help="Override sentence-transformers device (e.g. 'cpu', 'cuda', 'mps').",
    ),
    batch_size_override: int | None = typer.Option(
        None,
        "--batch-size",
        help="Override cfg.thematic.batch_size.",
    ),
) -> None:
    """Encode abstract-bearing papers into a Parquet of fp16 embeddings (V1-S05)."""
    import duckdb
    import numpy as np

    cfg = _load_config(config)
    thematic = cfg.thematic

    duckdb_path = Path(str(thematic.input.duckdb_path))
    table = str(thematic.input.table)
    filter_clause = str(thematic.input.filter)
    if not duckdb_path.exists():
        typer.echo(f"papers DuckDB not found at {duckdb_path}; run `scifield harvest` first.")
        raise typer.Exit(code=1)

    limit_clause = f"LIMIT {int(limit)}" if limit is not None else ""
    sql = (
        f"SELECT CAST(pmid AS BIGINT) AS pmid, title, abstract FROM {table} "
        f"WHERE {filter_clause} ORDER BY pmid {limit_clause}"
    ).strip()

    con = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        arrow_tbl = con.execute(sql).fetch_arrow_table()
    finally:
        con.close()

    n_papers = arrow_tbl.num_rows
    if n_papers == 0:
        typer.echo("no abstract-bearing papers matched filter")
        raise typer.Exit(code=1)

    if dry_run:
        typer.echo(
            f"dry_run=True  n_papers={n_papers}  model={thematic.model.name}  "
            f"output={thematic.output.parquet_path}"
        )
        raise typer.Exit(code=0)

    pmids_list = arrow_tbl.column("pmid").to_pylist()
    titles = arrow_tbl.column("title").to_pylist()
    abstracts = arrow_tbl.column("abstract").to_pylist()
    texts = [f"{(t or '')}. {(a or '')}" for t, a in zip(titles, abstracts, strict=False)]

    model_name = str(thematic.model.name)
    model_revision = str(thematic.model.revision)
    batch_size = (
        int(batch_size_override) if batch_size_override is not None else int(thematic.batch_size)
    )
    max_seq_length = int(thematic.model.max_seq_length)
    dtype_str = str(thematic.output.dtype)
    if dtype_str not in {"float16", "float32"}:
        typer.echo(f"unsupported output.dtype {dtype_str!r}; must be 'float16' or 'float32'")
        raise typer.Exit(code=2)
    np_dtype = np.float16 if dtype_str == "float16" else np.float32

    embedder = make_embedder(model_name, revision=model_revision, device=device)
    # Set on the public attribute; the embedder's lazy loader propagates this
    # to the underlying SentenceTransformer on first encode().
    embedder.max_seq_length = max_seq_length

    import sentence_transformers
    import torch

    t0 = time.perf_counter()
    vectors = embedder.encode(texts, batch_size=batch_size)
    runtime_s = time.perf_counter() - t0

    vectors = vectors.astype(np_dtype, copy=False)
    dim = int(vectors.shape[1])

    out_path = Path(str(thematic.output.parquet_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pa_value_type = pa.float16() if dtype_str == "float16" else pa.float32()
    try:
        list_type = pa.list_(pa_value_type, list_size=dim)
    except TypeError:  # pragma: no cover - older pyarrow without FixedSizeList kw
        list_type = pa.list_(pa_value_type)

    flat = pa.array(vectors.reshape(-1), type=pa_value_type)
    embedding_arr = pa.FixedSizeListArray.from_arrays(flat, dim)
    embedding_arr = embedding_arr.cast(list_type)

    out_table = pa.table(
        {
            "pmid": pa.array(pmids_list, type=pa.int64()),
            "embedding": embedding_arr,
            "model_name": pa.array([model_name] * n_papers, type=pa.string()),
        }
    )
    pq.write_table(out_table, out_path)

    gpu_model = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    sidecar_config = {
        "model_name": model_name,
        "model_revision": model_revision,
        "sentence_transformers_version": sentence_transformers.__version__,
        "torch_version": torch.__version__,
        "batch_size": batch_size,
        "max_seq_length": max_seq_length,
        "n_papers": n_papers,
        "total_runtime_s": round(runtime_s, 3),
        "gpu_model": gpu_model,
        "device": device or "auto",
    }
    record_run(
        artifact_path=out_path,
        inputs={"papers_duckdb": duckdb_path},
        config=sidecar_config,
    )

    typer.echo(
        f"n_papers={n_papers}  model={model_name}  dim={dim}  "
        f"runtime={runtime_s:.1f}s  output={out_path}"
    )


@app.command("faiss-build")
def faiss_build(
    config: str = typer.Option(
        "config",
        "--config",
        "-c",
        help="Hydra config name (composition root); reads cfg.thematic.faiss.*",
    ),
    embeddings: Path | None = typer.Option(  # noqa: B008
        None,
        "--embeddings",
        help="Override cfg.thematic.output.parquet_path.",
    ),
    out: Path | None = typer.Option(  # noqa: B008
        None,
        "--out",
        help="Override cfg.thematic.faiss.index_path.",
    ),
) -> None:
    """Build a FAISS HNSW index over the embeddings Parquet (V1-S05)."""
    import faiss
    import numpy as np

    cfg = _load_config(config)
    thematic = cfg.thematic

    embeddings_path = (
        Path(str(embeddings)) if embeddings else Path(str(thematic.output.parquet_path))
    )
    out_path = Path(str(out)) if out else Path(str(thematic.faiss.index_path))
    pmid_map_path = Path(str(thematic.faiss.pmid_map_path))

    if not embeddings_path.exists():
        typer.echo(f"embeddings parquet not found at {embeddings_path}")
        raise typer.Exit(code=1)

    table = pq.read_table(embeddings_path)
    pmids = table.column("pmid").to_numpy(zero_copy_only=False).astype(np.int64, copy=False)

    embedding_col = table.column("embedding")
    # Try the fast FixedSizeList path first; fall back to per-element conversion.
    arr: np.ndarray
    try:
        chunks = []
        for chunk in embedding_col.chunks:
            values = chunk.values.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
            list_size = chunk.type.list_size  # FixedSizeList only
            chunks.append(values.reshape(-1, list_size))
        arr = np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]
    except (AttributeError, TypeError):
        # Generic LIST<...> fallback: convert per row.
        py_lists = embedding_col.to_pylist()
        arr = np.asarray(py_lists, dtype=np.float32)

    if arr.ndim != 2:
        typer.echo(f"embedding column must produce a 2-D array; got shape {arr.shape}")
        raise typer.Exit(code=1)

    M = int(thematic.faiss.M)
    ef_construction = int(thematic.faiss.efConstruction)
    ef_search = int(thematic.faiss.efSearch)

    index = build_faiss_hnsw(arr, M=M, ef_construction=ef_construction, ef_search=ef_search)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pmid_map_path.parent.mkdir(parents=True, exist_ok=True)
    write_index(index, out_path)
    write_pmid_map(pmids, pmid_map_path)

    sidecar_config = {
        "M": M,
        "efConstruction": ef_construction,
        "efSearch": ef_search,
        "dim": int(arr.shape[1]),
        "n_vectors": int(arr.shape[0]),
        "faiss_version": faiss.__version__,
    }
    record_run(
        artifact_path=out_path,
        inputs={"embeddings_parquet": embeddings_path},
        config=sidecar_config,
    )
    record_run(
        artifact_path=pmid_map_path,
        inputs={"embeddings_parquet": embeddings_path},
        config=sidecar_config,
    )

    typer.echo(
        f"ntotal={index.ntotal}  dim={int(arr.shape[1])}  M={M}  "
        f"index={out_path}  pmid_map={pmid_map_path}"
    )


@app.command()
def topics(
    config: Path | None = typer.Option(  # noqa: B008
        None,
        "--config",
        "-c",
        help="Override path to topics config YAML. Defaults to conf/thematic/topics.yaml.",
    ),
    skip_sweep: bool = typer.Option(
        False,
        "--skip-sweep",
        help="Skip the sweep harness and fit a single config from defaults_config.",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        help="Smoke-test cap: use only the first N deduped (pmid, embedding) rows.",
    ),
) -> None:
    """Fit the V1-S06 BERTopic pipeline (V1-S06)."""
    import importlib.metadata as _im
    from dataclasses import asdict as _asdict

    import duckdb
    import numpy as np
    import pandas as pd

    from scifield.thematic import (
        build_hierarchy,
        ensure_papers_distinct_view,
        fit_topics,
        integrity_check_v1_carryover,
        load_deduped_embeddings,
        tokenise_for_coherence,
    )
    from scifield.thematic import sweep as run_sweep
    from scifield.thematic.topics import TopicConfig

    cfg = _load_topics_config(config)
    repo_root = Path(__file__).resolve().parents[2]
    topics_yaml_path = Path(config) if config else repo_root / "conf" / "thematic" / "topics.yaml"

    duckdb_path = Path(str(cfg.input.duckdb_path))
    embeddings_parquet = Path(str(cfg.input.embeddings_parquet))
    if not duckdb_path.exists():
        typer.echo(f"papers DuckDB not found at {duckdb_path}; run `scifield harvest` first.")
        raise typer.Exit(code=1)
    if not embeddings_parquet.exists():
        typer.echo(
            f"embeddings parquet not found at {embeddings_parquet}; run `scifield embed` first."
        )
        raise typer.Exit(code=1)

    t_total_start = time.perf_counter()

    # papers_distinct is a non-destructive VIEW; creating it requires write access,
    # so we open read-write for the dedup helpers, then reopen read-only for SELECTs.
    con_rw = duckdb.connect(str(duckdb_path))
    try:
        integrity = integrity_check_v1_carryover(con_rw)
        typer.echo(
            f"integrity: papers_total={integrity['papers_total']}  "
            f"papers_distinct={integrity['papers_distinct']}  "
            f"papers_duplicate_pmids={integrity['papers_duplicate_pmids']}"
        )
        baseline_duplicates = 13070
        observed_dups = int(integrity["papers_duplicate_pmids"])
        if abs(observed_dups - baseline_duplicates) > max(500, baseline_duplicates // 4):
            typer.echo(
                f"WARNING: papers_duplicate_pmids={observed_dups} diverges from V1-S05 "
                f"baseline ({baseline_duplicates}). Proceeding; inspect input_hashes in sidecar."
            )
        ensure_papers_distinct_view(con_rw)
    finally:
        con_rw.close()

    con = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        pmids, embeddings = load_deduped_embeddings(embeddings_parquet)
        if limit is not None:
            pmids = pmids[: int(limit)]
            embeddings = embeddings[: int(limit)]
        if pmids.shape[0] == 0:
            typer.echo("no deduped embeddings to fit")
            raise typer.Exit(code=1)

        pmid_list = pmids.tolist()
        arrow_tbl = con.execute(
            "SELECT CAST(pmid AS BIGINT) AS pmid, title, abstract "
            "FROM papers_distinct WHERE CAST(pmid AS BIGINT) IN (SELECT UNNEST(?))",
            [pmid_list],
        ).fetch_arrow_table()
    finally:
        con.close()

    pm_rows = arrow_tbl.to_pylist()
    by_pmid: dict[int, tuple[str, str]] = {
        int(r["pmid"]): (r.get("title") or "", r.get("abstract") or "") for r in pm_rows
    }
    documents: list[str] = []
    missing: list[int] = []
    for pid in pmid_list:
        ta = by_pmid.get(int(pid))
        if ta is None:
            missing.append(int(pid))
            continue
        documents.append(f"{ta[0]}. {ta[1]}")
    if missing:
        typer.echo(
            f"ERROR: {len(missing)} pmids in embeddings have no row in papers_distinct "
            f"(first 5: {missing[:5]})"
        )
        raise typer.Exit(code=1)

    coherence_texts = tokenise_for_coherence(documents)

    defaults_dict = cast(dict[str, Any], OmegaConf.to_container(cfg.defaults_config, resolve=True))

    selector = str(cfg.sweep.selector)
    constraints = cast(dict[str, Any], OmegaConf.to_container(cfg.sweep.constraints, resolve=True))
    n_leaf_min = int(constraints["n_leaf_topics_min"])
    n_leaf_max = int(constraints["n_leaf_topics_max"])
    noise_max = float(constraints["noise_fraction_max"])

    do_sweep = (not skip_sweep) and bool(cfg.sweep.enabled)
    sweep_wall = 0.0
    chosen_row_index: int | None = None
    constraints_unmet = False
    sweep_df = None

    if do_sweep:
        grid_overrides = cast(
            list[dict[str, Any]],
            OmegaConf.to_container(cfg.sweep.grid, resolve=True),
        )
        grid: list[TopicConfig] = []
        for override in grid_overrides:
            merged = dict(defaults_dict)
            merged.update(override)
            grid.append(TopicConfig(**merged))

        typer.echo(f"sweep: {len(grid)} configs")
        t_sweep_start = time.perf_counter()
        sweep_df = run_sweep(embeddings, documents, grid, coherence_texts)
        sweep_wall = time.perf_counter() - t_sweep_start

        for _, row in sweep_df.iterrows():
            typer.echo(
                f"  config={row['config']}; n_leaf={row['n_leaf_topics']}; "
                f"noise={row['noise_fraction']:.3f}; npmi={row['npmi_top10']:.3f}; "
                f"t={row['wall_seconds']:.1f}s"
                + (f"; error={row['error']}" if row.get("error") else "")
            )

        sweep_out = Path(str(cfg.output.sweep_parquet))
        sweep_out.parent.mkdir(parents=True, exist_ok=True)
        sweep_df.to_parquet(sweep_out, index=False)
        record_run(
            artifact_path=sweep_out,
            inputs={
                "papers_duckdb": duckdb_path,
                "embeddings_parquet": embeddings_parquet,
                "topics_config_yaml": topics_yaml_path,
            },
            config={
                "selector": selector,
                "constraints": constraints,
                "n_configs": len(grid),
                "grid": [_asdict(c) for c in grid],
            },
        )

        valid = sweep_df[sweep_df["error"].isna()].copy()
        eligible = valid[
            (valid["n_leaf_topics"] >= n_leaf_min)
            & (valid["n_leaf_topics"] <= n_leaf_max)
            & (valid["noise_fraction"] <= noise_max)
        ]
        if len(eligible) == 0:
            constraints_unmet = True
            typer.echo(
                "WARNING: no sweep config satisfies constraints "
                f"(n_leaf_topics in [{n_leaf_min},{n_leaf_max}], noise<={noise_max:.2f}). "
                "Falling back to global selector argmax."
            )
            if len(valid) == 0:
                typer.echo("ERROR: every sweep config errored out; cannot pick a winner.")
                raise typer.Exit(code=1)
            for _, row in valid.nlargest(3, selector).iterrows():
                typer.echo(
                    f"  closest: n_leaf={row['n_leaf_topics']}  "
                    f"noise={row['noise_fraction']:.3f}  {selector}={row[selector]:.3f}"
                )
            best_idx = int(valid[selector].idxmax())
        else:
            best_idx = int(eligible[selector].idxmax())
        chosen_row_index = best_idx
        chosen_dict = dict(sweep_df.loc[best_idx, "config"])
        typer.echo(
            f"chosen row {best_idx}: {chosen_dict}  "
            f"({selector}={sweep_df.loc[best_idx, selector]:.3f}, "
            f"n_leaf={sweep_df.loc[best_idx, 'n_leaf_topics']}, "
            f"noise={sweep_df.loc[best_idx, 'noise_fraction']:.3f}, "
            f"constraints_met={not constraints_unmet})"
        )
        chosen_cfg = TopicConfig(**chosen_dict)
    else:
        typer.echo("sweep skipped; using defaults_config")
        chosen_cfg = TopicConfig(**defaults_dict)

    typer.echo(f"fitting final model on {len(documents)} documents")
    t_fit_start = time.perf_counter()
    model = fit_topics(embeddings, documents, chosen_cfg)
    fit_wall = time.perf_counter() - t_fit_start

    t_hier_start = time.perf_counter()
    hier_df = build_hierarchy(
        model,
        documents,
        target_mid_levels=int(cfg.hierarchy.target_mid_levels),
        target_top_levels=int(cfg.hierarchy.target_top_levels),
    )
    hier_wall = time.perf_counter() - t_hier_start

    hier_out = Path(str(cfg.output.hierarchy_parquet))
    hier_out.parent.mkdir(parents=True, exist_ok=True)
    hier_df.to_parquet(hier_out, index=False)

    topics_arr = np.asarray(model.topics_)
    if topics_arr.shape[0] != pmids.shape[0]:
        typer.echo(f"ERROR: model.topics_ length {topics_arr.shape[0]} != n_pmids {pmids.shape[0]}")
        raise typer.Exit(code=1)
    assignments = pd.DataFrame(
        {
            "pmid": pmids.astype(np.int64),
            "topic_id": topics_arr.astype(np.int64),
            "is_noise": topics_arr == -1,
        }
    )
    topics_out = Path(str(cfg.output.topics_parquet))
    topics_out.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_parquet(topics_out, index=False)

    model_dir = Path(str(cfg.output.model_dir))
    model_dir.mkdir(parents=True, exist_ok=True)
    try:
        model.save(str(model_dir), serialization="safetensors", save_ctfidf=True)
        typer.echo(f"saved model with save_ctfidf=True to {model_dir}")
    except TypeError:
        model.save(str(model_dir), serialization="safetensors")
        typer.echo(f"saved model without save_ctfidf kwarg to {model_dir}")

    n_unique_leaf = int(len({int(t) for t in topics_arr.tolist() if int(t) != -1}))
    noise_fraction = float((topics_arr == -1).sum() / max(1, topics_arr.shape[0]))
    n_mid = int(hier_df["mid_level_id"].nunique()) if len(hier_df) else 0
    n_top = int(hier_df["top_level_id"].nunique()) if len(hier_df) else 0

    from scifield.thematic.coherence import compute_coherence
    from scifield.thematic.topics import _topic_words as _tw

    unique_leaf_ids = sorted({int(t) for t in topics_arr.tolist() if int(t) != -1})
    chosen_word_lists = [_tw(model, t, top_n=10) for t in unique_leaf_ids]
    coh = compute_coherence(chosen_word_lists, coherence_texts, top_n=10)

    def _ver(pkg: str, mod_attr: Any = None) -> str:
        if mod_attr is not None:
            try:
                return str(mod_attr.__version__)
            except AttributeError:
                pass
        try:
            return _im.version(pkg)
        except Exception:
            return "unknown"

    import bertopic as _bertopic
    import gensim as _gensim
    import sklearn as _sklearn
    import umap as _umap

    software_versions = {
        "bertopic": _ver("bertopic", _bertopic),
        "umap_learn": _ver("umap-learn", _umap),
        "hdbscan": _ver("hdbscan"),
        "gensim": _ver("gensim", _gensim),
        "numpy": _ver("numpy", np),
        "sklearn": _ver("scikit-learn", _sklearn),
    }

    total_wall = time.perf_counter() - t_total_start

    sidecar_config: dict[str, Any] = {
        "chosen_config": _asdict(chosen_cfg),
        "chosen_row_index": chosen_row_index,
        "constraints": constraints,
        "selector": selector,
        "constraints_unmet": constraints_unmet,
        "hierarchy": {
            "target_mid_levels": int(cfg.hierarchy.target_mid_levels),
            "target_top_levels": int(cfg.hierarchy.target_top_levels),
        },
        "n_pmids_deduped": int(pmids.shape[0]),
        "n_leaf_topics": n_unique_leaf,
        "n_mid_levels": n_mid,
        "n_top_levels": n_top,
        "noise_fraction": noise_fraction,
        "coherence": {
            "c_npmi": float(coh.get("c_npmi", float("nan"))),
            "c_v": float(coh.get("c_v", float("nan"))),
        },
        "wall_seconds": {
            "sweep": round(sweep_wall, 3),
            "final_fit": round(fit_wall, 3),
            "hierarchy": round(hier_wall, 3),
            "total": round(total_wall, 3),
        },
        "thread_count": int(os.cpu_count() or 1),
        "software_versions": software_versions,
        "deviations": ({"constraints_unmet": True} if constraints_unmet else {}),
        "integrity_check": integrity,
        "skip_sweep": bool(skip_sweep),
        "limit": int(limit) if limit is not None else None,
    }

    sidecar_inputs = {
        "papers_duckdb": duckdb_path,
        "embeddings_parquet": embeddings_parquet,
        "topics_config_yaml": topics_yaml_path,
    }
    record_run(artifact_path=topics_out, inputs=sidecar_inputs, config=sidecar_config)
    record_run(artifact_path=hier_out, inputs=sidecar_inputs, config=sidecar_config)
    record_run(artifact_path=model_dir, inputs=sidecar_inputs, config=sidecar_config)

    typer.echo(
        f"done: n_leaf={n_unique_leaf}  n_mid={n_mid}  n_top={n_top}  "
        f"noise={noise_fraction:.3f}  npmi={sidecar_config['coherence']['c_npmi']:.3f}  "
        f"cv={sidecar_config['coherence']['c_v']:.3f}  total={total_wall:.1f}s"
    )


@epistemic_app.command("sample")
def epistemic_sample(
    config: str = typer.Option("v1", "--config", "-c"),
    n_sample: int | None = typer.Option(None, "--n-sample"),
    seed: int | None = typer.Option(None, "--seed"),
) -> None:
    """Draw the stratified hand-labeling sample (V1-S07)."""
    import duckdb
    import pandas as pd

    from scifield.epistemic.sampling import SamplingConfig, stratified_sample

    cfg = _load_epistemic_config(config)

    duckdb_path = Path(str(cfg.input.duckdb_path))
    topics_parquet = Path(str(cfg.input.topics_parquet))
    sample_path = Path(str(cfg.output.sample_path))

    if not duckdb_path.exists():
        typer.echo(f"papers DuckDB not found at {duckdb_path}; run `scifield harvest` first.")
        raise typer.Exit(code=1)
    if not topics_parquet.exists():
        typer.echo(f"topics parquet not found at {topics_parquet}; run `scifield topics` first.")
        raise typer.Exit(code=1)

    sampling_cfg = SamplingConfig(
        duckdb_path=duckdb_path,
        topics_parquet=topics_parquet,
        n_sample=int(n_sample) if n_sample is not None else int(cfg.sampling.n_sample),
        seed=int(seed) if seed is not None else int(cfg.sampling.seed),
        eras=tuple(str(e) for e in cfg.sampling.eras),
        topic_coverage_min=int(cfg.sampling.topic_coverage_min),
    )

    # stratified_sample internally calls ensure_papers_distinct_view, which
    # issues CREATE OR REPLACE VIEW — open RW.
    con = duckdb.connect(str(duckdb_path))
    try:
        df: pd.DataFrame = stratified_sample(con, sampling_cfg)
    finally:
        con.close()

    sample_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, sample_path)

    n_cells = int(df[["journal", "era"]].drop_duplicates().shape[0])
    n_topics = int(df["topic_id"].dropna().nunique())

    record_run(
        artifact_path=sample_path,
        inputs={
            "papers_duckdb": duckdb_path,
            "topics_parquet": topics_parquet,
        },
        config={
            "n_sample": int(sampling_cfg.n_sample),
            "seed": int(sampling_cfg.seed),
            "eras": list(sampling_cfg.eras),
            "topic_coverage_min": int(sampling_cfg.topic_coverage_min),
            "n_cells": n_cells,
            "n_topics": n_topics,
        },
    )

    typer.echo(f"sampled n={len(df)} cells={n_cells} topic_coverage={n_topics} out={sample_path}")


@epistemic_app.command("export-labels")
def epistemic_export_labels(
    config: str = typer.Option("v1", "--config", "-c"),
    rater: str = typer.Option(
        ...,
        "--rater",
        help="Rater name (used in the xlsx filename and instructions sheet).",
    ),
    sample: Path | None = typer.Option(  # noqa: B008
        None,
        "--sample",
        help="Override path to the sample parquet.",
    ),
    out: Path | None = typer.Option(  # noqa: B008
        None,
        "--out",
        help="Override output xlsx path.",
    ),
) -> None:
    """Export a per-rater Excel labeling workbook from the sample parquet (V1-S07)."""
    from scifield.epistemic.labeling import export_to_xlsx

    cfg = _load_epistemic_config(config)

    sample_path = Path(sample) if sample is not None else Path(str(cfg.output.sample_path))
    out_path = (
        Path(out)
        if out is not None
        else Path(str(cfg.output.labels_xlsx_dir)) / f"labels_{rater}.xlsx"
    )

    if not sample_path.exists():
        typer.echo(
            f"sample parquet not found at {sample_path}; run `scifield epistemic sample` first."
        )
        raise typer.Exit(code=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_xlsx(sample_path, out_path, rater)
    typer.echo(f"wrote {out_path}")


@epistemic_app.command("import-labels")
def epistemic_import_labels(
    config: str = typer.Option("v1", "--config", "-c"),
    rater: str = typer.Option(..., "--rater"),
    file: Path = typer.Option(..., "--file"),  # noqa: B008
) -> None:
    """Import a filled-in xlsx into the long-form handlabel parquet (V1-S07)."""
    from scifield.epistemic.labeling import import_from_xlsx

    cfg = _load_epistemic_config(config)
    parquet_out = Path(str(cfg.output.handlabel_parquet))

    if not file.exists():
        typer.echo(f"labels xlsx not found at {file}")
        raise typer.Exit(code=1)

    summary = import_from_xlsx(file, rater, parquet_out)

    for e in summary["errors"]:
        typer.echo(f"  row={e['row']} pmid={e['pmid']} error={e['error']}")

    typer.echo(
        f"n_rows={summary['n_rows']} n_imported={summary['n_imported']} "
        f"n_errors={summary['n_errors']} out={summary['out_path']}"
    )

    if summary["n_errors"] > 0:
        raise typer.Exit(code=1)


@epistemic_app.command("pilot")
def epistemic_pilot(
    config: str = typer.Option("v1", "--config", "-c"),
    n: int | None = typer.Option(None, "--n"),
) -> None:
    """Run the 50-abstract Claude-Code pilot extractor (V1-S07)."""
    # Lazy imports: pilot.py is shipped by sibling Batch 4A; deferring the
    # import until the command body runs avoids any import-time race during
    # parallel batch development.
    from scifield.epistemic.extract import ExtractConfig
    from scifield.epistemic.pilot import PilotConfig, run_pilot

    cfg = _load_epistemic_config(config)

    extract_cfg = ExtractConfig(
        claude_cmd=tuple(str(c) for c in cfg.pilot.claude_cmd),
        model_id=str(cfg.model.id),
        prompt_version=str(cfg.prompt.version),
    )
    pilot_cfg = PilotConfig(
        sample_path=Path(str(cfg.output.sample_path)),
        pilot_path=Path(str(cfg.output.pilot_path)),
        pilot_failed_path=Path(str(cfg.output.pilot_failed_path)),
        n_pilot=int(n) if n is not None else int(cfg.pilot.n_pilot),
        extract_cfg=extract_cfg,
    )

    def progress(i: int, total: int, outcome: str) -> None:
        typer.echo(f"  [{i}/{total}] {outcome}")

    summary = run_pilot(pilot_cfg, progress=progress)

    typer.echo(
        f"pilot done: n_ok={summary.get('n_ok', 0)} "
        f"n_failed={summary.get('n_failed', 0)} "
        f"out={summary.get('pilot_path', pilot_cfg.pilot_path)} "
        f"failed_out={summary.get('pilot_failed_path', pilot_cfg.pilot_failed_path)}"
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()

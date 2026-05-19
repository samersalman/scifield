"""Command-line interface for scifield."""

from __future__ import annotations

import asyncio
import os
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
    EntrezClient,
    EntrezConfig,
    HarvestConfig,
    HarvestReport,
    JournalSpec,
    OutputConfig,
    RateLimiter,
    build_duckdb,
    harvest_corpus,
)
from scifield.corpus.pubmed_demo import fetch_demo_papers
from scifield.repro import record_run

app = typer.Typer(
    name="scifield",
    help="SciField — multi-axis framework for monitoring scientific field health.",
    no_args_is_help=True,
)


def _load_config(name: str) -> DictConfig:
    """Compose a Hydra config from the repo's `conf/` directory."""
    conf_dir = Path(__file__).resolve().parents[2] / "conf"
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(version_base="1.3", config_dir=str(conf_dir)):
        cfg = hydra.compose(config_name=name)
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


def main() -> None:
    app()


if __name__ == "__main__":
    main()

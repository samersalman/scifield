"""Command-line interface for scifield."""

from __future__ import annotations

from pathlib import Path
from statistics import mean
from typing import Any, cast

import hydra
import pyarrow as pa
import pyarrow.parquet as pq
import typer
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

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


def main() -> None:
    app()


if __name__ == "__main__":
    main()

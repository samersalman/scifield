"""Build notebooks/01_corpus_overview.ipynb per plan-s3 §7.

Run once with `uv run python scripts/_build_notebook_01.py`. This file may be
deleted after the notebook is generated; it exists only to author the .ipynb
deterministically.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO_ROOT = Path(__file__).resolve().parents[1]
NB_PATH = REPO_ROOT / "notebooks" / "01_corpus_overview.ipynb"


CELL_0_MD = """# Corpus v1 — descriptive overview

Reads `data/v1/papers.duckdb` (a view layer over `data/v1/parquet/<journal_slug>/<year>.parquet`).
Re-execute after a smoke run or after the overnight harvest:
```bash
uv run jupyter execute notebooks/01_corpus_overview.ipynb
```"""


CELL_1_SETUP = """from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd

DUCKDB_PATH = Path("data/v1/papers.duckdb")
assert DUCKDB_PATH.exists(), f"missing {DUCKDB_PATH}; run `uv run scifield harvest` first."
con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
con.execute("SELECT 1").fetchone()"""


CELL_2_COUNTS = """total = con.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
print(f"total papers: {total:,}")

journals_df = con.execute(
    "SELECT journal_slug, journal, journal_ta, COUNT(*) AS n_papers "
    "FROM papers GROUP BY 1, 2, 3 ORDER BY n_papers DESC"
).fetch_df()
journals_df"""


CELL_3_TREND = """trend = con.execute(
    "SELECT journal_slug, year, COUNT(*) AS n "
    "FROM papers WHERE year IS NOT NULL "
    "GROUP BY journal_slug, year ORDER BY journal_slug, year"
).fetch_df()

fig, ax = plt.subplots(figsize=(10, 5))
for slug, grp in trend.groupby("journal_slug"):
    ax.plot(grp["year"], grp["n"], marker="o", markersize=3, label=slug)
ax.set_xlabel("Publication year")
ax.set_ylabel("Papers")
ax.set_title("Papers per year per journal")
ax.legend(loc="best", fontsize=7, ncol=2)
fig.tight_layout()
plt.show()"""


CELL_4_ABSLEN = """abs_lens = con.execute(
    "SELECT LENGTH(abstract) AS chars FROM papers WHERE has_abstract"
).fetch_df()

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(abs_lens["chars"], bins=60)
ax.set_xlabel("Abstract length (characters)")
ax.set_ylabel("Papers")
ax.set_title(f"Abstract length distribution (n={len(abs_lens):,})")
fig.tight_layout()
plt.show()"""


CELL_5_ERA = '''era_q = """
SELECT
  journal_slug,
  CASE
    WHEN year < 2000 THEN 'pre-2000'
    WHEN year BETWEEN 2000 AND 2009 THEN '2000-2009'
    WHEN year BETWEEN 2010 AND 2019 THEN '2010-2019'
    WHEN year >= 2020 THEN '2020+'
    ELSE 'unknown'
  END AS era,
  AVG(CASE WHEN has_abstract THEN 1.0 ELSE 0.0 END) AS pct_with_abstract
FROM papers
GROUP BY 1, 2
"""
era = con.execute(era_q).fetch_df()
era_pivot = era.pivot(index="journal_slug", columns="era", values="pct_with_abstract")
era_pivot = era_pivot.reindex(columns=["pre-2000", "2000-2009", "2010-2019", "2020+"])

fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(era_pivot))))
im = ax.imshow(era_pivot.values, vmin=0, vmax=1, aspect="auto", cmap="viridis")
ax.set_xticks(range(era_pivot.shape[1]))
ax.set_xticklabels(era_pivot.columns)
ax.set_yticks(range(era_pivot.shape[0]))
ax.set_yticklabels(era_pivot.index)
for i in range(era_pivot.shape[0]):
    for j in range(era_pivot.shape[1]):
        v = era_pivot.values[i, j]
        if pd.isna(v):
            continue
        color = "red" if v < 0.90 else "white"
        ax.text(j, i, f"{v:.0%}", ha="center", va="center", color=color, fontsize=9)
plt.colorbar(im, ax=ax, label="% with abstract")
ax.set_title("Abstract availability by journal × era (red text: <90%)")
fig.tight_layout()
plt.show()'''


CELL_6_MESH = '''mesh_q = """
SELECT journal_slug,
  AVG(CASE WHEN LENGTH(mesh_headings) > 0 THEN 1.0 ELSE 0.0 END) AS pct_with_mesh
FROM papers
GROUP BY 1
ORDER BY pct_with_mesh DESC
"""
mesh_df = con.execute(mesh_q).fetch_df()

fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(mesh_df))))
ax.barh(mesh_df["journal_slug"], mesh_df["pct_with_mesh"])
ax.set_xlim(0, 1)
ax.set_xlabel("% papers with ≥1 MeSH heading")
ax.set_title("MeSH coverage by journal")
ax.invert_yaxis()
fig.tight_layout()
plt.show()'''


CELL_7_PRE2000 = """pre2000 = con.execute(
    "SELECT AVG(CASE WHEN has_abstract THEN 1.0 ELSE 0.0 END) AS pct "
    "FROM papers WHERE year < 2000"
).fetchone()[0]
n_pre2000 = con.execute(
    "SELECT COUNT(*) FROM papers WHERE year < 2000"
).fetchone()[0]
print(f"Pre-2000 abstract availability: {pre2000:.1%} of {n_pre2000:,} papers")
print("Append this number to docs/phases/1_corpus.md (manual, not auto-generated).")
con.close()"""


def main() -> None:
    nb = new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
    nb.metadata["language_info"] = {"name": "python"}

    nb.cells = [
        new_markdown_cell(CELL_0_MD),
        new_code_cell(CELL_1_SETUP),
        new_code_cell(CELL_2_COUNTS),
        new_code_cell(CELL_3_TREND),
        new_code_cell(CELL_4_ABSLEN),
        new_code_cell(CELL_5_ERA),
        new_code_cell(CELL_6_MESH),
        new_code_cell(CELL_7_PRE2000),
    ]

    # Belt-and-suspenders: ensure every code cell has empty outputs and no exec count.
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, NB_PATH)
    print(f"wrote {NB_PATH}")


if __name__ == "__main__":
    main()

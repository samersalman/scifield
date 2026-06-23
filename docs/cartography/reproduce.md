# Reproduce

How to rebuild the cartography map and docs from this repository, verify
provenance, and run the go-live / DOI release. Everything below is `$0` and
local except the final maintainer release step.

The map you see at `…/cartography/map/` is a **static, self-contained artifact**
checked into the docs site. Rebuilding it regenerates that artifact from the
derived tables under `V2/data_v2/`.

!!! note "Local environment note"
    On macOS the build needs `KMP_DUPLICATE_LIB_OK=TRUE` in the environment to
    avoid an OpenMP duplicate-runtime abort (torch + duckdb). It is included in
    the commands below.

---

## 1. Rebuild the map

This regenerates the map HTML and **publishes the docs copy** in one step
(`--publish-to` writes the identical HTML to `docs/cartography/map/index.html`,
alongside its `.run.json` provenance sidecar):

```bash
KMP_DUPLICATE_LIB_OK=TRUE .venv/bin/python V2/scripts/build_map.py --data-version v2 --publish-to docs/cartography/map
```

- `--data-version v2` builds the **78-journal expansion** map (the published
  cartography). The default `v1` builds the frozen 10-journal prove-phase map.
- `--publish-to docs/cartography/map` writes the page the docs site serves.

## 2. Run the tests

```bash
KMP_DUPLICATE_LIB_OK=TRUE uv run pytest tests/ -q
```

## 3. Build the docs (what CI / GitHub Pages runs)

```bash
uv sync --group docs && uv run mkdocs build --strict
```

This is the exact build the live Pages deploy runs. It must exit 0. The
mkdocs-material output prints a red "mkdocs 2.0" announcement banner — that is
**not** an error; only a non-zero exit or a `WARNING`/`ERROR` about a page or
link is a real failure.

---

## 4. Verify provenance (sidecars)

Every derived artifact ships a machine-readable provenance sidecar next to it,
named `<file>.run.json`. Each sidecar records:

- `git_sha` / `git_dirty` — the repo commit the artifact was built from;
- `input_hashes` — content hashes of the upstream inputs (detects input drift);
- `config` / `config_hash` — the parameters the artifact was built with;
- `software_versions` and a UTC `timestamp`.

The published map carries its own sidecar at
`docs/cartography/map/index.html.run.json`, and every table under
`V2/data_v2/**` has a `<table>.parquet.run.json`. Treat the sidecar — not the
prose — as the authoritative reproducibility record. The per-table schema is in
the [Data dictionary](data-dictionary.md#provenance-sidecars-parquetrunjson).

The **full upstream pipeline** that produces those tables (harvest → enrich →
embed → cluster → cartography) is documented on GitHub under
[`V2/docs/cartography/`](https://github.com/samersalman/scifield/tree/main/V2/docs/cartography)
(charter, validation protocol, and the per-layer findings write-ups).

## 5. Data deposit

The derived cartography tables (~1.7 MB) ship **in the repository** under
`V2/data_v2/` and are catalogued in the
[Data dictionary](data-dictionary.md). The **raw 1.49M-paper corpus** is
**not** redistributed — it is reproducible-from-pipeline (see the upstream
pipeline docs linked above and the repository `README.md`).

---

## 6. Go-live & DOI handoff runbook

The release is a one-button maintainer action on top of an automated pipeline:

1. **Push `main`.** Pushing to `main` deploys the docs site, making **GitHub
   Pages live** (this cartography section included).
2. **Tag + draft release (automated).** An annotated tag **`v0.2.0`** and a
   **draft GitHub Release** are created for that commit.
3. **Maintainer publishes the draft Release** (the one-click action).
   Publishing the Release triggers **Zenodo**, which **auto-archives the tagged
   source and mints the DOI**.
4. **Backfill the DOI.** Once minted, add the DOI to:
   - the **README badge** (replace the `ZENODO_DOI` placeholder), and
   - the **Zenodo record** / the citation block on the
     [Overview](index.md#citation-license) and
     [Data dictionary](data-dictionary.md#citation-license) pages.

!!! warning "Do not cite a DOI before it is minted"
    The DOI does not exist until the maintainer publishes the draft Release in
    step 3. The README badge and the in-docs citation blocks intentionally carry
    a clearly-marked **TODO placeholder** until then — do not fabricate or cite
    a DOI before step 3 completes.

import json
from pathlib import Path

from scifield.repro import record_run


def test_record_run_writes_sidecar(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.parquet"
    artifact.write_bytes(b"fake parquet bytes")

    input_file = tmp_path / "input.csv"
    input_file.write_text("col\n1\n")

    sidecar = record_run(
        artifact_path=artifact,
        inputs={"input": input_file},
        config={"k": "v"},
    )

    assert sidecar.exists()
    payload = json.loads(sidecar.read_text())
    expected_keys = {
        "git_sha",
        "git_dirty",
        "config_hash",
        "input_hashes",
        "software_versions",
        "timestamp",
    }
    assert expected_keys <= payload.keys()
    assert "input" in payload["input_hashes"]

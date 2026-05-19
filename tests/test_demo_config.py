from pathlib import Path

import hydra
from hydra.core.global_hydra import GlobalHydra


def test_demo_config_has_expected_keys() -> None:
    # Locate conf/ relative to repo root (test is at tests/test_demo_config.py,
    # repo root is one level up)
    conf_dir = Path(__file__).resolve().parents[1] / "conf"
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(version_base="1.3", config_dir=str(conf_dir)):
        cfg = hydra.compose(config_name="demo")
    expected = {"journal", "year_range", "max_papers", "email", "output_path"}
    assert expected <= set(cfg.demo.keys())

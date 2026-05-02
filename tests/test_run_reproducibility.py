"""Tests for the scripts/run_reproducibility.py harness.

Strategy: import the harness as a module (it lives under scripts/, which
is not on the import path by default), monkeypatch STEP_MODULES to point
at fake modules so we can verify dispatch order, error handling, and
manifest contents without invoking the real analysis modules.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def _load_harness():
    """Import scripts/run_reproducibility.py as a module."""
    path = REPO_ROOT / "scripts" / "run_reproducibility.py"
    spec = importlib.util.spec_from_file_location("run_reproducibility", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# Fake module factory
# --------------------------------------------------------------------------- #
def _install_fake_module(name: str, behaviour="ok", call_log=None):
    """Register a synthetic module with a run(config) function."""
    mod = types.ModuleType(name)

    def run(cfg):  # noqa: ANN001
        if call_log is not None:
            call_log.append(name)
        if behaviour == "ok":
            return None
        if behaviour == "raise":
            raise RuntimeError(f"forced failure in {name}")
        return None

    mod.run = run
    sys.modules[name] = mod
    return mod


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
@pytest.fixture
def harness(monkeypatch):
    h = _load_harness()
    # Replace the real STEP_MODULES with synthetic ones we fully control.
    fake_steps = {
        "fc": "fake_fc_mod",
        "reho": "fake_reho_mod",
        "ica": "fake_ica_mod",
        "graph": "fake_graph_mod",
        "scorecard": "fake_scorecard_mod",
    }
    monkeypatch.setattr(h, "STEP_MODULES", fake_steps)
    yield h, fake_steps


class TestDispatch:
    def test_runs_all_in_order(self, tmp_path, harness, monkeypatch) -> None:
        h, fake_steps = harness
        log: list = []
        for name, modname in fake_steps.items():
            _install_fake_module(modname, behaviour="ok", call_log=log)

        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(
            yaml.safe_dump(
                {
                    "paths": {"output_root": str(tmp_path / "out")},
                    "execution": {
                        "run_order": ["fc", "reho", "ica", "graph", "scorecard"],
                        "continue_on_error": True,
                        "write_manifest": True,
                    },
                }
            )
        )
        rc = h.main(["--config", str(cfg_path)])
        assert rc == 0
        assert log == ["fake_fc_mod", "fake_reho_mod", "fake_ica_mod",
                       "fake_graph_mod", "fake_scorecard_mod"]

        manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
        assert manifest["overall_status"] == "ok"
        assert [s["step"] for s in manifest["steps"]] == [
            "fc", "reho", "ica", "graph", "scorecard"
        ]
        assert all(s["status"] == "ok" for s in manifest["steps"])

    def test_continue_on_error_keeps_going(self, tmp_path, harness) -> None:
        h, fake_steps = harness
        for name, modname in fake_steps.items():
            behaviour = "raise" if name == "ica" else "ok"
            _install_fake_module(modname, behaviour=behaviour)

        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(
            yaml.safe_dump(
                {
                    "paths": {"output_root": str(tmp_path / "out")},
                    "execution": {
                        "run_order": ["fc", "reho", "ica", "graph", "scorecard"],
                        "continue_on_error": True,
                    },
                }
            )
        )
        rc = h.main(["--config", str(cfg_path)])
        assert rc == 1  # at least one error
        manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
        # All steps still appear in manifest; ica is recorded as "error".
        statuses = {s["step"]: s["status"] for s in manifest["steps"]}
        assert statuses["ica"] == "error"
        assert statuses["fc"] == "ok"
        assert statuses["graph"] == "ok"
        assert statuses["scorecard"] == "ok"
        assert manifest["overall_status"] == "errors"

    def test_halt_on_error_stops_after_first_failure(self, tmp_path, harness) -> None:
        h, fake_steps = harness
        for name, modname in fake_steps.items():
            behaviour = "raise" if name == "fc" else "ok"
            _install_fake_module(modname, behaviour=behaviour)

        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(
            yaml.safe_dump(
                {
                    "paths": {"output_root": str(tmp_path / "out")},
                    "execution": {
                        "run_order": ["fc", "reho", "graph"],
                        "continue_on_error": False,
                    },
                }
            )
        )
        rc = h.main(["--config", str(cfg_path)])
        assert rc == 1
        manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
        steps = [s["step"] for s in manifest["steps"]]
        # Only fc should have been attempted.
        assert steps == ["fc"]

    def test_only_filter(self, tmp_path, harness) -> None:
        h, fake_steps = harness
        log: list = []
        for name, modname in fake_steps.items():
            _install_fake_module(modname, behaviour="ok", call_log=log)
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(
            yaml.safe_dump(
                {
                    "paths": {"output_root": str(tmp_path / "out")},
                    "execution": {"run_order": list(fake_steps.keys())},
                }
            )
        )
        rc = h.main(["--config", str(cfg_path), "--only", "fc,scorecard"])
        assert rc == 0
        assert log == ["fake_fc_mod", "fake_scorecard_mod"]

    def test_skip_filter(self, tmp_path, harness) -> None:
        h, fake_steps = harness
        log: list = []
        for name, modname in fake_steps.items():
            _install_fake_module(modname, behaviour="ok", call_log=log)
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(
            yaml.safe_dump(
                {
                    "paths": {"output_root": str(tmp_path / "out")},
                    "execution": {"run_order": list(fake_steps.keys())},
                }
            )
        )
        rc = h.main(["--config", str(cfg_path), "--skip", "ica,graph"])
        assert rc == 0
        # ica + graph excluded, others run.
        assert "fake_ica_mod" not in log
        assert "fake_graph_mod" not in log
        assert "fake_fc_mod" in log

    def test_dry_run_does_not_execute(self, tmp_path, harness) -> None:
        h, fake_steps = harness
        log: list = []
        for name, modname in fake_steps.items():
            _install_fake_module(modname, behaviour="ok", call_log=log)
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(
            yaml.safe_dump(
                {
                    "paths": {"output_root": str(tmp_path / "out")},
                    "execution": {"run_order": list(fake_steps.keys())},
                }
            )
        )
        rc = h.main(["--config", str(cfg_path), "--dry-run"])
        assert rc == 0
        assert log == []
        assert not (tmp_path / "out" / "manifest.json").exists()

    def test_unknown_step_is_rejected(self, tmp_path, harness) -> None:
        h, _ = harness
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(
            yaml.safe_dump(
                {
                    "paths": {"output_root": str(tmp_path / "out")},
                    "execution": {"run_order": ["fc", "no_such_step"]},
                }
            )
        )
        rc = h.main(["--config", str(cfg_path)])
        assert rc == 2

    def test_disabled_steps_are_skipped(self, tmp_path, harness) -> None:
        h, fake_steps = harness
        log: list = []
        for name, modname in fake_steps.items():
            _install_fake_module(modname, behaviour="ok", call_log=log)
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(
            yaml.safe_dump(
                {
                    "paths": {"output_root": str(tmp_path / "out")},
                    "execution": {"run_order": ["fc", "reho", "scorecard"]},
                    "fc": {"enabled": False},
                    "reho": {"enabled": True},
                }
            )
        )
        rc = h.main(["--config", str(cfg_path)])
        assert rc == 0
        # fc disabled -> not called; reho/scorecard called.
        assert "fake_fc_mod" not in log
        assert "fake_reho_mod" in log
        assert "fake_scorecard_mod" in log

        manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
        statuses = {s["step"]: s["status"] for s in manifest["steps"]}
        assert statuses["fc"] == "skipped"

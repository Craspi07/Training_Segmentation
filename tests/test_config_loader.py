"""Tests for config_loader.py."""

import pytest


class TestLoadConfig:
    def test_defaults_merged(self, tmp_path):
        from cellseg_trainer.config_loader import load_config, DEFAULTS

        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("TRAIN:\n  MAX_ITER: 9999\n")
        cfg = load_config(cfg_file)
        assert cfg["TRAIN"]["MAX_ITER"] == 9999
        # Defaults preserved
        assert cfg["TRAIN"]["BASE_LR"] == DEFAULTS["TRAIN"]["BASE_LR"]

    def test_missing_file(self):
        from cellseg_trainer.config_loader import load_config

        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_save_and_reload(self, tmp_path):
        from cellseg_trainer.config_loader import load_config, save_config, DEFAULTS

        path = tmp_path / "saved.yaml"
        save_config(DEFAULTS, path)
        cfg = load_config(path)
        assert cfg["TRAIN"]["MAX_ITER"] == DEFAULTS["TRAIN"]["MAX_ITER"]


class TestOverrideFromArgs:
    def test_dot_notation(self):
        from cellseg_trainer.config_loader import override_from_args

        cfg = {"TRAIN": {"MAX_ITER": 100}}
        result = override_from_args(cfg, **{"TRAIN.MAX_ITER": 999})
        assert result["TRAIN"]["MAX_ITER"] == 999

    def test_none_skipped(self):
        from cellseg_trainer.config_loader import override_from_args

        cfg = {"TRAIN": {"MAX_ITER": 100}}
        result = override_from_args(cfg, **{"TRAIN.MAX_ITER": None})
        assert result["TRAIN"]["MAX_ITER"] == 100

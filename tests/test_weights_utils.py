"""Tests for prepare_weights_for_detectron2 in utils.py."""

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")


class TestPrepareWeightsForDetectron2:
    def test_missing_file_raises(self, tmp_path):
        from cellseg_trainer.utils import prepare_weights_for_detectron2

        with pytest.raises(FileNotFoundError):
            prepare_weights_for_detectron2(tmp_path / "nonexistent.pt")

    def test_raw_state_dict_pt(self, tmp_path):
        """A plain .pt state_dict should be wrapped in {"model": ...}."""
        import torch
        from cellseg_trainer.utils import prepare_weights_for_detectron2

        state_dict = {"layer1.weight": torch.zeros(4, 4), "layer1.bias": torch.zeros(4)}
        pt_path = tmp_path / "model.pt"
        torch.save(state_dict, str(pt_path))

        out = prepare_weights_for_detectron2(pt_path, cache_dir=tmp_path)
        loaded = torch.load(out, map_location="cpu")
        assert "model" in loaded
        assert "layer1.weight" in loaded["model"]

    def test_already_d2_format_pth(self, tmp_path):
        """A .pth already containing {"model": ...} should be returned as-is."""
        import torch
        from cellseg_trainer.utils import prepare_weights_for_detectron2

        d2_ckpt = {"model": {"layer.weight": torch.zeros(2, 2)}, "iteration": 100}
        pth_path = tmp_path / "model_final.pth"
        torch.save(d2_ckpt, str(pth_path))

        out = prepare_weights_for_detectron2(pth_path, cache_dir=tmp_path)
        assert out == str(pth_path)  # unchanged

    def test_pkl_passthrough(self, tmp_path):
        """A .pkl file should be returned unchanged (Detectron2/Caffe2 format)."""
        from cellseg_trainer.utils import prepare_weights_for_detectron2

        pkl_path = tmp_path / "model.pkl"
        pkl_path.write_bytes(b"")  # empty — just need the file to exist
        out = prepare_weights_for_detectron2(pkl_path, cache_dir=tmp_path)
        assert out == str(pkl_path)

    def test_module_prefix_stripped(self, tmp_path):
        """Keys with 'module.' prefix (from DataParallel) should be cleaned."""
        import torch
        from cellseg_trainer.utils import prepare_weights_for_detectron2

        state_dict = {"module.layer.weight": torch.zeros(3, 3)}
        pt_path = tmp_path / "ddp_model.pt"
        torch.save(state_dict, str(pt_path))

        out = prepare_weights_for_detectron2(pt_path, cache_dir=tmp_path)
        loaded = torch.load(out, map_location="cpu")
        assert "layer.weight" in loaded["model"]
        assert "module.layer.weight" not in loaded["model"]

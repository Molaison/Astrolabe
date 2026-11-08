import json
import argparse
import logging
import pathlib
import sys
from collections import OrderedDict
from contextlib import contextmanager
from threading import RLock
from typing import Dict, List

import torch
from ml_collections import ConfigDict

from astrolabe.mutators.runtime_stages import guarded_device


_ESM_LOADER_LOCK = RLock()


@contextmanager
def _local_esm_serialization(model_path):
    """Allow legacy Namespace metadata only inside the selected local loader.

    Static inspection of the original t33 checkpoint requires only Namespace;
    the contact-regression checkpoint requires no additional globals. Keep
    weights_only enabled, leave torch.load untouched, and restore the vendor's
    imported loader alias even if model construction fails.
    """
    import carbondesign.data.esm as embeddings

    selected = pathlib.Path(model_path).expanduser().resolve(strict=True)
    if not selected.is_file():
        raise FileNotFoundError(selected)
    with _ESM_LOADER_LOCK:
        original = embeddings.load_model_and_alphabet_local

        def local_loader(path, *args, **kwargs):
            if pathlib.Path(path).expanduser().resolve() != selected:
                return original(path, *args, **kwargs)
            with torch.serialization.safe_globals([argparse.Namespace]):
                return original(path, *args, **kwargs)

        embeddings.load_model_and_alphabet_local = local_loader
        try:
            yield
        finally:
            embeddings.load_model_and_alphabet_local = original


class CarbonDesignRunner:
    """Thin wrapper that exposes CarbonDesign inference as a reusable runner."""

    @guarded_device("carbondesign", phase="load")
    def __init__(self, args) -> None:
        self.args = args
        self.repo_path = self._resolve_absolute_path(self._get("repo_path"))
        self.output_dir = pathlib.Path(
            self._get("output_dir", self.repo_path / "results")
        ).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = self._select_device(self._get("device", "cuda"))
        self.batch_size = int(self._get("batch_size", 1))
        self.max_seq_len = int(self._get("max_seq_len", 600))
        self.temperature = float(self._get("temperature", 0.0))
        self.num_aa = 20

        self._extend_sys_path()
        from carbondesign.model.carbondesign import CarbonDesign
        from carbondesign.testloader import dataset_test

        self._dataset_test = dataset_test
        self._feats = self._load_feats(self._get("model_features"), self.device)
        model_config = self._load_config(self._get("model_config"))
        model_config = self._patch_model_paths(model_config)
        self.model = self._load_model(
            CarbonDesign, model_config, self._get("model_ckpt")
        )

    def prepare_design(self, data_dir: str, name_idx: List[str]):
        return str(data_dir), list(name_idx)

    @guarded_device("carbondesign")
    def execute_design(self, prepared):
        data_dir, name_idx = prepared
        if not name_idx:
            return []
        loader = self._dataset_test.load(
            data_dir=str(data_dir),
            name_idx=list(name_idx),
            feats=self._feats,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )
        outputs = []
        with torch.no_grad():
            # Feature transforms are configured with the actual device; retain
            # the iterator in this subgraph until vendor transforms are traced.
            for batch in loader:
                if batch is None:
                    continue
                if self.max_seq_len > 0 and batch["seq"].shape[1] > self.max_seq_len:
                    logging.warning("Skipping batch with length %s (max_seq_len=%s)",
                                    batch["seq"].shape[1], self.max_seq_len)
                    continue
                ret = self.model(batch=batch, compute_loss=True)
                logits = ret["heads"]["seqhead"]["logits"]
                temperature = self.temperature if self.temperature > 0 else 1.0
                probs = torch.softmax(logits[..., :self.num_aa] / temperature, dim=-1)
                mask = batch["mask"].to(torch.bool)
                outputs.append((batch["name"], probs.detach().cpu(), mask.cpu()))
        return outputs

    @staticmethod
    def finalize_design(batches):
        outputs = {}
        for names, probs, mask in batches:
            for i, name in enumerate(names):
                valid_len = int(mask[i].sum().item())
                outputs[name] = {"sampling_probs": probs[i:i+1, :valid_len]}
        return outputs

    def design_sequences(self, data_dir: str, name_idx: List[str]):
        return self.finalize_design(self.execute_design(self.prepare_design(data_dir, name_idx)))

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _get(self, key, default=None):
        if hasattr(self.args, key):
            return getattr(self.args, key)
        if isinstance(self.args, dict):
            return self.args.get(key, default)
        return default

    def _extend_sys_path(self) -> None:
        repo_str = str(self.repo_path)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

    def _resolve_absolute_path(self, path_value) -> pathlib.Path:
        if path_value is None:
            raise ValueError("Required CarbonDesign path is not provided.")
        path = pathlib.Path(path_value).expanduser()
        if not path.is_absolute():
            path = (pathlib.Path.cwd() / path).resolve()
        return path

    def _resolve_repo_path(self, path_value) -> pathlib.Path:
        if path_value is None:
            raise ValueError("Required CarbonDesign path is not provided.")
        path = pathlib.Path(path_value).expanduser()
        if not path.is_absolute():
            path = (self.repo_path / path).resolve()
        return path

    def _select_device(self, requested: str) -> torch.device:
        if requested == "cpu":
            return torch.device("cpu")
        if torch.cuda.is_available():
            gpu_idx = self._get("gpu_idx", [0])
            idx = gpu_idx[0] if isinstance(gpu_idx, (list, tuple)) and gpu_idx else 0
            return torch.device(f"cuda:{idx}")
        logging.warning("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")

    def _load_config(self, config_path) -> ConfigDict:
        config_path = self._resolve_repo_path(config_path)
        with config_path.open("r", encoding="utf-8") as f:
            content = json.load(f)
        return ConfigDict(content)

    def _patch_model_paths(self, config: ConfigDict) -> ConfigDict:
        lm = getattr(config.model.embeddings_and_seqformer, "lm", None)
        if lm is not None and hasattr(lm, "model_path"):
            lm_path = pathlib.Path(lm.model_path)
            if not lm_path.is_absolute():
                lm.model_path = str((self.repo_path / lm_path).resolve())
        return config

    def _load_feats(self, feats_path, device: torch.device):
        feats_path = self._resolve_repo_path(feats_path)
        with feats_path.open("r", encoding="utf-8") as f:
            feats = json.load(f)

        processed = []
        for feat_name, feat_args in feats:
            feat_args = dict(feat_args)
            if feat_args.get("device") == "%(device)s":
                feat_args["device"] = device
            processed.append((feat_name, feat_args))
        return processed

    def _load_model(self, model_cls, config: ConfigDict, ckpt_path):
        ckpt_path = self._resolve_repo_path(ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        lm = getattr(config.model.embeddings_and_seqformer, "lm", None)
        if lm is not None and getattr(lm, "enabled", False):
            with _local_esm_serialization(lm.model_path):
                model = model_cls(config=config.model)
        else:
            model = model_cls(config=config.model)
        state_dict = OrderedDict()
        for key, value in checkpoint["model"].items():
            clean_key = key[len("module.") :] if key.startswith("module.") else key
            state_dict[clean_key] = value
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()
        return model

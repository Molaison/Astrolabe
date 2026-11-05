"""Staged native NUMonomer inference without the upstream CLI or worker pool.

Upstream reference: yunda-si/NUMonomer@af38434e7ee3f82c6948841cce1ca4d65840742a.
"""

from __future__ import annotations

from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import random
import sys
from typing import Any


UPSTREAM_REVISION = "af38434e7ee3f82c6948841cce1ca4d65840742a"


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"NUMonomer {label} must be an integer >= {minimum}")
    return value


@contextmanager
def _imports(repo: Path):
    # Upstream uses generic absolute names, including model, config, np and utils.
    roots = {p.stem for p in repo.glob("*.py")} | {"np", "utils"}
    saved = {k: v for k, v in sys.modules.copy().items() if k.split(".")[0] in roots}
    original_path = sys.path[:]
    for key in saved:
        del sys.modules[key]
    sys.path.insert(0, str(repo))
    try:
        yield
    finally:
        for key in list(sys.modules):
            if key.split(".")[0] in roots:
                del sys.modules[key]
        sys.modules.update(saved)
        sys.path[:] = original_path


class NUMonomerBackend:
    """Native MonomerDataset, model, confidence selection, and save_struc.

    ``weights_path`` is a checkpoint file. Native defaults are eight recycles,
    a 512-residue confidence prefix, no chunking, CIF output, and confidence
    selection rather than ``save_last``. ``topology`` defaults to linear.
    """

    def __init__(self, settings: dict):
        self.settings = dict(settings)
        if not settings.get("repo_path") or not settings.get("weights_path"):
            raise ValueError("NUMonomer requires repo_path and weights_path")
        self.repo = Path(settings["repo_path"]).expanduser().resolve()
        self.weights = Path(settings["weights_path"]).expanduser().resolve()
        for label, path in (("repo_path", self.repo), ("weights_path", self.weights),
                            ("dataset.py", self.repo / "dataset.py"),
                            ("model.py", self.repo / "model.py"),
                            ("config.py", self.repo / "config.py"),
                            ("save_struc.py", self.repo / "utils/save_struc.py")):
            if not path.exists() or (label != "repo_path" and not path.is_file()):
                raise FileNotFoundError(f"NUMonomer {label} is missing: {path}")
        self.num_iter = _integer(settings.get("num_iter", 8), "num_iter", 1)
        self.clamp_plddt = _integer(settings.get("clamp_plddt", 512), "clamp_plddt", 1)
        self.split_seq = _integer(settings.get("split_seq", 0), "split_seq")
        self.split_atom = _integer(settings.get("split_atom", 0), "split_atom")
        self.save_last = settings.get("save_last", False)
        if not isinstance(self.save_last, bool):
            raise ValueError("NUMonomer save_last must be a boolean")
        self.ftype = settings.get("ftype", "cif")
        if self.ftype not in {"pdb", "cif"}:
            raise ValueError("NUMonomer ftype must be pdb or cif")
        self.topology = settings.get("topology", "linear")
        if self.topology not in {"linear", "circular"}:
            raise ValueError("NUMonomer topology must be linear or circular")
        self.initial_seed = _integer(settings.get("seed", 42), "seed")
        if self.initial_seed >= 2**32:
            raise ValueError("NUMonomer seed must be less than 2**32")

        import numpy as np
        import torch

        self.np, self.torch = np, torch
        with torch.random.fork_rng(devices=[]), _imports(self.repo):
            torch.random.default_generator.manual_seed(self.initial_seed)
            cfg = importlib.import_module("config").get_cfg()
            self.model_module = importlib.import_module("model")
            self.dataset_class = importlib.import_module("dataset").MonomerDataset
            self.save_struc = importlib.import_module("utils.save_struc")
            self.compute_plddt = importlib.import_module("utils.seq_utils").compute_plddt
            self.model_kwargs = {key: getattr(cfg.model, key) for key in (
                "blocks_structure_decoder", "blocks_confidence", "atom_channel", "atom_nhead",
                "pair_channel", "pair_nhead", "dropout_p", "dropout_p2d", "num_atom")}
            self.model_kwargs.update(split_seq=self.split_seq or cfg.inference.split_seq,
                                     split_atom=self.split_atom or cfg.inference.split_atom)
            self.model = self.model_module.NUMonomer(**self.model_kwargs)
            self.initial_cpu_rng = torch.get_rng_state().clone()
            self.model.load_state_dict(torch.load(self.weights, map_location="cpu", weights_only=True))
            self.model.eval()
        self.native_log_memory = self.model_module.log_memory
        self.device = None

    def place(self, device: str):
        target = self.torch.device(device)
        if target.type not in {"cpu", "cuda"}:
            raise ValueError(f"NUMonomer does not support device {device!r}")
        if target.type == "cuda" and not self.torch.cuda.is_available():
            raise RuntimeError(f"NUMonomer requested {device}, but CUDA is unavailable")
        self.model.to(target)
        # The native diagnostic calls CUDA memory APIs even for a CPU model.
        self.model_module.log_memory = self.native_log_memory if target.type == "cuda" else lambda _: None
        self.device = target

    def prepare(self, request: dict) -> dict:
        seq = request.get("sequence")
        chain = request.get("chain_id", "A")
        if not isinstance(seq, str) or not seq or set(seq) - set("ACGU"):
            raise ValueError("NUMonomer requires one nonempty, unmodified A/C/G/U RNA sequence")
        if not isinstance(chain, str) or not chain or not chain.isascii() or not chain.isalnum():
            raise ValueError("NUMonomer requires one alphanumeric chain_id")
        if self.ftype == "pdb" and len(chain) != 1:
            raise ValueError("NUMonomer PDB output requires a one-character chain_id")
        topology = request.get("topology", self.topology)
        if topology not in {"linear", "circular"}:
            raise ValueError("NUMonomer topology must be linear or circular")
        seed = _integer(request.get("seed", self.initial_seed), "seed")
        if seed >= 2**32:
            raise ValueError("NUMonomer seed must be less than 2**32")
        if not request.get("output_dir"):
            raise ValueError("NUMonomer requires a request-private output_dir")
        output_dir = Path(request["output_dir"]).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        fasta = output_dir / "input.fasta"
        fasta.write_text(f">{chain}|rna|{topology}\n{seq}\n")
        torch = self.torch
        random.seed(seed)
        self.np.random.seed(seed)
        with torch.random.fork_rng(devices=[]):
            if seed == self.initial_seed:
                torch.set_rng_state(self.initial_cpu_rng)
            else:
                # Dataset coordinates follow model initialization in the native CLI.
                torch.random.default_generator.manual_seed(seed)
                initialized = self.model_module.NUMonomer(**self.model_kwargs)
                del initialized
            dataset = self.dataset_class(targets_list=[str(fasta)], num_structure_recycle=self.num_iter,
                                         clamp_plddt=self.clamp_plddt, mask_seq=False)
            features = next(iter(torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=0)))
            cpu_rng = torch.get_rng_state().clone()
        return {"features": features, "sequence": seq, "seed": seed, "cpu_rng": cpu_rng,
                "output_dir": str(output_dir), "topology": topology}

    def execute(self, prepared: dict) -> dict:
        if self.device is None:
            raise RuntimeError("NUMonomer place(device) must precede execute")
        torch = self.torch
        random.seed(prepared["seed"])
        self.np.random.seed(prepared["seed"])
        torch.manual_seed(prepared["seed"])
        torch.set_rng_state(prepared["cpu_rng"])
        host_features = prepared["features"]
        features = {key: value.to(self.device) if torch.is_tensor(value) else value
                    for key, value in host_features.items()}
        with torch.no_grad(), torch.autocast(
                device_type=self.device.type, dtype=torch.bfloat16,
                enabled=torch.amp.autocast_mode.is_autocast_available(self.device.type)):
            preds = self.model(features, last_only=self.save_last)
        with torch.no_grad():
            pred_lddts = torch.repeat_interleave(self.compute_plddt(preds["pred_plddts"]).unsqueeze(-1), 28, -1)
            means = pred_lddts.mean(dim=(1, 2, 3, 4))
            idx = len(preds["pred_coords"]) - 1 if self.save_last else int(torch.argmax(means).item())
            coords = preds["pred_coords"][idx, 0].detach().float().cpu()
            confidence = pred_lddts[idx, 0, ..., -1].detach().float().cpu().reshape(-1)
            iteration_means = means.detach().float().cpu().tolist()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return {"coords": coords, "confidence": confidence,
                "label_seq": host_features["label_seq"], "sel_idx": host_features["sel_idx"],
                "mol_type": host_features["mol_type"], "chain_id": host_features["chain_list"],
                "sequence": prepared["sequence"], "output_dir": prepared["output_dir"],
                "seed": prepared["seed"], "topology": prepared["topology"],
                "selected_output_index": idx, "iteration_mean_plddt": iteration_means}

    def finalize(self, host_payload: dict) -> dict:
        output_dir = Path(host_payload["output_dir"])
        structure = output_dir / f"pred_{self.weights.stem}_{host_payload['seed']}.{self.ftype}"
        self.save_struc.to_pdb(host_payload["coords"], host_payload["confidence"],
                               host_payload["label_seq"].reshape(-1), host_payload["sel_idx"],
                               host_payload["mol_type"], host_payload["chain_id"], str(structure), self.ftype)
        if not structure.is_file() or structure.stat().st_size == 0:
            raise RuntimeError(f"NUMonomer did not write a structure: {structure}")
        confidence = host_payload["confidence"].tolist()
        full_confidence = len(confidence) == len(host_payload["sequence"])
        metadata = {"upstream_revision": UPSTREAM_REVISION, "seed": host_payload["seed"],
                    "topology": host_payload["topology"], "save_last": self.save_last,
                    "num_iter": self.num_iter, "clamp_plddt": self.clamp_plddt,
                    "selected_output_index": host_payload["selected_output_index"],
                    "selected_recycle": self.num_iter - 1 if self.save_last else host_payload["selected_output_index"],
                    "iteration_mean_plddt": host_payload["iteration_mean_plddt"],
                    "confidence_length": len(confidence), "confidence_scale": "0-100",
                    "confidence_prefix": confidence,
                    "confidence_is_full_length": full_confidence,
                    "unscored_bfactors": "Upstream save_struc pads the unscored tail with zeros, not predicted confidence." if not full_confidence else None}
        log = output_dir / f"log_{self.weights.stem}_{host_payload['seed']}.json"
        log.write_text(json.dumps(metadata, indent=2) + "\n")
        return {"structure_path": str(structure.resolve()),
                "plddt": confidence if full_confidence else None,
                "artifacts": {"confidence_log": str(log)}, "metadata": metadata}

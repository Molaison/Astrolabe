"""Staged native DRfold2 inference, followed by CPU PotentialFold and Arena.

Upstream reference: leeyang/DRfold2@3667d6ac44c5b14cff94c36e0371224acb2cdae5.
"""

from __future__ import annotations

import ast
from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
from types import ModuleType
from typing import Any


UPSTREAM_REVISION = "3667d6ac44c5b14cff94c36e0371224acb2cdae5"
_EXPERIMENTS = ("cfg_95", "cfg_96", "cfg_97", "cfg_99")


def _file(value: Any, label: str) -> Path:
    if not value:
        raise ValueError(f"DRfold2 requires {label}")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"DRfold2 {label} is missing: {path}")
    return path


def _integer(value: Any, label: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"DRfold2 {label} must be an integer in [{low}, {high}]")
    return value


@contextmanager
def _imports(directory: Path):
    # The four configurations reuse absolute module names for different code.
    roots = {p.stem for p in directory.glob("*.py")} | {"RNALM2"}
    saved = {k: v for k, v in sys.modules.copy().items() if k.split(".")[0] in roots}
    original_path = sys.path[:]
    for key in saved:
        del sys.modules[key]
    sys.path.insert(0, str(directory))
    try:
        yield
    finally:
        for key in list(sys.modules):
            if key.split(".")[0] in roots:
                del sys.modules[key]
        sys.modules.update(saved)
        sys.path[:] = original_path


def _network_module(directory: Path, rclm_path: Path) -> ModuleType:
    path = directory / "EvoMSA2XYZ.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    replacements = {"device": "cpu", "saved_model": str(rclm_path)}
    found = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in replacements:
                node.value = ast.copy_location(ast.Constant(replacements[target.id]), node.value)
                found.append(target.id)
    if sorted(found) != sorted(replacements):
        raise RuntimeError(f"Unsupported DRfold2 import bootstrap in {path}")
    module = ModuleType("EvoMSA2XYZ")
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[module.__name__] = module
    # Retain upstream model code while removing argv/device and fixed-path coupling.
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), module.__dict__)
    return module


@dataclass
class _Experiment:
    name: str
    module: ModuleType
    data: ModuleType
    util: ModuleType
    model: Any
    base: Any
    checkpoints: list[tuple[str, Any]]


class DRfold2Backend:
    """One owner, one RNA chain, and a host-only payload at each stage boundary.

    ``weights_path`` is the model_hub directory, containing RCLM/epoch_67000
    and cfg_*/ checkpoints. Defaults retain all four configurations and every
    native checkpoint. ``experiments`` and ``checkpoints`` can explicitly select
    subsets; ``num_ensemble`` and ``num_cycles`` can reduce the native 3 and 8.
    CPU refinement keeps upstream fixed seeds, torch=6 and NumPy/Python=9.
    """

    def __init__(self, settings: dict):
        self.settings = dict(settings)
        if not settings.get("repo_path") or not settings.get("weights_path"):
            raise ValueError("DRfold2 requires repo_path and weights_path")
        self.repo = Path(settings["repo_path"]).expanduser().resolve()
        self.weights = Path(settings["weights_path"]).expanduser().resolve()
        if not self.repo.is_dir():
            raise FileNotFoundError(f"DRfold2 repo_path is missing: {self.repo}")
        experiments = settings.get("experiments", _EXPERIMENTS)
        if (not isinstance(experiments, (list, tuple)) or not experiments
                or len(set(experiments)) != len(experiments)
                or any(name not in _EXPERIMENTS for name in experiments)):
            raise ValueError(f"DRfold2 experiments must be a nonempty subset of {_EXPERIMENTS}")
        self.num_ensemble = _integer(settings.get("num_ensemble", 3), "num_ensemble", 1, 3)
        self.num_cycles = _integer(settings.get("num_cycles", 8), "num_cycles", 1, 8)
        rclm = _file(settings.get("rclm_path", self.weights / "RCLM/epoch_67000"), "RCLM checkpoint")
        self.selection_config = _file(settings.get("selection_config", self.repo / "cfg_for_selection.json"), "selection_config")
        self.folding_config = _file(settings.get("folding_config", self.repo / "cfg_for_folding.json"), "folding_config")
        self.selection_script = _file(self.repo / "PotentialFold/Selection.py", "Selection.py")
        self.optimization_script = _file(self.repo / "PotentialFold/Optimization.py", "Optimization.py")
        self.arena = _file(settings.get("arena_path", self.repo / "Arena/Arena"), "Arena executable")
        if not os.access(self.arena, os.X_OK):
            raise PermissionError(f"DRfold2 Arena is not executable: {self.arena}")
        self.python_bin = str(_file(settings.get("python_bin", sys.executable), "python_bin"))
        self.timeout = settings.get("subprocess_timeout")
        if self.timeout is not None and (not isinstance(self.timeout, (int, float)) or self.timeout <= 0):
            raise ValueError("DRfold2 subprocess_timeout must be positive")
        selected = settings.get("checkpoints", {})
        if not isinstance(selected, dict) or set(selected) - set(experiments):
            raise ValueError("DRfold2 checkpoints must map selected experiments to checkpoint filenames")
        assets = []
        for name in experiments:
            directory = self.repo / name
            for filename in ("EvoMSA2XYZ.py", "data.py", "util.py", "base.npy"):
                _file(directory / filename, f"{name}/{filename}")
            weight_dir = self.weights / name
            if not weight_dir.is_dir():
                raise FileNotFoundError(f"DRfold2 checkpoint directory is missing: {weight_dir}")
            available = sorted(p.name for p in weight_dir.iterdir() if p.is_file() and "model" in p.name and "opt" not in p.name)
            names = selected.get(name, available)
            if (not isinstance(names, (list, tuple)) or not names or len(set(names)) != len(names)
                    or any(n not in available for n in names)):
                raise ValueError(f"DRfold2 {name} needs native checkpoint filenames; available={available}")
            assets.append((name, directory, [(n, weight_dir / n) for n in sorted(names)]))

        import numpy as np
        import torch

        self.np, self.torch = np, torch
        self.rclm_state = torch.load(rclm, map_location="cpu", weights_only=True)
        self.experiments = []
        for name, directory, checkpoints in assets:
            with torch.random.fork_rng(devices=[]), _imports(directory):
                torch.random.default_generator.manual_seed(0)
                module = _network_module(directory, rclm)
                data = importlib.import_module("data")
                util = importlib.import_module("util")
                model = module.MSA2XYZ(6, 7, self.num_ensemble, self.num_cycles, 64, 64, 64)
                states = [(n, torch.load(p, map_location="cpu", weights_only=True)) for n, p in checkpoints]
                for _, state in states:
                    model.load_state_dict(state, strict=name != "cfg_99")
                model.eval()
                self.experiments.append(_Experiment(name, module, data, util, model,
                                                   np.load(directory / "base.npy"), states))
        self.device = None

    def place(self, device: str):
        torch = self.torch
        target = torch.device(device)
        if target.type not in {"cpu", "cuda"}:
            raise ValueError(f"DRfold2 does not support device {device!r}")
        if target.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"DRfold2 requested {device}, but CUDA is unavailable")
        for experiment in self.experiments:
            experiment.model.to(target)
            experiment.module.RNAlm.to(target)
            experiment.module.device = str(target)
        self.device = target

    def prepare(self, request: dict) -> dict:
        seq = request.get("sequence")
        chain = request.get("chain_id", "A")
        if not isinstance(seq, str) or not seq or set(seq) - set("ACGU"):
            raise ValueError("DRfold2 requires one nonempty, unmodified A/C/G/U RNA sequence")
        if not isinstance(chain, str) or len(chain) != 1 or not chain.isascii() or not chain.isalnum():
            raise ValueError("DRfold2 PDB output requires one alphanumeric chain_id character")
        seed = _integer(request.get("seed", 0), "seed", 0, 2**32 - 1)
        if not request.get("output_dir"):
            raise ValueError("DRfold2 requires a request-private output_dir")
        output_dir = Path(request["output_dir"]).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        fasta = output_dir / "input.fasta"
        fasta.write_text(f">{chain}\n{seq}\n")
        features = []
        torch = self.torch
        for experiment in self.experiments:
            # Strict=False leaves initialized weights in use, so replay native
            # CPU construction for each request seed rather than reusing leftovers.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed)
                language_model = experiment.module.Model.RNA2nd(experiment.module.lmcfg)
                language_model.load_state_dict(self.rclm_state, strict=False)
                model = experiment.module.MSA2XYZ(6, 7, self.num_ensemble, self.num_cycles, 64, 64, 64)
                model_state = model.state_dict()
                language_state = language_model.state_dict()
                cpu_rng = torch.get_rng_state().clone()
            msa = torch.from_numpy(experiment.data.parse_seq(seq)[None, :])
            msa = torch.nn.functional.one_hot(torch.cat([msa, msa], 0).long(), 6).float()
            features.append({"msa": msa,
                             "base": torch.from_numpy(experiment.data.Get_base(seq, experiment.base)).float(),
                             "seq_idx": torch.arange(1, len(seq) + 1, dtype=torch.long),
                             "model_state": model_state, "language_state": language_state,
                             "cpu_rng": cpu_rng})
        return {"sequence": seq, "chain_id": chain, "seed": seed, "output_dir": str(output_dir),
                "fasta": str(fasta), "features": features}

    def execute(self, prepared: dict) -> dict:
        if self.device is None:
            raise RuntimeError("DRfold2 place(device) must precede execute")
        torch, np = self.torch, self.np
        results = []
        with torch.no_grad():
            for experiment, features in zip(self.experiments, prepared["features"]):
                seed = prepared["seed"]
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.set_rng_state(features["cpu_rng"])
                experiment.model.load_state_dict(features["model_state"])
                experiment.module.RNAlm.load_state_dict(features["language_state"])
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                msa = features["msa"].to(self.device)
                base = features["base"].to(self.device)
                seq_idx = features["seq_idx"].to(self.device)
                for name, state in experiment.checkpoints:
                    experiment.model.load_state_dict(state, strict=experiment.name != "cfg_99")
                    ret = experiment.model.pred(msa, seq_idx, None, base, np.array(list(prepared["sequence"])))
                    # Native pred returns NumPy arrays; finish any tensor downloads here.
                    ret = {key: value.detach().cpu().numpy() if torch.is_tensor(value) else value
                           for key, value in ret.items()}
                    results.append((experiment.name, name, ret))
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return {"sequence": prepared["sequence"], "chain_id": prepared["chain_id"],
                "seed": prepared["seed"], "output_dir": prepared["output_dir"],
                "fasta": prepared["fasta"], "results": results}

    def finalize(self, host_payload: dict) -> dict:
        output_dir = Path(host_payload["output_dir"])
        ret_dir, folds, relax = (output_dir / name for name in ("rets_dir", "folds", "relax"))
        for directory in (ret_dir, folds, relax):
            directory.mkdir(parents=True, exist_ok=True)
        ret_files = []
        utilities = {e.name: e.util for e in self.experiments}
        seq = host_payload["sequence"]
        for experiment, name, ret in host_payload["results"]:
            prefix = ret_dir / f"{experiment}_{name}"
            utilities[experiment].outpdb(ret["coor"], self.np.arange(1, len(seq) + 1), seq, str(prefix) + ".pdb")
            path = Path(str(prefix) + ".ret")
            with path.open("wb") as handle:
                pickle.dump(ret, handle)
            ret_files.append(str(path))
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        selection = folds / "sel_0"
        opt_prefix = folds / "opt_0"
        commands = [
            [self.python_bin, str(self.selection_script), host_payload["fasta"], str(self.selection_config), str(selection), *ret_files],
            [self.python_bin, str(self.optimization_script), host_payload["fasta"], str(opt_prefix), str(ret_dir), str(selection), str(self.folding_config)],
        ]
        logs = []
        for label, command in zip(("selection", "optimization"), commands):
            log = output_dir / f"{label}.log"
            with log.open("w") as handle:
                subprocess.run(command, cwd=self.repo, env=env, stdout=handle, stderr=subprocess.STDOUT,
                               check=True, timeout=self.timeout)
            logs.append(str(log))
        candidates = sorted(folds.glob("opt_0*.pdb"))
        if len(candidates) != 1:
            raise RuntimeError(f"DRfold2 expected one optimized structure, found {candidates}")
        structure = relax / "model_1.pdb"
        log = output_dir / "arena.log"
        with log.open("w") as handle:
            subprocess.run([str(self.arena), str(candidates[0]), str(structure), "7"], cwd=self.repo,
                           env=env, stdout=handle, stderr=subprocess.STDOUT, check=True, timeout=self.timeout)
        _file(structure, "Arena output")
        chain = host_payload["chain_id"]
        if chain != "A":
            lines = structure.read_text().splitlines(keepends=True)
            structure.write_text("".join(line[:21] + chain + line[22:] if line.startswith(("ATOM  ", "HETATM", "TER   ")) and len(line) > 21 else line for line in lines))
        return {"structure_path": str(structure.resolve()), "plddt": None,
                "artifacts": {"native_predictions": ret_files, "selection_scores": str(selection),
                              "coarse_structure": str(candidates[0]), "logs": [*logs, str(log)]},
                "metadata": {"upstream_revision": UPSTREAM_REVISION, "seed": host_payload["seed"],
                             "experiments": [e.name for e in self.experiments],
                             "num_ensemble": self.num_ensemble, "num_cycles": self.num_cycles,
                             "confidence": "Native pairwise confidence is retained in .ret files; no residue pLDDT exists for the refined structure.",
                             "refinement_seeds": {"torch": 6, "numpy": 9, "python": 9}}}

"""RNAbpFlow library adapter without vendoring its GPL algorithm source.

``repo_path`` and ``weights_path`` select the upstream checkout and checkpoint.
Conditioning is an explicit ordered ``map_paths`` list containing one, two, or
three L x L npy maps, or ``conditioning_path`` naming one such map. A literal
``conditioning_dot_bracket`` is also supported. Using request
``target_dot_bracket`` instead requires ``use_target_dot_bracket=True`` in
backend settings. Explicit request seeds start a consecutive sample-seed
sequence. Without a request seed, the native 420, 840, ... schedule is used.
No optimization target is read implicitly. ``samples_per_sequence`` and
``num_timesteps`` override the upstream inference configuration. The first
sample is returned without ranking, and all samples remain artifacts.
"""

from __future__ import annotations

import importlib
import string
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


UPSTREAM_REVISION = "e8b1c077f7951d4d6ebe00c851d22c37777dfc9f"
UPSTREAM_URL = "https://github.com/Bhattacharya-Lab/RNAbpFlow"


@dataclass(frozen=True)
class _Prepared:
    sequence: str
    chain_id: str
    output_dir: Path
    seeds: tuple[int, ...]
    ss: Any
    onehot: Any
    mapfeat: dict[str, Any]
    conditioning: dict[str, Any]


def _existing_path(value: Any, label: str) -> Path:
    if not value:
        raise ValueError(f"RNAbpFlow requires explicit {label}")
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"RNAbpFlow {label} does not exist: {path}")
    return path


def _dot_bracket_map(dot_bracket: str, length: int):
    import numpy as np

    if len(dot_bracket) != length:
        raise ValueError("RNAbpFlow conditioning dot bracket must match sequence length")
    pairs = dict(zip("([{<" + string.ascii_uppercase, ")]}>" + string.ascii_lowercase))
    stacks = {opening: [] for opening in pairs}
    closing = {right: left for left, right in pairs.items()}
    matrix = np.zeros((length, length), dtype=np.float32)
    for index, symbol in enumerate(dot_bracket):
        if symbol in stacks:
            stacks[symbol].append(index)
        elif symbol in closing and stacks[closing[symbol]]:
            partner = stacks[closing[symbol]].pop()
            matrix[index, partner] = matrix[partner, index] = 1
        elif symbol != ".":
            raise ValueError("RNAbpFlow conditioning contains invalid or unmatched dot bracket")
    if any(stacks.values()):
        raise ValueError("RNAbpFlow conditioning contains unmatched dot bracket")
    return matrix


class RNAbpFlowBackend:
    """Prepare on CPU, sample natively on the placed device, serialize on CPU."""

    def __init__(self, settings: dict):
        self.settings = dict(settings)
        self.repo = _existing_path(settings.get("repo_path"), "repo_path")
        self.weights = _existing_path(settings.get("weights_path"), "weights_path")
        config_path = _existing_path(
            settings.get("config_path", self.repo / "configs/inference.yaml"), "config_path"
        )
        sys.path.insert(0, str(self.repo))

        def upstream(name):
            module = importlib.import_module(name)
            if not Path(module.__file__).resolve().is_relative_to(self.repo):
                raise ImportError(f"RNAbpFlow import {name} resolves outside {self.repo}")
            return module

        import torch
        from omegaconf import OmegaConf

        self.transforms = upstream("src.data.data_transform")
        self.constants = upstream("src.data.base_constants")
        self.nucleicacid = upstream("src.data.nucleicacid")
        flow_api = upstream("src.models.flow_module_inf")
        config = OmegaConf.load(config_path).inference
        self.sample_count = int(settings.get("samples_per_sequence", config.samples.samples_per_sequence))
        self.timesteps = int(settings.get("num_timesteps", config.interpolant.sampling.num_timesteps))
        if self.sample_count < 1 or self.timesteps < 2:
            raise ValueError("RNAbpFlow requires positive sample count and at least two timesteps")
        with torch.device("cpu"):
            self.flow = flow_api.FlowModule.load_from_checkpoint(
                checkpoint_path=str(self.weights), map_location="cpu"
            )
        self.flow.eval()
        self.device = torch.device("cpu")
        self.flow.interpolant.set_device(self.device)

    def place(self, device: str):
        import torch

        self.device = torch.device(device)
        self.flow.to(self.device)
        self.flow.interpolant.set_device(self.device)

    def _conditioning(self, request, length):
        import numpy as np

        options = dict(self.settings)
        options.update(request)
        paths = options.get("map_paths")
        single = options.get("conditioning_path")
        dot_bracket = options.get("conditioning_dot_bracket")
        use_target = bool(self.settings.get("use_target_dot_bracket", False))
        if use_target:
            if dot_bracket is not None:
                raise ValueError("Choose conditioning_dot_bracket or use_target_dot_bracket, not both")
            dot_bracket = request.get("target_dot_bracket")
            if dot_bracket is None:
                raise ValueError("Explicit target conditioning requires request target_dot_bracket")
        if sum(value is not None for value in (paths, single, dot_bracket)) != 1:
            raise ValueError("RNAbpFlow requires exactly one explicit conditioning source")
        if dot_bracket is not None:
            if not isinstance(dot_bracket, str):
                raise ValueError("RNAbpFlow conditioning dot bracket must be a string")
            maps = [_dot_bracket_map(dot_bracket, length)]
            metadata = {"dot_bracket": dot_bracket, "from_target": use_target}
        else:
            if single is not None:
                paths = [single]
            if not isinstance(paths, (list, tuple)) or not 1 <= len(paths) <= 3:
                raise ValueError("RNAbpFlow map_paths must be an ordered list of one to three files")
            resolved = [_existing_path(path, "conditioning map") for path in paths]
            maps = [np.load(path, allow_pickle=False) for path in resolved]
            metadata = {"map_paths": [str(path) for path in resolved]}
        for matrix in maps:
            if matrix.shape != (length, length) or not np.isfinite(matrix).all():
                raise ValueError("RNAbpFlow conditioning maps must be finite L x L arrays")
        # The upstream dataset repeats one map, or maps two inputs to [0, 1, 0].
        channels = (0, 0, 0) if len(maps) == 1 else (0, 1, 0) if len(maps) == 2 else (0, 1, 2)
        metadata["channel_indices"] = list(channels)
        return np.stack([maps[index] for index in channels], axis=-1), metadata

    def prepare(self, request: dict) -> _Prepared:
        sequence = request["sequence"]
        chain_id = request.get("chain_id", "A")
        if not isinstance(sequence, str) or not sequence or set(sequence) - set("ACGU"):
            raise ValueError("RNAbpFlow accepts one nonempty RNA sequence containing only ACGU")
        if not isinstance(chain_id, str) or len(chain_id) != 1 or chain_id not in string.ascii_letters + string.digits:
            raise ValueError("RNAbpFlow requires one PDB-compatible chain_id")
        ss, conditioning = self._conditioning(request, len(sequence))
        aatype = self.transforms.sequence_to_tensor(sequence).cpu()
        onehot = self.transforms.get_one_hot(sequence)
        mapfeat = {
            "aatype": aatype,
            "atom_map": self.transforms.genMap(aatype).cpu(),
            "onehot": onehot,
        }
        if "seed" in request:
            base_seed = int(request["seed"])
            seeds = tuple(base_seed + index for index in range(self.sample_count))
        else:
            seeds = tuple(42 * (index + 1) * 10 for index in range(self.sample_count))
        if any(seed < 0 or seed >= 2**32 for seed in seeds):
            raise ValueError("RNAbpFlow sample seeds must fit NumPy's unsigned 32-bit seed range")
        output_dir = Path(request["output_dir"]).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        return _Prepared(sequence, chain_id, output_dir, seeds, ss, onehot, mapfeat, conditioning)

    def execute(self, prepared: _Prepared) -> dict:
        import numpy as np
        import torch

        ss = torch.as_tensor(prepared.ss, dtype=torch.float32, device=self.device).unsqueeze(0)
        onehot = torch.as_tensor(prepared.onehot, dtype=torch.float32, device=self.device).unsqueeze(0)
        mapfeat = {
            key: torch.as_tensor(value, device=self.device).unsqueeze(0)
            for key, value in prepared.mapfeat.items()
        }
        samples = []
        with torch.no_grad():
            for seed in prepared.seeds:
                sample = self.flow.interpolant.sample(
                    1, len(prepared.sequence), ss, self.flow.model, seed, onehot,
                    mapfeat, self.timesteps,
                )
                if isinstance(sample, torch.Tensor):
                    sample = sample.detach().cpu().numpy()
                samples.append(np.asarray(sample).copy())
        return {
            "prepared": prepared, "samples": samples,
            "aatype": prepared.mapfeat["aatype"].detach().cpu().numpy(),
        }

    def finalize(self, host_payload: dict) -> dict:
        import numpy as np

        prepared = host_payload["prepared"]
        aatype = host_payload["aatype"]
        features = {
            "aatype": aatype[None, :],
            "residue_index": np.arange(len(prepared.sequence))[None, :],
        }
        # genMap and the native atom23-to-atom28 gather use this same atom_types order.
        atom_mask = self.constants.STANDARD_ATOM_MASK[aatype][None, :]
        paths = []
        for index, coordinates in enumerate(host_payload["samples"]):
            results = {"final_atom_positions": coordinates, "final_atom_mask": atom_mask}
            structure = self.nucleicacid.from_prediction(features, results)
            pdb_text = self.nucleicacid.to_pdb(structure)
            if prepared.chain_id != "A":
                pdb_text = "\n".join(
                    line[:21] + prepared.chain_id + line[22:]
                    if line.startswith(("ATOM  ", "HETATM", "TER   ")) else line
                    for line in pdb_text.split("\n")
                )
            path = prepared.output_dir / f"Sample_{index}.pdb"
            path.write_text(pdb_text)
            paths.append(str(path))
        return {
            "structure_path": paths[0], "plddt": None,
            "artifacts": {"sample_paths": paths},
            "metadata": {
                "upstream_url": UPSTREAM_URL, "upstream_revision": UPSTREAM_REVISION,
                "weights_path": str(self.weights), "conditioning": prepared.conditioning,
                "sample_seeds": list(prepared.seeds), "samples_per_sequence": len(paths),
                "num_timesteps": self.timesteps, "selected_sample": 0,
                "selection_method": "first_sample_no_ranking",
            },
        }

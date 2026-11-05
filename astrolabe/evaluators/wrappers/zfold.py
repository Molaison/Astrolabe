"""Native Zfold/structRFM geometry followed by CPU-only Rosetta folding.

``repo_path`` is the structRFM checkout. ``weights_path`` names a Zfold
checkpoint or its directory, and ``lm_weights_path`` names the structRFM
checkpoint directory. Default ``msa_mode='file'`` requires a sequence-matching
``msa_path``. Explicit ``msa_mode='single_sequence'`` builds a fresh query-only
alignment for each request. It has no homologs and is not rMSA-equivalent.
Both modes require ``ss_path``. Secondary-structure formats are dot_bracket,
ct, spot_prob, and npy. Geometry settings retain the upstream crop rules.
The request seed controls Python, NumPy and torch sampling during execute.
Native fold.py has no seed option, so CPU folding remains uncontrolled.
"""

from __future__ import annotations

import importlib
import json
import os
import random
import string
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


UPSTREAM_REVISION = "f0c1add19f8dfc786f16bc4abb7aa775dbadddbf"
UPSTREAM_URL = "https://github.com/heqin-zhu/structRFM"

# The crop/reduction procedure below follows upstream predict.py.
# MIT License. Copyright (c) 2025 Heqin Zhu.
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


@dataclass(frozen=True)
class _Prepared:
    sequence: str
    chain_id: str
    output_dir: Path
    seed: int
    msa: Any
    ss: Any
    inputs: dict[str, str]
    msa_mode: str = "file"


def _existing_path(value: Any, label: str) -> Path:
    if not value:
        raise ValueError(f"Zfold requires explicit {label}")
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Zfold {label} does not exist: {path}")
    return path


def _import_upstream(name: str, root: Path):
    module = importlib.import_module(name)
    if not Path(module.__file__).resolve().is_relative_to(root):
        raise ImportError(f"Zfold import {name} resolves outside {root}")
    return module


class ZfoldBackend:
    """The caller owns scheduling and calls place only inside its device stage."""

    def __init__(self, settings: dict):
        self.settings = dict(settings)
        self.repo = _existing_path(settings.get("repo_path"), "repo_path")
        self.source = _existing_path(
            settings.get("zfold_path", self.repo / "tasks/Zfold/src/Zfold"),
            "zfold_path",
        )
        weights = _existing_path(settings.get("weights_path"), "weights_path")
        weights_dir = weights if weights.is_dir() else weights.parent
        if weights.is_dir():
            candidates = sorted(p for p in weights.iterdir() if p.suffix in {".pt", ".pth"})
            if not candidates:
                raise FileNotFoundError(f"No Zfold checkpoint in {weights}")
            weights = candidates[-1]
        self.weights = weights
        config_path = _existing_path(
            settings.get("config_path", weights_dir / "config.json"), "config_path"
        )
        lm_weights = _existing_path(
            settings.get("lm_weights_path", settings.get("structrfm_weights_path")),
            "lm_weights_path",
        )
        self.fold_script = _existing_path(
            settings.get("fold_script_path", self.source / "fold.py"), "fold_script_path"
        )
        self.feature_type = settings.get("stru_feat_type", "LM")
        if self.feature_type not in {"LM", "both"}:
            raise ValueError("Zfold requires structRFM features, stru_feat_type must be LM or both")
        self.window = int(settings.get("window", 100))
        if not 50 <= self.window <= 256:
            raise ValueError("Zfold window must be 50..256 for native shift=50 and 512-base LM crops")
        self.nrows = int(settings.get("nrows", 200))
        self.nmodels = int(settings.get("nmodels", 5))
        self.cpu = int(settings.get("cpu", 5))
        if min(self.nrows, self.nmodels, self.cpu) < 1:
            raise ValueError("Zfold nrows, nmodels and cpu must be positive")

        for root in (self.repo / "src", self.source, self.source / "network"):
            sys.path.insert(0, str(root))
        import torch

        self.misc = _import_upstream("util.misc", self.source)
        predictor = _import_upstream("network.RNAformer", self.source)
        self.geometry_config = _import_upstream("network.config", self.source)
        lm_api = _import_upstream("structRFM.infer", self.repo)
        config = json.loads(config_path.read_text())
        checkpoint = torch.load(str(weights), map_location="cpu", weights_only=False)
        with torch.device("cpu"):
            self.model = predictor.DistPredictor(
                dim_2d=config["channels"], layers_2d=config["n_blocks"],
                stru_feat_type=self.feature_type,
            )
            self.lm = lm_api.structRFM_infer(
                from_pretrained=str(lm_weights), max_length=514, device="cpu"
            )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.lm_frozen = bool(settings.get("lm_frozen", "freeze" in weights_dir.name))
        if not self.lm_frozen:
            self.lm.model.load_state_dict(checkpoint["LM_state_dict"])
        self.model.eval()
        self.lm.model.eval()
        self.device = torch.device("cpu")

    def place(self, device: str):
        import torch

        self.device = torch.device(device)
        self.model.to(self.device)
        # Pinned model_forward places tokens on model.device, not a wrapper device.
        self.lm.get_model().to(self.device)

    def prepare(self, request: dict) -> _Prepared:
        import numpy as np

        sequence = request["sequence"]
        chain_id = request.get("chain_id", "A")
        if not isinstance(sequence, str) or not sequence or set(sequence) - set("ACGU"):
            raise ValueError("Zfold accepts one nonempty RNA sequence containing only ACGU")
        if not isinstance(chain_id, str) or len(chain_id) != 1 or chain_id not in string.ascii_letters + string.digits:
            raise ValueError("Zfold requires one PDB-compatible chain_id")
        seed = int(request.get("seed", 0))
        if not 0 <= seed < 2**32:
            raise ValueError("Zfold seed must fit NumPy's unsigned 32-bit seed range")
        msa_mode = request.get("msa_mode", self.settings.get("msa_mode", "file"))
        if msa_mode not in {"file", "single_sequence"}:
            raise ValueError("Zfold msa_mode must be file or single_sequence")
        ss_path = _existing_path(
            request.get("ss_path", self.settings.get("ss_path", self.settings.get("ss_file"))),
            "ss_path",
        )
        inputs = {"ss_path": str(ss_path)}
        if msa_mode == "single_sequence":
            msa = np.array([["AUCG".index(nt) for nt in sequence]], dtype=np.uint8)
        else:
            msa_path = _existing_path(
                request.get("msa_path", self.settings.get("msa_path")), "msa_path"
            )
            inputs["msa_path"] = str(msa_path)
            records = []
            for line in msa_path.read_text().splitlines():
                if line.startswith(">"):
                    records.append([])
                elif line.strip():
                    if not records:
                        raise ValueError("Zfold MSA must contain FASTA headers")
                    records[-1].append(line.strip())
            if not records or any(len(lines) != 1 for lines in records):
                raise ValueError("Zfold native A3M parser requires one alignment row per FASTA record")
            query = records[0][0].translate(str.maketrans("", "", string.ascii_lowercase)).replace("T", "U")
            if query.replace("-", "") != sequence:
                raise ValueError("Zfold MSA query does not match the requested RNA sequence")
            msa = self.misc.parse_a3m(str(msa_path), limit=20000)
        if msa.ndim != 2 or msa.shape[1] != len(sequence):
            raise ValueError("Zfold MSA columns do not match the requested RNA sequence")
        fmt = request.get("ss_fmt", self.settings.get("ss_fmt", "dot_bracket"))
        if fmt == "dot_bracket":
            ss = self.misc.ss2mat(ss_path.read_text().rstrip().splitlines()[-1].strip())
        elif fmt == "ct":
            ss = self.misc.parse_ct(str(ss_path), length=len(sequence))
        elif fmt == "spot_prob":
            ss = np.loadtxt(ss_path)
            ss = ss + ss.T
        elif fmt == "npy":
            ss = np.load(ss_path, allow_pickle=False)
        else:
            raise ValueError(f"Unsupported Zfold ss_fmt: {fmt}")
        if ss.shape != (len(sequence), len(sequence)) or not np.isfinite(ss).all():
            raise ValueError("Zfold secondary-structure map must be a finite L x L array")
        output_dir = Path(request["output_dir"]).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        return _Prepared(
            sequence, chain_id, output_dir, seed, msa, ss, inputs, msa_mode,
        )

    def _structural_feature(self, sequence, ss):
        import torch

        features = self.lm.extract_feature(sequence)
        if self.settings.get("use_attn_feat", False):
            matrix = features["last_mean_attn_feat"]
        elif self.settings.get("use_outer_product_mean", False):
            embedding = features["seq_feat"]
            shape = embedding.shape
            matrix = (
                embedding.unsqueeze(-3).unsqueeze(-2)
                * embedding.unsqueeze(-2).unsqueeze(-1)
            ).reshape(*shape[:-1], shape[-2], -1).mean(dim=-1)
        else:
            matrix = features["mat_feat"]
        matrix = matrix.unsqueeze(-1)
        return torch.cat([ss, matrix], dim=-1) if self.feature_type == "both" else matrix

    def execute(self, prepared: _Prepared) -> dict:
        import numpy as np
        import torch

        random.seed(prepared.seed)
        np.random.seed(prepared.seed)
        torch.manual_seed(prepared.seed)
        with torch.no_grad():
            msa = torch.from_numpy(prepared.msa).to(self.device)
            ss = torch.from_numpy(prepared.ss).to(self.device).unsqueeze(-1)
            length = len(prepared.sequence)
            residue_ids = torch.arange(length, device=self.device).view(1, length)
            if length <= 2 * self.window:
                feature = self._structural_feature(prepared.sequence, ss)
                geometry = self.model(
                    msa.unsqueeze(0), feature.unsqueeze(0), res_id=residue_ids,
                    msa_cutoff=self.nrows,
                )["geoms"]
            else:
                geometry = {
                    "contact": torch.zeros((length, length), device=self.device),
                    "distance": {
                        atom: torch.zeros(
                            (length, length, self.geometry_config.n_bins["2D"]["distance"]),
                            device=self.device,
                        )
                        for atom in self.geometry_config.obj["2D"]["distance"]
                    },
                }
                counts = torch.zeros((length, length), device=self.device)
                starts = np.arange(0, length - self.window + 50, 50)
                for i, first in enumerate(starts):
                    for second in starts[i:]:
                        selected = np.zeros(length, dtype=bool)
                        selected[first:min(first + self.window, length)] = True
                        selected[second:min(second + self.window, length)] = True
                        cropped_msa = msa[:, selected]
                        keep = (cropped_msa == 4).sum(dim=-1) < 0.7 * selected.sum()
                        cropped_seq = "".join(nt for nt, flag in zip(prepared.sequence, selected) if flag)
                        feature = self._structural_feature(cropped_seq, ss[selected][:, selected])
                        crop = self.model(
                            cropped_msa[keep].unsqueeze(0), feature.unsqueeze(0),
                            res_id=residue_ids[:, selected], msa_cutoff=self.nrows,
                        )["geoms"]
                        indices = torch.as_tensor(np.flatnonzero(selected), device=self.device)
                        pair_indices = (indices[:, None], indices[None, :])
                        counts[pair_indices] += 1
                        geometry["contact"][pair_indices] += crop["contact"]
                        for atom in geometry["distance"]:
                            geometry["distance"][atom][pair_indices] += crop["distance"][atom]
                geometry["contact"] /= counts
                for atom in geometry["distance"]:
                    geometry["distance"][atom] /= counts[:, :, None]
            host_geometry = {
                key: ({atom: tensor.detach().cpu().numpy() for atom, tensor in value.items()}
                      if isinstance(value, dict) else value.detach().cpu().numpy())
                for key, value in geometry.items()
            }
        return {"prepared": prepared, "geometry": host_geometry}

    def finalize(self, host_payload: dict) -> dict:
        import numpy as np

        prepared = host_payload["prepared"]
        geometry_path = prepared.output_dir / "geometry.npz"
        np.savez_compressed(geometry_path, **host_payload["geometry"])
        fasta = prepared.output_dir / "sequence.fasta"
        fasta.write_text(f">query\n{prepared.sequence}\n")
        structure = prepared.output_dir / "prediction.pdb"
        fold_dir = prepared.output_dir / "rosetta"
        fold_dir.mkdir(exist_ok=True)
        command = [
            str(self.settings.get("fold_python_bin", sys.executable)), str(self.fold_script),
            "-npz", str(geometry_path), "-fa", str(fasta), "-out", str(structure),
            "-nm", str(self.nmodels), "-tmp", str(fold_dir), "--CPU", str(self.cpu),
        ]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        log = prepared.output_dir / "rosetta.log"
        with log.open("w") as stream:
            subprocess.run(command, cwd=self.source, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        if not structure.is_file() or not structure.stat().st_size:
            raise RuntimeError(f"Zfold Rosetta folding produced no structure; see {log}")
        lines = structure.read_text().splitlines(keepends=True)
        structure.write_text("".join(
            line[:21] + prepared.chain_id + line[22:]
            if line.startswith(("ATOM  ", "HETATM", "TER   ")) else line
            for line in lines
        ))
        return {
            "structure_path": str(structure), "plddt": None,
            "artifacts": {"geometry_path": str(geometry_path), "fold_log": str(log)},
            "metadata": {
                "upstream_url": UPSTREAM_URL, "upstream_revision": UPSTREAM_REVISION,
                "weights_path": str(self.weights), "conditioning": prepared.inputs,
                "seed": prepared.seed, "model_seed_controlled": True,
                "fold_seed_controlled": False,
                "fold_seed_limitation": "Pinned upstream fold.py exposes no seed option",
                "nmodels": self.nmodels, "window": self.window, "shift": 50,
                "msa_mode": prepared.msa_mode,
                "msa_method": (
                    "query_only_no_homologs_not_rmsa_equivalent"
                    if prepared.msa_mode == "single_sequence"
                    else "provided_alignment_query_matched"
                ),
                "msa_rows": len(prepared.msa), "msa_cutoff": self.nrows,
                "stru_feat_type": self.feature_type, "lm_frozen": self.lm_frozen,
            },
        }

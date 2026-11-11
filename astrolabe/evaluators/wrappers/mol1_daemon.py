"""Pinned Molence API inside Astrolabe's existing native kernel lifecycle.

The admitted operation is native composite inference (loader, model, terminal
geometry correction, dump and cleanup), NOT a GPU kernel timing claim.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from astrolabe.evaluators.mol1_contract import BASE_MODEL_NAME, active_constraints, local_assets, output_samples


class Mol1Handler:
    def __init__(self, settings):
        from astrolabe.model_execution import _verify_asset_hashes

        self.settings = settings
        if int(os.environ.get("WORLD_SIZE", "1")) != 1 or int(os.environ.get("LOCAL_RANK", "0")) != 0:
            raise ValueError("Mol-1 supports one local native device, not distributed execution")
        _verify_asset_hashes(settings["asset_sha256"])
        _, references = local_assets(settings, model_name=settings["model_name"])
        import mol
        from mol.runner import batch_inference as batch
        from mol.runner.inference import infer_predict

        source = Path(settings["repo_path"]).expanduser().resolve() / "mol"
        if not Path(mol.__file__).resolve().is_relative_to(source) or not Path(batch.__file__).resolve().is_relative_to(source):
            raise ValueError("Mol-1 imported unexpected source")
        self.batch = batch
        self.infer_predict = infer_predict
        with tempfile.TemporaryDirectory(prefix="astrolabe-mol1-load-") as bootstrap:
            original = batch.download_inference_cache

            def verify(config):
                if config.model_name != settings["model_name"]:
                    raise ValueError("Mol-1 native model identity changed")
                if Path(config.load_checkpoint_path).resolve() != Path(settings["load_checkpoint_path"]).expanduser().resolve():
                    raise ValueError("Mol-1 native checkpoint identity changed")
                if any(str(Path(config.data[key]).resolve()) != value for key, value in references.items()):
                    raise ValueError("Mol-1 native reference identity changed")
                if not config.load_strict:
                    raise ValueError("Mol-1 requires strict native checkpoint loading")

            batch.download_inference_cache = verify
            try:
                self.runner = batch.get_default_runner(
                    model_name=settings["model_name"], load_checkpoint_path=settings["load_checkpoint_path"],
                    dump_dir=bootstrap, seeds=None, use_seeds_in_json=True,
                    n_cycle=settings["num_cycles"], n_step=settings["num_steps"], n_sample=settings["num_samples"],
                    dtype=settings["dtype"], device=settings["device"], use_msa=settings["use_msa"],
                    use_template=False, use_rna_msa=False, need_atom_confidence=True,
                    trimul_kernel=settings["triangle_multiplicative"], triatt_kernel=settings["triangle_attention"],
                    enable_fusion=settings["enable_fusion"], enable_tf32=settings["enable_tf32"],
                    memory_settings={**{f"data.{key}": value for key, value in references.items()},
                                     "num_workers": settings["num_workers"],
                                     "deterministic": settings["deterministic"],
                                     "load_strict": True,
                                     # Keep sample IDs, not confidence ranking, as the output order.
                                     "sorted_by_ranking_score": False},
                )
            finally:
                batch.download_inference_cache = original

    def run(self, request):
        input_json, out = Path(request["input_json"]), Path(request["out_dir"])
        jobs = json.loads(input_json.read_text())
        if not isinstance(jobs, list) or len(jobs) != 1:
            raise ValueError("Mol-1 native request requires exactly one job")
        job = jobs[0]
        if self.settings["model_name"] == BASE_MODEL_NAME and active_constraints(job.get("constraints")):
            raise ValueError("Mol-1 base must never receive active constraints")
        runner = self.runner
        runner.configs.dump_dir = runner.dump_dir = str(out)
        runner.error_dir = str(out / "ERR")
        runner.init_dumper(need_atom_confidence=True, sorted_by_ranking_score=False)
        original_dump = runner.dumper.dump

        def dump(**kwargs):
            self._dump_identity(kwargs, job)
            return original_dump(**kwargs)

        runner.dumper.dump = dump
        try:
            # Supplied MSA files have already been checked by the adapter; no search fallback.
            from mol.runner.msa_search import need_msa_search

            if self.settings["use_msa"] and need_msa_search(job):
                raise ValueError("Mol-1 supplied MSA is missing; automatic search is not supported")
            runner.configs.input_json_path = self.batch.preprocess_input(
                str(input_json), out_dir=str(out), use_msa=self.settings["use_msa"],
                use_template=False, use_rna_msa=False,
            )
            processed = json.loads(Path(runner.configs.input_json_path).read_text())
            if processed != jobs:
                raise ValueError("Mol-1 preprocessing changed input identities, MSA or constraints")
            self.infer_predict(runner, runner.configs)
            samples = output_samples(out, job, self.settings["num_samples"])
            # Parse every native result before publishing a successful inference receipt.
            from astrolabe.evaluators.mol1 import collect_sample
            for _, _, summary, full, cif in samples:
                collect_sample(job, summary, full, cif)
            return {"output_dir": str(out), "job": job["name"], "modelSeeds": job["modelSeeds"],
                    "model_name": self.settings["model_name"]}
        finally:
            runner.dumper.dump = original_dump

    def _dump_identity(self, values, job):
        import numpy as np

        if values["pdb_id"] != job["name"] or values["seed"] not in job["modelSeeds"]:
            raise ValueError("Mol-1 native dump belongs to another job/seed")
        full = values["pred_dict"]["full_data"]
        if len(full) != self.settings["num_samples"]:
            raise ValueError("Mol-1 native sample count differs from request")
        atoms = values["atom_array"]
        first = full[0]
        atom_token = first["atom_to_token_idx"].detach().cpu().numpy()
        token_asym = first["token_asym_id"].detach().cpu().numpy()
        if len(atom_token) != len(atoms):
            raise ValueError("Mol-1 atom/token axis differs from native atoms")
        tokens = []
        native_atoms = [[str(a), int(b), str(c)] for a, b, c in
                        zip(atoms.chain_id, atoms.label_seq_id, atoms.atom_name)]
        for token in range(len(token_asym)):
            ids = {(native_atoms[i][0], native_atoms[i][1]) for i in np.flatnonzero(atom_token == token)}
            if len(ids) != 1:
                raise ValueError("Mol-1 protein token does not map to exactly one chain/residue")
            cid, position = ids.pop()
            tokens.append([cid, position, int(token_asym[token])])
        for sample in full:
            if (not np.array_equal(sample["atom_to_token_idx"].detach().cpu().numpy(), atom_token)
                    or not np.array_equal(sample["token_asym_id"].detach().cpu().numpy(), token_asym)):
                raise ValueError("Mol-1 samples have different native identity axes")
        directory = Path(self.runner.dump_dir) / job["name"] / f"seed_{values['seed']}" / "predictions"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "native_identity.json"
        # Repeated native job/seed processing must never overwrite evidence.
        with path.open("x") as handle:
            json.dump({"job": job["name"], "seed": values["seed"], "samples": len(full),
                       "tokens": tokens, "atoms": native_atoms, "atom_to_token_idx": atom_token.tolist()}, handle)


def create_handler(settings):
    from astrolabe.evaluators.runtime_stages import configured_device_hint
    from astrolabe.model_execution import KernelStages, kernel_handler

    hint = configured_device_hint(settings)
    # This port does not qualify inference concurrency, even if a caller supplies
    # a profile. Use the existing kernel's conservative admission policy.
    if isinstance(settings.get("_runtime_profile"), dict):
        settings = dict(settings, _runtime_profile=dict(settings["_runtime_profile"], concurrency_certified=False))

    def load(raw_settings):
        handler = Mol1Handler(raw_settings)
        device = str(handler.runner.device)
        return KernelStages(lambda value: value, handler.run,
                            resource="cpu" if device == "cpu" else "gpu"), {
            "backend": "mol1", "device": device, "model_name": raw_settings["model_name"],
            "checkpoint_path": raw_settings["load_checkpoint_path"],
            "source_commit": raw_settings.get("source_commit"),
            "residency": "resident_model", "inference_concurrency": "fixed1",
            "staging": "native_composite_not_kernel_time",
        }

    return kernel_handler(settings, load, torch_device=hint or "auto", lifecycle={
        "backend": "mol1", "device": hint, "residency": "resident_model",
        "weights_loaded": True, "concurrency_certified": False,
    })

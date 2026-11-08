"""Native Protenix transfer, complete forward, and output materialization.

Preserves the forward_prepared/finalize_prediction contracts from
Astrolabe-bioemu2 commit 1bdaf3a53227c9197b930dc00a01b752acd90067,
without importing its resident worker or lifecycle runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, TypedDict

import torch

from astrolabe.inference_pipeline import Pending
from astrolabe.model_execution import EvaluatorStages


class ProtenixRequest(TypedDict):
    configs: Any
    data: Mapping[str, Any]
    atom_array: Any
    dataset_name: str
    pdb_id: str
    seed: int
    entity_poly_type: Mapping[str, str]


@dataclass(frozen=True)
class _PreparedPrediction:
    request: ProtenixRequest
    data: Mapping[str, Any]


@dataclass(frozen=True)
class _Prediction:
    prepared: _PreparedPrediction
    prediction: dict[str, Any]


def evaluator_stages(runner) -> EvaluatorStages:
    """Use inside the serial scientific owner, including config mutation."""
    resource = "cpu" if torch.device(runner.device).type == "cpu" else "gpu"

    def pending(value, *, keepalive):
        if resource == "cpu":
            return value
        with torch.cuda.device(runner.device):
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(runner.device))
        return Pending(value, event.synchronize, (runner, keepalive, event))

    def prepare(request: ProtenixRequest):
        operation = request.get("operation", "predict_and_dump")
        if operation != "predict_and_dump":
            raise ValueError(f"Unknown Protenix operation {operation!r}")
        configs = request["configs"]
        if isinstance(configs, dict):
            from omegaconf import OmegaConf

            configs = OmegaConf.create(configs)
        dump_dir = getattr(configs, "dump_dir", None)
        if dump_dir is not None:
            runner.dump_dir = str(dump_dir)
            runner.error_dir = str(Path(dump_dir) / "ERR")
            Path(runner.error_dir).mkdir(parents=True, exist_ok=True)
            runner.dumper.base_dir = str(dump_dir)
        runner.update_model_configs(configs)
        prepared = _PreparedPrediction(request, runner.prepare_data(request["data"]))
        return pending(prepared, keepalive=prepared)

    def execute(prepared: _PreparedPrediction):
        output = _Prediction(prepared, runner.predict_prepared(prepared.data))
        return pending(output, keepalive=output)

    def finalize(output: _Prediction):
        request = output.prepared.request
        runner.dumper.dump(
            dataset_name=request["dataset_name"],
            pdb_id=request["pdb_id"],
            seed=request["seed"],
            pred_dict=output.prediction,
            atom_array=request["atom_array"],
            entity_poly_type=request["entity_poly_type"],
        )
        return pending(output.prediction, keepalive=output)

    return EvaluatorStages(
        prepare, execute, finalize, resource=resource,
        prepare_resource=resource, finalize_resource=resource,
    )

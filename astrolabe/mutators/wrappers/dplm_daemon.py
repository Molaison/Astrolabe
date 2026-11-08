#!/usr/bin/env python3
"""Long-lived DPLM inference daemon (JSONL stdin/stdout).

Protocol (one JSON object per line):
  - {"type":"infer","id":"<uuid>","seed":123,"tasks":[{"name":"X","masked_sequence":"...","num_samples":1}]}
    -> {"type":"result","id":"<uuid>","results":[{"name":"X","samples":["..."]}]}
  - {"type":"shutdown"} -> {"type":"ok","message":"bye"}
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from typing import Any, Dict, List, Optional

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from astrolabe.mutators.wrappers.dplm_infer import _resolve_model_name
from astrolabe.model_execution import KernelStages, host_result
from astrolabe.mutators.runtime_stages import mutator_kernel_handler, protocol_infer, device_resource


def _ensure_byprot_on_path() -> None:
    """Make `byprot` importable without requiring an installed package."""
    repo_path = pathlib.Path.cwd()
    src_path = repo_path / "src"
    if src_path.exists():
        src_str = str(src_path)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DPLM JSONL daemon.")
    p.add_argument("--model-name", default="airkingbd/dplm_150m")
    p.add_argument("--device", default="auto")
    p.add_argument("--sampling-strategy", default="gumbel_argmax")
    p.add_argument("--max-iter", type=int, default=500)
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--temperature", type=float, default=1.0)
    return p.parse_args()


def _select_device(requested: str) -> str:
    import torch

    if requested in ("auto", None):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def _seed_everything(seed: Optional[int]) -> None:
    if seed is None:
        return
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _clean_sequence(seq: str) -> str:
    return "".join(str(seq).split())


def _run_generation(
    *,
    model,
    tokenizer,
    masked_sequence: str,
    num_samples: int,
    max_iter: int,
    sampling_strategy: str,
    temperature: float,
    device: str,
) -> List[str]:
    import torch

    if num_samples <= 0:
        return []

    batch_sequences = [masked_sequence] * num_samples
    encoded = tokenizer.batch_encode_plus(
        batch_sequences,
        add_special_tokens=True,
        padding=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    partial_masks = input_ids.ne(model.mask_id)

    with torch.no_grad():
        outputs = model.generate(
            input_tokens=input_ids,
            tokenizer=tokenizer,
            max_iter=max_iter,
            sampling_strategy=sampling_strategy,
            temperature=temperature,
            partial_masks=partial_masks,
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return [_clean_sequence(seq) for seq in decoded]


def _write(obj: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def create_handler(settings):
    def load(config):
        _ensure_byprot_on_path()
        from byprot.models.dplm.dplm import DiffusionProteinLanguageModel
        device = _select_device(config.get("device", "auto"))
        _seed_everything(random.randint(0, 1_000_000))
        model = DiffusionProteinLanguageModel.from_pretrained(
            _resolve_model_name(config.get("model_name", "airkingbd/dplm_150m"))
        ).to(device)
        model.eval()
        tokenizer = model.tokenizer
        def prepare(message):
            tasks = []
            for task in message.get("tasks", []) or []:
                name, sequence = task.get("name"), task.get("masked_sequence")
                if name is None or sequence is None:
                    continue
                count = int(task.get("num_samples", config.get("num_samples", 1)) or 0)
                tokens = None
                if count > 0:
                    tokens = tokenizer.batch_encode_plus(
                        [str(sequence)] * count, add_special_tokens=True,
                        padding=True, return_tensors="pt",
                    )["input_ids"]
                tasks.append((name, tokens))
            return message, tasks
        def execute(prepared):
            import torch
            message, tasks = prepared
            results = []
            for name, tokens in tasks:
                if tokens is None:
                    results.append((name, None))
                    continue
                input_ids = tokens.to(device)
                partial_masks = input_ids.ne(model.mask_id)
                with torch.no_grad():
                    outputs = model.generate(
                        input_tokens=input_ids, tokenizer=tokenizer,
                        max_iter=int(message.get("max_iter", config.get("max_iter", 500))),
                        sampling_strategy=str(message.get("sampling_strategy", config.get("sampling_strategy", "gumbel_argmax"))),
                        temperature=float(message.get("temperature", config.get("temperature", 1.0))),
                        partial_masks=partial_masks,
                    )
                results.append((name, host_result(outputs)))
            return results
        def finalize(results):
            return [{"name": name, "samples": [] if outputs is None else [
                _clean_sequence(seq) for seq in tokenizer.batch_decode(outputs, skip_special_tokens=True)
            ]} for name, outputs in results]
        return KernelStages(prepare, execute, finalize, resource=device_resource(device)), {
            "device": device, "model_name": config.get("model_name", "airkingbd/dplm_150m"),
            "weights_loaded": True,
        }
    return mutator_kernel_handler(settings, load, backend="dplm",
                                  torch_device=_select_device(settings.get("device", "auto")))


def main() -> int:
    args = _parse_args()
    handler = create_handler(vars(args))

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            _write({"type": "error", "message": f"invalid json: {exc}"})
            continue

        mtype = msg.get("type")
        if mtype == "shutdown":
            _write({"type": "ok", "message": "bye"})
            return 0
        if mtype != "infer":
            _write({"type": "error", "message": f"unknown type: {mtype}"})
            continue

        req_id = msg.get("id")
        _seed_everything(msg.get("seed"))

        results = protocol_infer(handler, msg)
        metadata = handler({"operation": "describe"})["metadata"]
        _write({"type": "result", "id": req_id, "results": results,
                "metadata": metadata, "runtime": handler.last_runtime})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

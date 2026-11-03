"""Private external-environment process hosting one native NA-MPNN model."""
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import sys
import traceback


def main():
    output = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    spec = json.loads(Path(sys.argv[1]).read_text())
    native_path = Path(spec["native_path"]).resolve(strict=True)
    sys.path.insert(0, str(native_path.parent))
    from na_mpnn_native import NativeSession, SOURCE_SHA256
    model = NativeSession(native_path, spec["argv"])
    import torch
    device = str(next(model.model.parameters()).device)

    def send(value):
        output.write(json.dumps(value, allow_nan=False) + "\n")
        output.flush()

    send({"type": "ready", "pid": os.getpid(), "model_load_count": model.load_count,
          "load_count": model.load_count, "weights_loaded": True,
          "gpu_resident": device.startswith("cuda"), "device": device,
          "source_sha256": SOURCE_SHA256})
    try:
        while True:
            raw = sys.stdin.buffer.readline(16 * 1024**2 + 1)
            if not raw:
                break
            if len(raw) > 16 * 1024**2 or not raw.endswith(b"\n"):
                raise ValueError("NA-MPNN command exceeds JSONL byte budget")
            request = json.loads(raw)
            request_id = request.get("id")
            if request.get("type") == "shutdown":
                send({"type": "shutdown", "id": request_id})
                break
            try:
                if request.get("type") != "infer" or not isinstance(request_id, str):
                    raise ValueError("invalid NA-MPNN request")
                model.execute(request["argv"])
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                send({"type": "result", "id": request_id, "model_load_count": model.load_count})
            except BaseException as exc:
                traceback.print_exc(file=sys.stderr)
                send({"type": "error", "id": request_id, "error": str(exc)})
                # Native failure may leave opaque CUDA/model state. No reload
                # or replay is permitted inside this scientific request.
                break
    finally:
        model.close()
        output.close()


if __name__ == "__main__":
    main()

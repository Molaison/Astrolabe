"""Verified native ESMFold encoding, complete-model and reduction stages."""
from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap


_NATIVE_INFER_BODY = "9f490a34182f59fe5b3df5b36b689fdc3ab21dd1b005ec4df4d58dde12ef0438"


def native_encoding_boundary(model):
    import torch

    method = inspect.unwrap(model.infer)
    try:
        function = ast.parse(textwrap.dedent(inspect.getsource(method))).body[0]
        body = function.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            body = body[1:]
        digest = hashlib.sha256(ast.dump(
            ast.Module(body=body, type_ignores=[]), include_attributes=False,
        ).encode()).hexdigest()
        signature = inspect.signature(model.infer)
        namespace = method.__globals__
    except (OSError, TypeError, AttributeError, SyntaxError, ValueError):
        return None
    if digest != _NATIVE_INFER_BODY:
        return None
    if hasattr(torch, "get_default_device") and str(torch.get_default_device()) != "cpu":
        return None
    encode = namespace.get("batch_encode_sequences")
    if not callable(encode):
        return None
    defaults = {key: parameter.default for key, parameter in signature.parameters.items()}
    # The facade has always called infer(sequences) without overriding these.
    if defaults.get("residx") is not None or defaults.get("masking_pattern") is not None:
        return None

    def prepare(request):
        operation = request.get("operation")
        if (request.get("return_residue_plddt") or request.get("return_pdb")) and operation != "infer":
            raise ValueError("Residue confidence reduction/PDB output requires an infer request")
        if operation == "output_to_pdb":
            return request, request["output"]
        if operation != "infer":
            raise ValueError(f"Unknown ESMFold operation {operation!r}")
        sequences = request["sequences"]
        if isinstance(sequences, str):
            sequences = [sequences]
        with torch.no_grad():
            encoded = encode(sequences, defaults["residue_index_offset"], defaults["chain_linker"])
            aatype, mask, residx, linker_mask, chain_index = encoded
            aatype, mask, residx, linker_mask = map(
                lambda x: x.to(model.device), (aatype, mask, residx, linker_mask)
            )
        return request, (aatype, mask, residx, linker_mask, chain_index)

    def execute(prepared):
        from astrolabe.inference_pipeline import Pending

        request, value = prepared
        if request["operation"] == "output_to_pdb":
            return prepared
        aatype, mask, residx, linker_mask, chain_index = value
        with torch.no_grad():
            output = model.forward(
                aatype, mask=mask, residx=residx,
                masking_pattern=defaults["masking_pattern"],
                num_recycles=defaults["num_recycles"],
            )
            raw = request, output, linker_mask, chain_index
            if torch.device(model.device).type == "cuda":
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(model.device))
                return Pending(raw, event.synchronize, keepalive=(prepared, event))
        return raw

    def finalize(raw):
        from astrolabe.model_execution import host_result

        if raw[0]["operation"] == "output_to_pdb":
            return model.output_to_pdb(raw[1])
        request, output, linker_mask, chain_index = raw
        with torch.no_grad():
            output["atom37_atom_exists"] = output["atom37_atom_exists"] * linker_mask.unsqueeze(2)
            output["mean_plddt"] = (output["plddt"] * output["atom37_atom_exists"]).sum(
                dim=(1, 2)
            ) / output["atom37_atom_exists"].sum(dim=(1, 2))
            output["chain_index"] = chain_index
            # 8a0dfcc's device boundary matters even without its host-only
            # formatter. Native atom14-to-atom37 must precede materialization.
            pdbs = model.output_to_pdb(output) if request.get("return_pdb") else None
            if request.get("return_residue_plddt"):
                residue_plddt = [
                    output["plddt"][i, output["chain_index"][i] != -1].mean(-1)
                    for i in range(len(output["chain_index"]))
                ]
                output = {"output": output, "residue_plddt": residue_plddt}
            elif request.get("return_pdb"):
                output = {"output": output}
            if request.get("return_pdb"):
                output["pdbs"] = pdbs
            return host_result(output)

    return prepare, execute, finalize


def evaluator_stages(model, *, device=None):
    from astrolabe.evaluators.runtime_stages import operation_resource
    from astrolabe.model_execution import EvaluatorStages

    boundary = native_encoding_boundary(model)
    if boundary is None:
        raise ValueError("Unsupported ESMFold infer body; cannot establish complete-model stages")
    resource = "cpu" if str(device or model.device) == "cpu" else "gpu"
    return EvaluatorStages(
        *boundary,
        resource=operation_resource({"output_to_pdb"}, device=device or model.device),
        prepare_resource=resource, finalize_resource=resource,
    )

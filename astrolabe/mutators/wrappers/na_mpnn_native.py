"""Version-gated resident boundary for the native NA-MPNN inference program.

Upstream 835974f3 exposes only ``main(args)`` and reconstructs ProteinMPNN on
every call. We split its parsed function at ``model.eval()`` without rewriting
the remaining sampling/selection/output program or modifying the vendor tree.
For an explicit, fixed nonzero native seed, restoring the state observed just
after construction preserves the RNG draws consumed by that constructor.
Unseeded native CLI calls and unreviewed source versions fail closed here.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
from pathlib import Path
import random
import sys


SOURCE_SHA256 = "8d8937cceb84fbfc2e1b49b65e21b730a74e9dafab9e9c6720d1af75bda403d7"
SOURCE_COMMIT = "835974f3d420de18b3eb1dc66f25fbd17c53cc09"


def _is_call(node, value, attr):
    return (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == value and node.value.func.attr == attr)


def _functions(source, filename):
    tree = ast.parse(source, filename=filename)
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    boundary = next(i for i, node in enumerate(main.body) if _is_call(node, "model", "eval"))
    initializer = copy.deepcopy(main)
    initializer.name = "_initialize"
    initializer.body = initializer.body[:boundary + 1] + [ast.Return(value=ast.Name(id="model", ctx=ast.Load()))]
    request = copy.deepcopy(main)
    request.name = "_execute"
    replaced = []
    replacements = set()
    for i, node in enumerate(request.body):
        if i > boundary:
            replaced.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "checkpoint" for t in node.targets):
            replacements.add("checkpoint")
            replaced.append(ast.Pass())
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "model" for t in node.targets):
            replacements.add("model")
            node.value = ast.Name(id="_resident_model", ctx=ast.Load())
            replaced.append(node)
        elif _is_call(node, "model", "load_state_dict") or _is_call(node, "model", "to"):
            replacements.add(node.value.func.attr)
            replaced.append(ast.Pass())
        elif _is_call(node, "model", "eval"):
            replacements.add("eval")
            replaced.append(ast.Expr(value=ast.Call(func=ast.Name(id="_restore_post_init", ctx=ast.Load()), args=[], keywords=[])))
        else:
            replaced.append(node)
    if replacements != {"checkpoint", "model", "load_state_dict", "to", "eval"}:
        raise ValueError("NA-MPNN native initialization boundary changed")
    request.body = replaced
    guard = next(node for node in tree.body if isinstance(node, ast.If)
                 and isinstance(node.test, ast.Compare)
                 and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__")
    parser_statements = []
    for node in guard.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "args" for t in node.targets):
            break
        parser_statements.append(copy.deepcopy(node))
    namespace = {"argparse": argparse, "sys": sys}
    ast.fix_missing_locations(initializer)
    ast.fix_missing_locations(request)
    compiled = ast.Module(body=[initializer, request] + parser_statements, type_ignores=[])
    exec(compile(ast.fix_missing_locations(compiled), filename, "exec"), namespace)
    return namespace


class NativeSession:
    def __init__(self, native_path, argv):
        path = Path(native_path).resolve(strict=True)
        source = path.read_bytes()
        if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
            raise ValueError("NA-MPNN resident mode only supports reviewed source " + SOURCE_COMMIT)
        self.namespace = _functions(source.decode(), str(path))
        self.args = self.parse(argv)
        if not self.args.seed:
            raise ValueError("NA-MPNN resident mode requires explicit nonzero extra_args.seed; native seed=0 draws fresh subprocess entropy")
        self.signature = self._signature(self.args)
        self.load_count = 0
        self.model = self.namespace["_initialize"](self.args)
        self.load_count += 1
        import numpy as np
        import torch
        self._post_init = {
            "python": random.getstate(), "numpy": np.random.get_state(),
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
        }
        self.namespace["_resident_model"] = self.model
        self.namespace["_restore_post_init"] = self._restore

    def parse(self, argv):
        args = self.namespace["argparser"].parse_args(argv)
        if args.mode != "design" or args.model_type != "na_mpnn":
            raise ValueError("NA-MPNN mutator resident mode requires native design/na_mpnn")
        if args.catch_failed_inferences:
            raise ValueError("resident NA-MPNN must report native exceptions, not swallow failed requests")
        if args.checkpoint_na_mpnn is None or args.batch_size is None or args.temperature is None:
            raise ValueError("resident NA-MPNN requires explicit native checkpoint, batch size, and temperature")
        return args

    @staticmethod
    def _signature(args):
        # These fields affect construction or its RNG state. Native per-request
        # masks, ties, sampling count and temperature remain request arguments.
        return (str(Path(args.checkpoint_na_mpnn).resolve()), args.model_type,
                args.na_shared_tokens, args.k_neighbors, args.seed)

    def _restore(self):
        import numpy as np
        import torch
        random.setstate(self._post_init["python"])
        np.random.set_state(self._post_init["numpy"])
        torch.set_rng_state(self._post_init["cpu"])
        if self._post_init["cuda"] is not None:
            torch.cuda.set_rng_state_all(self._post_init["cuda"])

    def execute(self, argv):
        args = self.parse(argv)
        if self._signature(args) != self.signature:
            raise ValueError("NA-MPNN request changed resident initialization identity")
        self.namespace["_execute"](args)

    def close(self):
        import torch
        model, self.model = self.model, None
        self.namespace.pop("_resident_model", None)
        if model is not None:
            model.cpu()
        del model
        self._post_init.clear()
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

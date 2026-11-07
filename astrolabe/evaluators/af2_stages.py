"""Complete-model stages for the inspected native AlphaFold2 RunModel."""
from __future__ import annotations

import ast
import copy
import inspect
import textwrap


_NATIVE_PREDICT = '''
def predict(self, feat, random_seed):
    self.init_params(feat)
    logging.info(
        'Running predict with shape(feat) = %s',
        tree.map(lambda x: x.shape, feat),
    )
    result = self.apply(self.params, jax.random.PRNGKey(random_seed), feat)
    tree.map(lambda x: x.block_until_ready(), result)
    result.update(
        get_confidence_metrics(result, multimer_mode=self.multimer_mode)
    )
    logging.info('Output shape was %s', tree.map(lambda x: x.shape, result))
    return result
'''


def _body(source):
    body = ast.parse(textwrap.dedent(source)).body[0].body
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    return ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)


def evaluator_stages(runner, *, device_hint=None):
    from astrolabe.evaluators.runtime_stages import jax_device_scope, operation_resource
    from astrolabe.inference_pipeline import Pending
    from astrolabe.model_execution import EvaluatorStages, host_result

    method = inspect.unwrap(runner.predict)
    try:
        supported = _body(inspect.getsource(method)) == _body(_NATIVE_PREDICT)
        namespace = method.__globals__
    except (OSError, TypeError, AttributeError, SyntaxError, ValueError):
        supported = False
    if not supported:
        raise ValueError("Unsupported AF2 predict body; cannot establish complete-model stages")
    jax = namespace["jax"]
    tree = namespace["tree"]
    logging = namespace["logging"]
    confidence = namespace["get_confidence_metrics"]
    # Reuse the audited resident worker's parameter placement, not its broker.
    # Lazy initialization still belongs to the original serial RNG owner.
    if runner.params:
        with jax_device_scope(device_hint):
            runner.params = jax.device_put(runner.params)
            tree.map(lambda x: x.block_until_ready(), runner.params)

    def prepare(request):
        operation = request.get("operation")
        if operation == "process_features":
            return request, runner.process_features(
                copy.deepcopy(request["features"]), random_seed=request["random_seed"],
            )
        if operation == "pair_and_merge":
            from alphafold.data import feature_processing

            with jax_device_scope(device_hint):
                return request, feature_processing.pair_and_merge(
                    all_chain_features=request["all_chain_features"],
                )
        if operation != "predict":
            raise ValueError(f"Unknown AlphaFold2 operation {operation!r}")
        features = request["features"]
        with jax_device_scope(device_hint):
            initialized = bool(runner.params)
            runner.init_params(features)
            if not initialized:
                runner.params = jax.device_put(runner.params)
            logging.info('Running predict with shape(feat) = %s',
                         tree.map(lambda x: x.shape, features))
            key = jax.random.PRNGKey(request["random_seed"])
            features = jax.device_put(features)
            params = runner.params
        prepared = request, (params, features, key)
        return Pending(prepared,
                       lambda: tree.map(lambda x: x.block_until_ready(), (params, features, key)),
                       keepalive=request)

    def execute(prepared):
        request, value = prepared
        if request["operation"] != "predict":
            return prepared
        params, features, key = value
        with jax_device_scope(device_hint):
            result = runner.apply(params, key, features)
        return Pending((request, result),
                       lambda: tree.map(lambda x: x.block_until_ready(), result),
                       keepalive=prepared)

    def finalize(raw):
        request, result = raw
        with jax_device_scope(device_hint):
            if request["operation"] == "predict":
                result.update(confidence(result, multimer_mode=runner.multimer_mode))
                logging.info('Output shape was %s', tree.map(lambda x: x.shape, result))
            return host_result(result)

    resource = "cpu" if device_hint == "cpu" else "gpu"
    return EvaluatorStages(
        prepare, execute, finalize,
        resource=operation_resource({"process_features", "pair_and_merge"}, device=device_hint),
        prepare_resource=resource, finalize_resource=resource,
    )


def _staged_kernel(runner):
    from astrolabe.evaluators.runtime_stages import DirectKernel

    kernel = getattr(runner, "_astrolabe_staged_kernel", None)
    if kernel is None:
        device_hint = getattr(runner, "_astrolabe_device_hint", None)
        kernel = DirectKernel(
            {}, lambda _: (evaluator_stages(runner, device_hint=device_hint),
                           {"multimer_mode": runner.multimer_mode}),
            lifecycle={"backend": "alphafold2", "residency": "resident_model",
                       "device": device_hint, "weights_loaded": bool(runner.params),
                       "resource": "cpu" if device_hint == "cpu" else "gpu"},
        )
        runner._astrolabe_staged_kernel = kernel
    return kernel


def staged_predict(runner, features, *, random_seed):
    if getattr(runner, "_client", None) is not None:
        return runner.predict(features, random_seed=random_seed)
    return _staged_kernel(runner)(
        {"operation": "predict", "features": features, "random_seed": random_seed}
    )


def staged_pair_and_merge(runner, *, all_chain_features):
    if getattr(runner, "_client", None) is not None:
        return runner.pair_and_merge(all_chain_features=all_chain_features)
    return _staged_kernel(runner)(
        {"operation": "pair_and_merge", "all_chain_features": all_chain_features}
    )

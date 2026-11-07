from __future__ import annotations

import csv
import dataclasses
import functools
import itertools
import logging
import os
import pathlib
import string
import threading
import time
import typing
import uuid
from collections import Counter, defaultdict
from typing import Sequence, Callable

import haiku as hk
import jax
import numpy as np
from jax import numpy as jnp

from alphafold3.common import folding_input
from alphafold3.jax.attention import attention
from alphafold3.model import model, params, features, post_processing
from alphafold3.model.components import utils

BUCKETS = (
    256,
    512,
    768,
    1024,
    1280,
    1536,
    2048,
    2560,
    3072,
    3584,
    4096,
    4608,
    5120,
)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ResultsForSeed:
    """Stores the inference results (diffusion samples) for a single seed.

    Attributes:
      seed: The seed used to generate the samples.
      inference_results: The inference results, one per sample.
      full_fold_input: The fold input that must also include the results of
        running the data pipeline - MSA and templates.
      embeddings: The final trunk single and pair embeddings, if requested.
    """

    seed: int
    inference_results: Sequence[model.InferenceResult]
    full_fold_input: folding_input.Input
    embeddings: dict[str, np.ndarray] | None = None


class ModelRunner:
    """Helper class to run structure prediction stages."""

    def __init__(
        self,
        config: model.Model.Config,
        device: jax.Device,
        model_dir: pathlib.Path,
    ):
        self._model_config = config
        self._device = device
        self._model_dir = model_dir
        self._resident_executables = None
        self._compile_lock = threading.Lock()
        self._launch_lock = threading.Lock()

    def enable_resident_execution(self):
        """Called once at readiness; weights and loaded executables stay here."""
        if self._resident_executables is None:
            self._resident_executables = {}

    def set_cyclic_asym_ids(self, cyclic_asym_ids: Sequence[int] | None) -> None:
        ids = tuple(int(x) for x in (cyclic_asym_ids or ()))
        current = tuple(
            int(x)
            for x in getattr(self._model_config.global_config, "cyclic_asym_ids", ())
        )
        if current == ids:
            return
        if hasattr(self, "_inference_owner_settings"):
            raise ValueError("Staged AF3 requires the immutable owner cyclic configuration")
        self._model_config.global_config.cyclic_asym_ids = ids
        self.__dict__.pop("_model", None)

    def set_cyclic_relative_patches(
        self, cyclic_relative_patches: Sequence[Sequence[int]] | None
    ) -> None:
        patches = tuple(
            tuple(int(y) for y in x) for x in (cyclic_relative_patches or ())
        )
        current = tuple(
            tuple(int(y) for y in x)
            for x in getattr(
                self._model_config.global_config, "cyclic_relative_patches", ()
            )
        )
        if current == patches:
            return
        if hasattr(self, "_inference_owner_settings"):
            raise ValueError("Staged AF3 requires the immutable owner cyclic configuration")
        self._model_config.global_config.cyclic_relative_patches = patches
        self.__dict__.pop("_model", None)

    @functools.cached_property
    def model_params(self) -> hk.Params:
        """Loads model parameters from the model directory."""
        return params.get_model_haiku_params(model_dir=self._model_dir)

    @functools.cached_property
    def _model(
        self,
    ) -> Callable[[jnp.ndarray, features.BatchDict], model.ModelResult]:
        """Loads model parameters and returns a jitted model forward pass."""

        @hk.transform
        def forward_fn(batch):
            return model.Model(self._model_config)(batch)

        # Place weights during owner readiness. Passing host NumPy parameters
        # into the JIT would otherwise hide their H2D transfer inside infer.
        device_params = jax.device_put(self.model_params, self._device)
        jax.block_until_ready(device_params)
        return functools.partial(
            jax.jit(forward_fn.apply, device=self._device), device_params
        )

    def run_inference(
        self, featurised_example: features.BatchDict, rng_key: jnp.ndarray
    ) -> model.ModelResult:
        """Synchronous compatibility entrypoint to the shared evaluator stages."""
        return self._inference_kernel(self._inference_request(
            featurised_example, rng_key=rng_key))

    @functools.cached_property
    def _inference_kernel(self):
        from astrolabe.evaluators.af3_pipeline import evaluator_stages
        from astrolabe.evaluators.runtime_stages import DirectKernel

        def loader(settings):
            # Readiness, not per-request inference. Shape-dependent first JIT
            # compilation still occurs at the native model call.
            self.model_params
            self._model
            return evaluator_stages(self, allow_extraction=True), {
                "backend": "alphafold3", "device": str(self._device),
                "owner_configuration": "immutable_cyclic",
                "cold_shape_jit": "inside_native_model_call",
            }

        cpu = self._device.platform == "cpu"
        device = "cpu" if cpu else f"cuda:{self._device.id}"
        return DirectKernel({}, loader, lifecycle={
            "backend": "alphafold3", "device": device,
            "resource": "cpu" if cpu else "gpu", "residency": "resident_model",
            "weights_loaded": True, "concurrency_certified": False,
        })

    def _inference_request(self, batch, **fields):
        owner = self._model_config.global_config
        return dict(model="alphafold3", features=batch,
                    cyclic_asym_ids=tuple(getattr(owner, "cyclic_asym_ids", ())),
                    cyclic_relative_patches=tuple(getattr(owner, "cyclic_relative_patches", ())),
                    **fields)

    def run_inference_and_extract_for_seed(self, batch, seed, target_name):
        return self._inference_kernel(self._inference_request(
            batch, rng_seed=int(seed), target_name=target_name))

    def prepare_features(self, featurised_example: features.BatchDict):
        """CPU feature filtering, before any JAX device conversion."""
        return utils.remove_invalidly_typed_feats(featurised_example)

    def upload_features(self, cleaned_features):
        """Preserve the original JAX conversions and chosen device placement."""
        return jax.device_put(
            jax.tree_util.tree_map(jnp.asarray, cleaned_features),
            self._device,
        )

    def run_model(self, device_features, rng_key):
        """Dispatch the cached JIT model; returned arrays may still be in flight."""
        if self._resident_executables is not None:
            from astrolabe.evaluators.inference_service import mark
            leaves, tree = jax.tree_util.tree_flatten((rng_key, device_features))
            signature = (tree, tuple((tuple(x.shape), str(x.dtype), bool(getattr(x, "weak_type", False)))
                                     for x in leaves))
            # One compilation/load for a new shape. Requests use the same
            # in-memory executable thereafter, with private inputs/RNG/outputs.
            with self._compile_lock:
                executable = self._resident_executables.get(signature)
                if executable is None:
                    started = time.monotonic()
                    compiled_model = self._model
                    executable = compiled_model.func.lower(
                        *compiled_model.args, rng_key, device_features).compile()
                    self._resident_executables[signature] = executable
                    mark("resident_executable_loaded", elapsed_s=time.monotonic() - started,
                         executable_count=len(self._resident_executables),
                         executable_id=id(executable))
            # This PJRT backend submits all work for a device to one compute
            # stream. Concurrent host RunAsync calls can interleave the mutable
            # thunk/control-flow state of an executable; more Python launchers
            # do not create independent GPU streams. Keep host submission
            # ordered while retaining asynchronous device completion and
            # request-local upload/materialization outside this boundary.
            waiting = time.monotonic()
            with self._launch_lock:
                started = time.monotonic()
                mark("resident_executable_dispatch", executable_id=id(executable),
                     launch_wait_s=started - waiting, thread_id=threading.get_ident())
                result = executable(*self._model.args, rng_key, device_features)
                mark("resident_executable_enqueued", executable_id=id(executable),
                     host_dispatch_s=time.monotonic() - started,
                     thread_id=threading.get_ident())
                return result
        return self._model(rng_key, device_features)

    def materialize_result(self, result) -> model.ModelResult:
        """Host arrays with the native bfloat16 conversion and model identifier."""
        result = jax.tree.map(np.asarray, result)
        result = jax.tree.map(
            lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x,
            result,
        )
        result = dict(result)
        identifier = self.model_params["__meta__"]["__identifier__"].tobytes()
        result["__identifier__"] = identifier
        return result

    def inference_stages(self, rng_key, *, target_name=None):
        """CPU -> transfer -> GPU -> CPU, optionally followed by extraction.

        Submit the original featurised batch to these stages. The executor must
        supply ``cpu``, ``transfer`` and a single-thread ``gpu`` resource. Keep
        model configuration fixed until its accepted requests have drained.
        The JIT model itself remains one explicit device stage.
        """
        from astrolabe.inference_backends import jax_pending
        from astrolabe.inference_pipeline import Stage

        def prepare(batch):
            return dict(batch=batch, cleaned=self.prepare_features(batch))

        def upload(item):
            item['device_features'] = self.upload_features(item.pop('cleaned'))
            item['rng_key'] = jax.device_put(rng_key, self._device)
            return jax_pending(item, wait_for=(item['device_features'], item['rng_key']),
                               keepalive=(item['batch'], rng_key))

        def dispatch(item):
            item['device_result'] = self.run_model(item['device_features'], item['rng_key'])
            return jax_pending(item, wait_for=item['device_result'],
                               keepalive=(item['device_features'], item['rng_key']))

        def materialize(item):
            result = self.materialize_result(item.pop('device_result'))
            if target_name is None:
                return result
            return item['batch'], result

        stages = [Stage('af3_features', 'cpu', prepare),
                  Stage('af3_upload', 'transfer', upload),
                  Stage('af3_model', 'gpu', dispatch),
                  Stage('af3_materialize', 'cpu', materialize)]
        if target_name is not None:
            stages.append(Stage('af3_extract', 'cpu', lambda pair: self.extract_structures(
                pair[0], pair[1], target_name)))
        return tuple(stages)

    def extract_structures(
        self,
        batch: features.BatchDict,
        result: model.ModelResult,
        target_name: str,
    ) -> list[model.InferenceResult]:
        """Generates structures from model outputs."""
        return list(
            model.Model.get_inference_result(
                batch=batch, result=result, target_name=target_name
            )
        )

    def extract_embeddings(
        self,
        result: model.ModelResult,
    ) -> dict[str, np.ndarray] | None:
        """Extracts embeddings from model outputs."""
        embeddings = {}
        if "single_embeddings" in result:
            embeddings["single_embeddings"] = result["single_embeddings"]
        if "pair_embeddings" in result:
            embeddings["pair_embeddings"] = result["pair_embeddings"]
        return embeddings or None


def service_model_settings(config) -> dict:
    """Model-defining options which must agree with the shared owner."""
    return {
        "model_dir": str(config.model_dir),
        "flash_attention_implementation": str(config.flash_attention_implementation),
        "num_samples": int(config.num_samples),
        "num_recycles": int(config.num_recycles),
        "save_embeddings": bool(config.save_embeddings),
    }


class RemoteModelRunner:
    """Inference-only proxy; extraction and scoring remain in each CPU worker."""

    extract_structures = ModelRunner.extract_structures
    extract_embeddings = ModelRunner.extract_embeddings

    def __init__(self, config):
        self._socket = str(config.service_socket)
        self._opaque_rpc = bool(getattr(config, "service_opaque_rpc", False))
        self._transport = str(getattr(config, "service_transport", "pickle"))
        if self._transport not in {"pickle", "buffers"}:
            raise ValueError("AF3 service_transport must be pickle or buffers")
        self._settings = service_model_settings(config)
        self._cyclic_asym_ids = tuple(int(x) for x in getattr(config, "cyclic_asym_ids", ()))
        self._cyclic_relative_patches = tuple(tuple(int(y) for y in x) for x in
                                             getattr(config, "cyclic_relative_patches", ()))

    def set_cyclic_asym_ids(self, cyclic_asym_ids):
        self._cyclic_asym_ids = tuple(int(x) for x in (cyclic_asym_ids or ()))

    def set_cyclic_relative_patches(self, cyclic_relative_patches):
        self._cyclic_relative_patches = tuple(
            tuple(int(y) for y in x) for x in (cyclic_relative_patches or ())
        )

    def run_inference(self, featurised_example, rng_key):
        from astrolabe.evaluators.inference_service import call

        if self._transport == "buffers":
            from astrolabe.evaluators.af3_buffers import infer
            return infer(self._socket, self._buffer_request(), {
                "features": utils.remove_invalidly_typed_feats(featurised_example),
                "rng_key": np.asarray(rng_key),
            })
        request = {
            "model": "alphafold3",
            "identity": {
                "trajectory": os.environ.get("PH_TRAJECTORY_ID"),
                "pid": os.getpid(),
                "request_uuid": uuid.uuid4().hex,
            },
            "settings": self._settings,
            "features": jax.tree.map(np.asarray, featurised_example),
            "rng_key": np.asarray(rng_key),
            "cyclic_asym_ids": self._cyclic_asym_ids,
            "cyclic_relative_patches": self._cyclic_relative_patches,
        }
        return call(self._socket, request, opaque=True) if self._opaque_rpc else call(self._socket, request)

    def _buffer_request(self):
        return {
            "model": "alphafold3", "settings": dict(self._settings),
            "identity": {"trajectory": os.environ.get("PH_TRAJECTORY_ID"),
                         "pid": os.getpid(), "request_uuid": uuid.uuid4().hex},
            "cyclic_asym_ids": self._cyclic_asym_ids,
            "cyclic_relative_patches": self._cyclic_relative_patches,
        }

    def run_inference_and_extract_for_seed(self, batch, seed, target_name):
        from astrolabe.evaluators.inference_service import call

        if self._transport == "buffers":
            from astrolabe.evaluators.af3_buffers import infer
            from astrolabe.evaluators.inference_service import mark
            import time
            request = dict(self._buffer_request(), rng_seed=int(seed))
            result = infer(self._socket, request, {
                "features": utils.remove_invalidly_typed_feats(batch),
            })
            # Full native batch never leaves this CPU consumer. Same extractor,
            # same result, same downstream write/score/selection ordering.
            started = time.monotonic()
            extracted = self.extract_structures(batch, result, target_name)
            mark("buffer_local_extracted", elapsed_s=time.monotonic() - started,
                 identity=request["identity"])
            return extracted
        request = {
            "model": "alphafold3", "settings": dict(self._settings),
            "features": jax.tree.map(np.asarray, batch), "rng_seed": int(seed),
            "target_name": target_name,
            "identity": {"trajectory": os.environ.get("PH_TRAJECTORY_ID"),
                         "pid": os.getpid(), "request_uuid": uuid.uuid4().hex},
            "cyclic_asym_ids": self._cyclic_asym_ids,
            "cyclic_relative_patches": self._cyclic_relative_patches,
        }
        return call(self._socket, request, opaque=True) if self._opaque_rpc else call(self._socket, request)


def make_model_config(
    *,
    flash_attention_implementation: attention.Implementation = "triton",
    num_diffusion_samples: int = 5,
    num_recycles: int = 10,
    return_embeddings: bool = False,
) -> model.Model.Config:
    """Returns a model config with some defaults overridden."""
    config = model.Model.Config()
    config.global_config.flash_attention_implementation = flash_attention_implementation
    config.global_config.cyclic_asym_ids = ()
    config.global_config.cyclic_relative_patches = ()
    config.heads.diffusion.eval.num_samples = num_diffusion_samples
    config.num_recycles = num_recycles
    config.return_embeddings = return_embeddings
    # On some consumer GPUs (e.g. RTX 3090) Triton/Pallas GLU kernels can hit
    # shared-memory limits at runtime. Disabling the fused kernel preserves
    # model math while avoiding the kernel selection.
    try:
        device_kind = getattr(jax.devices("gpu")[0], "device_kind", "")
    except Exception:
        device_kind = ""

    if ("RTX" in device_kind) or ("GeForce" in device_kind):
        seen: set[int] = set()
        stack: list[object] = [config]
        while stack:
            obj = stack.pop()
            obj_id = id(obj)
            if obj_id in seen:
                continue
            seen.add(obj_id)

            if hasattr(obj, "use_glu_kernel"):
                try:
                    setattr(obj, "use_glu_kernel", False)
                except Exception:
                    pass

            if isinstance(obj, (str, bytes, int, float, bool, type(None))):
                continue
            if isinstance(obj, dict):
                stack.extend(obj.values())
                continue
            if isinstance(obj, (list, tuple, set)):
                stack.extend(list(obj))
                continue
            if dataclasses.is_dataclass(obj):
                for f in dataclasses.fields(obj):
                    stack.append(getattr(obj, f.name))
                continue
            try:
                stack.extend(vars(obj).values())
            except Exception:
                continue
    return config


def _pairwise_distances(coords: np.ndarray) -> np.ndarray:
    """Compute Euclidean pairwise distances without sklearn.

    Args:
      coords: (N, 3) coordinate array.
    """
    coords = np.asarray(coords, dtype=np.float32)
    diffs = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diffs * diffs, axis=-1))


def transform_pae_matrix(pae_matrix, pae_cutoff):
    # Initialize the transformed matrix with zeros
    transformed_pae = np.zeros_like(pae_matrix)

    # Apply transformation: pae = 0 -> score = 1, pae = cutoff -> score = 0, above cutoff -> score = 0
    # Linearly scale values between 0 and cutoff to fall between 1 and 0
    within_cutoff = pae_matrix < pae_cutoff
    transformed_pae[within_cutoff] = 1 - (pae_matrix[within_cutoff] / pae_cutoff)

    return transformed_pae


def aggregate_matrix(matrix, labels):
    matrix = np.asarray(matrix)
    labels = np.asarray(labels)
    unique_labels, inverse = np.unique(labels, return_inverse=True)
    K = len(unique_labels)
    flat_groups = inverse[:, None] * K + inverse[None, :]
    sums = np.bincount(
        flat_groups.ravel(),
        weights=matrix.ravel(),
        minlength=K * K,
    ).reshape(K, K)
    counts = np.bincount(flat_groups.ravel(), minlength=K * K).reshape(K, K)
    return np.divide(sums, counts, out=np.zeros_like(sums), where=counts != 0)


def get_LIS_scores(infer_rslt, distance_threshold=8.0, pae_cutoff=12.0):
    pae_matrix = infer_rslt.numerical_data["full_pae"]
    token_res_ids = infer_rslt.metadata["token_res_ids"]
    token_chain_ids = infer_rslt.metadata["token_chain_ids"]
    predicted_structure = infer_rslt.predicted_structure
    # Calculate the chain breaks
    chain_residue_counts = Counter(token_chain_ids)
    subunit_number = list(chain_residue_counts.values())

    # Transform the PAE matrix, setting values above the cutoff to zero, making sure lengths match (nres, nres)
    transformed_pae_matrix = transform_pae_matrix(pae_matrix, pae_cutoff)
    transformed_pae_matrix = np.nan_to_num(transformed_pae_matrix)
    token_labels = [
        f"{chain_id}_{res_id}"
        for chain_id, res_id in zip(token_chain_ids, token_res_ids)
    ]
    transformed_pae_matrix = aggregate_matrix(transformed_pae_matrix, token_labels)

    # Calculate the contact map, making sure lengths match (nres, nres).
    #
    # NOTE: `aggregate_matrix(contact_map, atom_labels)` is O(N_atoms^2) in Python
    # and can be extremely slow for typical AF3 outputs. Instead, compute residue
    # contact existence directly via a KD-tree on atom coordinates, then build a
    # residue-level contact map aligned to `token_labels` / `transformed_pae_matrix`.
    #
    # For downstream use we only need `contact_map != 0` as a mask.
    from scipy.spatial import cKDTree  # type: ignore

    unique_labels = np.unique(token_labels)
    label_map = {val: i for i, val in enumerate(unique_labels)}
    n_res = len(unique_labels)

    atom_labels = np.array(
        [
            f"{chain_id}_{res_id}"
            for chain_id, res_id in zip(
                predicted_structure.chain_id, predicted_structure.res_auth_seq_id
            )
        ],
        dtype=object,
    )
    atom_groups = np.array([label_map.get(lbl, -1) for lbl in atom_labels], dtype=np.int32)
    valid = atom_groups >= 0
    coords = np.asarray(predicted_structure.coords, dtype=np.float32)[valid]
    atom_groups = atom_groups[valid]

    contact_map = np.zeros((n_res, n_res), dtype=np.uint8)
    if coords.shape[0]:
        tree = cKDTree(coords)
        pairs = tree.query_pairs(r=distance_threshold, output_type="ndarray")
        if pairs.size:
            gi = atom_groups[pairs[:, 0]]
            gj = atom_groups[pairs[:, 1]]
            contact_map[gi, gj] = 1
            contact_map[gj, gi] = 1
    np.fill_diagonal(contact_map, 1)

    # Calculate the mean Local Interaction Score(LIS) for the PAE matrix and the contact map
    mean_lis = calculate_mean_lis(transformed_pae_matrix, subunit_number)

    combined_map = np.where(
        (transformed_pae_matrix > 0) & (contact_map != 0), transformed_pae_matrix, 0
    )
    contact_lis = calculate_mean_lis(combined_map, subunit_number)
    return mean_lis, contact_lis


def calculate_mean_lis(transformed_pae, subunit_number):
    # Calculate the cumulative sum of protein lengths to get the end indices of the submatrices
    cum_lengths = np.cumsum(subunit_number)

    # Add a zero at the beginning of the cumulative lengths to get the start indices
    start_indices = np.concatenate(([0], cum_lengths[:-1]))

    # Initialize an empty matrix to store the mean LIS
    mean_lis_matrix = np.zeros((len(subunit_number), len(subunit_number)))

    # Iterate over the start and end indices
    for i in range(len(subunit_number)):
        for j in range(len(subunit_number)):
            # Get the start and end indices of the interaction submatrix
            start_i, end_i = start_indices[i], cum_lengths[i]
            start_j, end_j = start_indices[j], cum_lengths[j]

            # Get the interaction submatrix
            submatrix = transformed_pae[start_i:end_i, start_j:end_j]

            # Calculate the mean LIS, considering only non-zero values
            mean_lis = submatrix[submatrix > 0].mean() if submatrix.any() else 0

            # Store the mean LIS in the matrix
            mean_lis_matrix[i, j] = mean_lis

    return mean_lis_matrix


def get_af3_scores(all_inference_results):
    # only take the inference result of first seeds

    # to be moved into process_output
    all_inference_results = all_inference_results[0]
    result_scores = defaultdict(list)
    for infer_rslt in all_inference_results.inference_results:
        # iter scores by all_inference_results.inference_results[0].metadata['token_res_ids']
        metadata = infer_rslt.metadata
        predicted_structure = infer_rslt.predicted_structure
        # AlphaFold3 stores per-chain summary arrays under `iptm_ichain`/`iptm_xchain`
        # (see vendor `StructureConfidenceSummary.from_inference_result`). Expose them
        # using the more descriptive `chain_ptm`/`chain_iptm` keys used elsewhere in
        # this codebase (e.g. binder_ptm/binder_iptm losses).
        if "chain_ptm" not in metadata and "iptm_ichain" in metadata:
            metadata["chain_ptm"] = metadata["iptm_ichain"]
        if "chain_iptm" not in metadata and "iptm_xchain" in metadata:
            metadata["chain_iptm"] = metadata["iptm_xchain"]
        take_scores = lambda name, data: {
            key: value for key, value in data.items() if key in name
        }
        per_str_scores_indicators = (
            "ranking_score",
            "fraction_disordered",
            "predicted_tm_score",
            "predicted_distance_error",
            "interface_predicted_tm_score",
            "ptm",
            "iptm",
            "actifptm",
            "ptm_iptm_average",
            "ranking_confidence",
            "ranking_confidence_pae",
        )

        chain_pair_scores_indicators = (
            "chain_ptm",
            "chain_iptm",
            "chain_pair_pde_mean",
            "chain_pair_pde_min",
            "chain_pair_pae_min",
            "chain_pair_iptm",
            "chain_pair_lis",
            "chain_pair_contact_lis",
            "chain_pair_actifptm",
        )

        inter_o_cross_scores_indicators = (
            "intra_chain_single_pde",
            "cross_chain_single_pde",
            "pae_ichain",
            "pae_xchain",
            "iptm_ichain",
            "iptm_xchain",
        )
        chain_pair_lis, chain_pair_contact_lis = get_LIS_scores(infer_rslt)
        infer_rslt.metadata["chain_pair_lis"] = chain_pair_lis
        infer_rslt.metadata["chain_pair_contact_lis"] = chain_pair_contact_lis
        result_scores["per_str_scores"].append(
            take_scores(per_str_scores_indicators, metadata)
        )
        result_scores["chain_pair_scores"].append(
            take_scores(chain_pair_scores_indicators, metadata)
        )
        result_scores["inter_o_cross_scores"].append(
            take_scores(inter_o_cross_scores_indicators, metadata)
        )
        result_scores["per_res_scores"].append(
            {
                "per_res_plddt": extract_plddt(
                    predicted_structure.res_auth_seq_id,
                    predicted_structure.atom_b_factor,
                )
            }
        )

    fmt_result_scores = {}

    for score_name, score_ls in result_scores.items():
        fmt_result_scores[score_name] = {}
        for score in score_ls:
            for key, value in score.items():
                if key not in fmt_result_scores[score_name]:
                    fmt_result_scores[score_name][key] = []
                fmt_result_scores[score_name][key].append(value)

    return fmt_result_scores
    # input_chains = all_inference_results.full_fold_input.chains
    # chain_ids = np.array([chain.id for chain in input_chains])
    # chain_ids = np.array([chain_ids]).squeeze()
    # imask = chain_ids[:, None] == chain_ids[None, :]
    # xmask = ~(chain_ids[:, None] == chain_ids[None, :])


def process_chain_pair_scores(eval_result):
    chain_pair_scores = eval_result["chain_pair_scores"]

    intra_chain_scores = defaultdict(dict)
    cross_chain_scores = defaultdict(dict)

    for oligo_name, value in chain_pair_scores.items():
        for key, val_ls in value.items():
            intra_mask = np.diag(np.ones(len(val_ls[0]), dtype=bool))
            cross_mask = ~intra_mask

            intra_chain_scores[oligo_name][key.replace("chain_pair", "intra_chain")] = (
                np.mean([val[intra_mask] for val in val_ls])
            )
            cross_chain_scores[oligo_name][key.replace("chain_pair", "cross_chain")] = (
                np.mean([val[cross_mask] for val in val_ls])
            )


def extract_plddt(atom2res, atom2bfactor):
    per_res_plddt = {}
    i = 0
    for res, plddt in itertools.groupby(zip(atom2res, atom2bfactor), lambda x: x[0]):
        _, plddts = zip(*plddt)
        plddts = np.array(plddts).mean()
        per_res_plddt[i] = plddts
        i += 1
    return [per_res_plddt[i] for i in range(len(per_res_plddt))]


from astrolabe.af3_featurisation_settings import (
    _configure_refstructure_env,
    _configure_msa_env,
)


def alphafold3_setup(config):
    from astrolabe.evaluators.runtime_stages import configured_device_hint

    _configure_refstructure_env(config)
    _configure_msa_env(config)
    if getattr(config, "service_socket", None):
        return RemoteModelRunner(config)
    device_hint = configured_device_hint(config)
    try:
        devices = jax.local_devices(backend="cpu" if device_hint == "cpu" else "gpu")
    except Exception as exc:
        if device_hint is not None and device_hint != "cpu":
            raise RuntimeError(f"Configured AF3 device {device_hint} is unavailable") from exc
        logging.warning(
            "JAX GPU backend unavailable (%s); falling back to CPU for AlphaFold3.",
            exc,
        )
        devices = jax.local_devices(backend="cpu")
    device_index = int(device_hint.split(":", 1)[1]) if device_hint and device_hint.startswith("cuda:") else 0
    if device_index >= len(devices):
        raise ValueError(f"Configured AF3 device {device_hint} is not visible to JAX")
    cache_dir = getattr(config, "jax_compilation_cache_dir", None)
    if cache_dir:
        cache_dir_path = pathlib.Path(os.path.expanduser(str(cache_dir)))
        try:
            cache_dir_path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            fallback = pathlib.Path("/tmp/jax_cache")
            try:
                fallback.mkdir(parents=True, exist_ok=True)
            except OSError:
                logging.warning(
                    "Failed to create JAX cache dir %s and fallback %s; compilation cache disabled.",
                    cache_dir_path,
                    fallback,
                )
                cache_dir_path = None
            else:
                logging.warning(
                    "Failed to create JAX cache dir %s (%s); falling back to %s.",
                    cache_dir_path,
                    exc,
                    fallback,
                )
                cache_dir_path = fallback
        if cache_dir_path is not None:
            jax.config.update("jax_compilation_cache_dir", str(cache_dir_path))
    model_runner = ModelRunner(
        config=make_model_config(
            flash_attention_implementation=typing.cast(
                attention.Implementation, config.flash_attention_implementation
            ),
            num_diffusion_samples=config.num_samples,
            num_recycles=config.num_recycles,
            return_embeddings=config.save_embeddings,
        ),
        device=devices[device_index],
        model_dir=pathlib.Path(config.model_dir),
    )
    if getattr(config, "cyclic_asym_ids", ()):
        model_runner.set_cyclic_asym_ids(config.cyclic_asym_ids)
    if getattr(config, "cyclic_relative_patches", ()):
        model_runner.set_cyclic_relative_patches(config.cyclic_relative_patches)
    return model_runner


def sanitised_name(name) -> str:
    """Returns sanitised version of the name that can be used as a filename."""
    lower_spaceless_name = name.lower().replace(" ", "_")
    allowed_chars = set(string.ascii_lowercase + string.digits + "_-.")
    return "".join(l for l in lower_spaceless_name if l in allowed_chars)


def write_outputs(
    all_inference_results: Sequence[ResultsForSeed],
    output_dir: os.PathLike[str] | str,
    job_name: str,
    save_all_samples: bool = False,
) -> None:
    """Writes outputs to the specified output directory."""
    ranking_scores = []
    max_ranking_score = None
    max_ranking_result = None

    output_terms = ""

    os.makedirs(output_dir, exist_ok=True)
    for results_for_seed in all_inference_results:
        seed = results_for_seed.seed
        for sample_idx, result in enumerate(results_for_seed.inference_results):
            if save_all_samples:
                sample_dir = os.path.join(
                    output_dir, "all_samples", f"seed-{seed}_sample-{sample_idx}"
                )
                os.makedirs(sample_dir, exist_ok=True)
                post_processing.write_output(
                    inference_result=result,
                    output_dir=sample_dir,
                    name=f"{job_name}_seed-{seed}_sample-{sample_idx}",
                )
            ranking_score = float(result.metadata["ranking_score"])
            ranking_scores.append((seed, sample_idx, ranking_score))
            if max_ranking_score is None or ranking_score > max_ranking_score:
                max_ranking_score = ranking_score
                max_ranking_result = result

    if max_ranking_result is not None:  # True iff ranking_scores non-empty.
        post_processing.write_output(
            inference_result=max_ranking_result,
            output_dir=output_dir,
            # The output terms of use are the same for all seeds/samples.
            terms_of_use=output_terms,
            name=job_name,
        )
        # Save csv of ranking scores with seeds and sample indices, to allow easier
        # comparison of ranking scores across different runs.
        with open(os.path.join(output_dir, "ranking_scores.csv"), "wt") as f:
            writer = csv.writer(f)
            writer.writerow(["seed", "sample", "ranking_score"])
            writer.writerows(ranking_scores)

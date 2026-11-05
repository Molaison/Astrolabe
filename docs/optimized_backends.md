# Optimized backend selection

`astrolabe.optimized_backends` connects an existing Astrolabe backend to an
Anthropic optimization kit only when the manifest pins a compatible model.
`config/anthropic_kits.yaml` is the source of truth for kit names, advertised
modes, model roles, source revision, and startup variables. The helper has no
second list of available kits.

## API contract

```python
load_manifest(path=None) -> dict
kit_names(manifest=None) -> tuple[str, ...]
selected_kit_specs(manifest, names=None, software_ids=None)
verify_source_checkout(source_root, revision) -> Path
validate_manifest_source(manifest=None, *, source_root=None)
build_kit_selection(
    backend,
    mode="exact",
    model_variant=None,
    *,
    kit_root=None,
    manifest=None,
    verify_source=True,
) -> KitSelection
provider_environment(
    backend,
    mode="exact",
    model_variant=None,
    kit_root=None,
) -> dict[str, str]
activate_provider(selection) -> ActivationReport
reset_activation_state() -> None
```

`KitSelection` contains the validated `spec`, exact kit revision, requested
`mode`, provider model identity, child-startup `env`, and the provider API
record. `ActivationReport` repeats the kit, model identity, mode, and revision;
its `state` carries a provider's `applied`/`deferred` status. `active=True` is
possible only when the called library hook returns a mapping whose own `active`
value is the boolean `True`. A mode-resolution report alone does not activate a
backend. The helper also requires non-empty `levers_applied` evidence or an
explicit applied state. Missing evidence and states such as `armed`, `pending`,
`deferred`, `dry-run`, `failed`, and `mismatch` are refused. An Anthropic
`mode="off"` selection reports `state="upstream-stock"`; it is distinct from the
unconfigured native provider.

Call `provider_environment(...)` before starting a child interpreter. Call
`activate_provider(selection)` inside the process that owns the model, before
the provider's documented construction/import boundary. Repeated compatible
calls return the cached report and call the provider hook once.

`astrolabe.provider_runtime` provides the daemon boundary:

```python
ProviderConfig(
    provider: Literal["native", "anthropic"],
    mode: str,
    model_variant: str | None = None,
    kit_root: str | None = None,
)
prepare_provider_spec(spec) -> spec
activate_child_provider(spec) -> dict
```

`ASTROLABE_MODEL_PROVIDERS` is a JSON object keyed by backend. An optional
`DaemonSpec.provider` value takes precedence and contains the selected
`ProviderConfig` fields. `prepare_provider_spec` validates the backend, mode,
and model family without reading the kit checkout or importing its packages. It
merges only the selected provider's startup environment and stores a declarative
`settings["_provider_request"]` before daemon identity and child startup.

The child calls `activate_child_provider` after loading its spec and before
importing the model factory. It rebuilds the manifest-backed selection,
verifies the independent checkout at the pinned revision, checks that the
prepared request still matches, calls the provider hook when supported, and
requires an actual active report. The returned event records backend, provider,
kit, mode, revision, model identity, state, activation API, and the provider's
report. Native and upstream-stock states are reported as inactive. Provider
errors propagate; the helper does not retry unoptimized.

Provider mode defaults to `exact`; `fast` and `big` are used only when requested.
The native provider accepts only `mode="off"` and does not import a provider
package. The exact-mode claim remains the upstream kit's definition; this helper
does not establish numerical equivalence, GPU fit, or throughput.

## Provider boundaries

| Backend | Mode | Boundary and result |
|---|---|---|
| `protenix` | `exact`, explicit `fast`, `big` | `protenix_opt.enable(...)` patches Protenix before `ProtenixRunner` constructs its model. The loaded runner can then serve repeated predictions. |
| `chai1` | `exact`, explicit `fast`, `big` | `chai1_opt.enable(...)` patches `chai_lab.chai1`. Upstream `run_inference` reloads TorchScript components each call, so a daemon needs a new direct runner that caches them. |
| `esmfold2` | `exact`, explicit `fast`, `big` | `esmfold2_opt.enable(...)` arms the first-fold hook and must run before model construction. This is ESMFold2 only; `esmfold` is rejected as a different model family. |
| `opendde` | `exact`, explicit `fast`, `big` | `opendde_opt.enable(...)` installs the served-runner hook. The existing Astrolabe wrapper starts a new CLI process per request, so retaining the model requires a new in-process owner. |
| `rf3` | `exact`, explicit `fast`, `big` | `rosettafold3_opt.enable(...)` must run before `rf3.graph_flags` imports, but its immediate report is `applied="deferred"`. The helper refuses that report. Only a later qualified configured report can establish activation. |

`alphafold3` returns provider environment for the explicit `p2` OF3-converted
route, but `activate_provider` refuses it because the existing loaded
`ModelRunner` is not patched and the converted checkpoint is not a licensed
native AlphaFold 3 replacement.

`boltz` is rejected because the kit is Boltz 2 while `Boltz1Runner` loads
Boltz 1 and `boltz1_conf.ckpt`. `esmfold` is rejected because the kit is
ESMFold2. Unmapped backends remain native and raise `NativeBackendOnlyError`
if provider selection is requested.

The ESM-IF1 kit is a separate batched-sampling command path. Its environment
can be generated for that child, but `activate_provider` rejects calls for the
`esmif` backend. An `esm_if1_opt.enable("fast")` report must not be presented
as optimization of `ESMIF1Mutator` probability logits.

The pinned ESMFold2 implementation returns `active=True`,
`applied="deferred"`, and `levers_applied=[]` from `enable()`. That report means
the lazy hook is armed. The helper refuses it. After the kit's `apply_to()`
configures a model, the same report changes to `applied="configured"` with the
model's applied levers; that qualified form can be published as active. The
current Astrolabe ESMFold2 worker still starts a fresh batch process, so daemon
integration remains unsupported independently of this report gate.
This comes from the pinned
[`stack.py` deferred report](https://github.com/anthropics/uplifting-biomolecular-modeling/blob/f4f62fa6592ae4938d49b1757bea0cfeff9f468e/esmfold2/opt/esmfold2_opt/stack.py#L1265)
and its
[`configured` update](https://github.com/anthropics/uplifting-biomolecular-modeling/blob/f4f62fa6592ae4938d49b1757bea0cfeff9f468e/esmfold2/opt/esmfold2_opt/stack.py#L1370).

Chai's activation gate requires the installed `chai_lab` distribution version
and Python-file bytes to match its pinned wheel. `activate_provider` therefore
requires `pins.chai_lab.pinned=True` before publishing an active Chai report.
A vendored or otherwise patched `chai_lab` tree is unqualified and refused. The
current Astrolabe evaluator also reloads inference components per call, so its
loaded-daemon capability remains unsupported.
The gate is visible in the pinned
[`check_pins.py` byte comparison](https://github.com/anthropics/uplifting-biomolecular-modeling/blob/f4f62fa6592ae4938d49b1757bea0cfeff9f468e/chai1/stock/check_pins.py#L174)
and
[`stack.py` activation gate](https://github.com/anthropics/uplifting-biomolecular-modeling/blob/f4f62fa6592ae4938d49b1757bea0cfeff9f468e/chai1/opt/chai1_opt/stack.py#L738).

## Environment names

The helper emits only the variable for the selected provider:

| Backend | Variable |
|---|---|
| `alphafold3` | `AF3_JAX_OPT` |
| `protenix` | `PROTENIX_OPT` |
| `chai1` | `CHAI1_OPT` |
| `esmif` | `ESM_IF1_OPT` |
| `esmfold2` | `ESMFOLD2_OPT`, plus `ESMFOLD2_VARIANT` when supplied |
| `opendde` | `OPENDDE_OPT` |
| `rf3` | `ROSETTAFOLD3_OPT` |

`mode="off"` emits no optimization variable and does not call a hook.

## Source verification

The selected source root must contain a top-level `.git` marker. Its
`git rev-parse --show-toplevel` must equal that root, and `HEAD` must equal the
manifest revision. This rejects an empty vendor path whose Git command would
otherwise resolve through the enclosing Astrolabe checkout. The selected kit
must also carry `run.sh` and `opt/pyproject.toml`.

The provider hook performs the kit's runtime version and stock pin checks. The
helper requires the hook to return `active=True`; missing `opt_core`, a drifted
upstream package, failed checkpoint checks, or an unsupported late activation
therefore fail instead of producing a false active report.

The helper establishes source/API compatibility only. Numerical equivalence of
`exact`, GPU memory fit, and end-to-end performance still require separate
qualification.

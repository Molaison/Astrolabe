# Startup checks and evaluator initialization

`astrolabe run` validates the selected configuration before CUDA initialization,
output creation, or importing the scientific runner. The Python `Astrolabe`
entry point uses the same validation. Errors identify the configuration field
and evaluator alias; multiple configuration errors are reported together.

The checks cover component-map shape, selected stage aliases, evaluator/loss
types, required explicit loss parameters, expression syntax, seed/step/repeat
values, selected evaluator interpolations and execution controls. Unused
backend defaults are left unresolved. Backend-specific checks that need
scientific input objects still run in their owning backend.

`--dry-run` checks configuration structure and selections without requiring the
execution host's environment values or binaries. Normal execution additionally
checks DAlphaBall when selected losses require it. A complete one-sphere
`surf_vol` request verifies that the binary actually runs; a file-existence
check alone cannot detect missing shared libraries. The probe timeout defaults
to five seconds. The error includes the binary path and loader/protocol error.
An invalid explicit `GH_DALPHABALL` is not replaced with another executable.
Preflight does not install libraries or change file permissions.

PyRosetta loss implementations are imported on their first actual use.
Dependency classification reads source metadata without importing model
frameworks. The existing seeding function and loss mathematics are unchanged.

## Parallel initialization

The default keeps the existing lazy initialization and shared caller random
stream. To initialize independent native AF2, AF3 and ESMFold v1 evaluators
together, configure:

```yaml
evaluator:
  initialization:
    workers: 3
    seed: 17
```

`seed` explicitly selects independent initialization streams, derived from this
value and each stable evaluator alias. It is a different initialization policy
from the default shared stream. For a serial/parallel comparison under the same
policy, keep this seed and all aliases unchanged and vary only `workers`.
The model's later inference RNG transfer protocol is unchanged.

Parallel initialization currently requires distinct supported backends using
private native daemons. Attached services, in-process models and repeated
backend output directories are rejected before construction. Initialization
does not change any existing resource admission rules; enough resources must
still be available for the selected models.

All selected configurations are resolved before the first constructor runs.
Each evaluator reports its start, readiness time, or failure with its alias and
type. A failed constructor releases partially created owners; queued
constructors are cancelled and already-running constructors are drained before
returning the error. The error is logged immediately even when cleanup must
wait for another constructor.

CPU fixtures exercise real daemon processes, concurrent construction,
serial/parallel initialization RNG equality, parent RNG preservation and
process cleanup. They do not establish a trained-model speedup or GPU
capacity for a particular evaluator combination.

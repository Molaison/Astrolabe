# Getting started

Astrolabe has a packaged CPU path and a source-only model installation path.
Start with the CPU path to verify the package, then plan the scientific backend
you need.

## Install the CPU package

Astrolabe requires Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
astrolabe setup
astrolabe doctor
astrolabe demo --out-dir outputs/cpu-demo
```

`astrolabe setup` uses the `cpu` profile by default. The CPU profile downloads
no models, builds no container, and imports no model runtime. It reports the
package checks and the next commands.

`astrolabe doctor` checks Python, packaged configuration, the bundled demo
input, and source-installer availability. Add `--task TASK`, `--env PATH`,
`--strict`, or `--python-deps` to delegate task-aware path and import checks to
the existing `astrolabe env doctor` implementation.

`astrolabe demo` reads the bundled `9TP_cid.cif` prediction, computes contact
geometry for chains A:1-85 and B:1-87 at an 8 Angstrom cutoff, and runs the
existing posthoc collect, filter, and report commands. It performs no model
inference, downloads no data, and imports no Torch or JAX runtime. The output
directory must not exist.

The demo writes:

```text
outputs/cpu-demo/
  raw/9tp_geometry/001_run/
    config.yaml
    provenance.json
    scores.csv
    structures/step_000_ABC_model.cif
  logs/posthoc.log
  report/
  selected/
  onboarding.json
  scores.csv
  scores.sqlite
```

The input is an existing prediction. The demo verifies packaged analysis and
reporting behavior. It does not predict a structure or evaluate a new design.

## Inspect an installation plan

Run a setup plan before changing an environment.

```bash
astrolabe setup --profile protein --dry-run
astrolabe setup --profile protein --json
astrolabe setup --profile af3 --af3-weights /approved/af3/models --dry-run
```

`--dry-run` prints the selected software, each catalog asset source and URL,
required and missing licenses, AF3 weights, installer path, and exact argv.
It does not install, download, or write.

`--json` also prints a complete plan. JSON output is always plan-only, even
when `--dry-run` is omitted.

## Choose a profile

| Profile | Software selection | Intended use |
|---|---|---|
| `cpu` | None | CLI, configuration, diagnostics, and the bundled coordinate demo. |
| `protein` | `alphafold2`, `ligandmpnn` | A smaller public protein setup that combines structure evaluation with sequence design. |
| `af3` | `alphafold3` | AlphaFold 3 with user-supplied licensed parameters. |
| `rna` | `rhofold`, `nampnn` | The existing catalog RNA profile. |
| `all` | Every software ID in the installed catalog | Full inventory planning and explicitly prepared site installs. |

The `protein` profile is smaller than the catalog's `core` and `public`
profiles, but it still installs scientific runtimes and downloads public model
assets. It is not a CPU profile.

`all` includes restricted, gated, public, bundled, container-provided, and
manual software. It requires the AF3 weights path, every applicable license
acceptance, source code for Astrolabe's installer, and any manual resources
listed by the selected software.

## Override the profile

Use repeatable or comma-separated `--software` values to select from the
installed catalog. Existing catalog profiles and the onboarding `protein`
profile are accepted.

```bash
astrolabe setup --software boltz,chai1 --dry-run
astrolabe setup --software rna --dry-run
astrolabe setup --software public --json
astrolabe setup --software all --accept-license pyrosetta-license --dry-run
```

`--software` replaces the selected profile's software list. It does not change
the source-installer, asset, license, or runtime requirements of those
software IDs.

The default installation root is `~/.gh_envs`. Select another persistent root
with `--root`.

## Install AlphaFold 3

AlphaFold 3 parameters are restricted and require an authorized local
directory.

```bash
astrolabe setup \
  --profile af3 \
  --af3-weights /approved/af3/models \
  --dry-run
```

After reviewing the plan, run the same command without `--dry-run` from a full
source checkout. Source model setup activates the checkout after the installer's
build, asset, import-smoke, and doctor checks succeed. Pass
`--no-activate-checkout` to keep the existing interpreter selection.

The `af3` profile does not use `--accept-license`. Its catalog license ID,
`af3-terms`, is reported in the plan, and the installer requires the authorized
parameters directory.

## Accept licenses explicitly

Astrolabe never accepts a license for you. For the current catalog, the
installation path can require:

```text
esm3-noncommercial
esmfold2-aup
pyrosetta-license
ananas-academic
rosetta-license
```

Pass each required ID separately:

```bash
astrolabe setup \
  --software esm3,pyrosetta,ananas,rosetta \
  --accept-license esm3-noncommercial \
  --accept-license pyrosetta-license \
  --accept-license ananas-academic \
  --accept-license rosetta-license \
  --dry-run
```

Unknown license IDs fail before the plan is accepted. A plan reports missing
licenses. Actual installation stops before the backend installer starts when a
required acceptance is missing.

## Use the source model installer

The wheel and Python source distribution support the CPU package, setup plans,
doctor, and demo. Model installation also needs `scripts/install.py`, container
recipes, and vendor source metadata from a Git checkout or the full
`astrolabe-source` archive.

If that installer is missing, `astrolabe setup --dry-run` and `--json` still
work from the packaged catalog. An actual model install explains how to obtain
the full source archive. It does not run a partial setup from the wheel.

The onboarding CLI forwards these installer controls as argv items:

```text
--setup-host-envs
--fakeroot
--skip-build
--skip-import-smoke
--skip-doctor
--no-nv
--force
--apptainer-bin PATH
--sif PATH
--container-build-cache PATH
--container-builder apptainer|docker
--runner-backend apptainer|docker-apptainer
--bootstrap-apptainer auto|always|never
```

For non-AF3 profiles, onboarding adds `--allow-missing-af3`. The selected
software therefore reaches its own installation and validation path without
requiring an unrelated AF3 parameters directory.

`scripts/install.py` retains ownership of submodule initialization, asset
download and validation, license and credential preflight, container creation,
runner generation, import smoke, and environment doctor checks.

For a short model installation preview, compare the onboarding plan with the
source installer help:

```bash
python3 scripts/install.py --help
python3 scripts/install.py --list-software
```

## Check state and failures

Use these commands in this order:

```bash
astrolabe setup --profile protein --dry-run
astrolabe doctor
astrolabe demo --out-dir outputs/cpu-demo
```

After a model installation, use the task-aware check:

```bash
astrolabe doctor --task release_protein --strict
```

The CPU demo does not establish that a GPU predictor works. GPU correctness or
throughput requires the selected model weights, an allocated GPU, and a real
model smoke test.

"""Model ownership metadata without importing model frameworks."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from astrolabe.evaluators.rna import RNA_BACKENDS


@dataclass(frozen=True)
class Backend:
    name: str
    component: Literal["evaluator", "mutator", "initializer", "processor"]
    aliases: tuple[str, ...]
    software: tuple[str, ...]
    execution: Literal["daemon", "in_process", "subprocess", "custom", "local", "tool"]
    detail: str


BACKENDS = (
    Backend("alphafold3", "evaluator", ("alphafold3",), ("alphafold3",), "daemon",
            "AF3 parameter owner and explicit service_socket attachment are available."),
    Backend("alphafold2", "evaluator", ("alphafold2",), ("alphafold2",), "daemon",
            "One loaded RunModel is shared by configured oligomers in its model owner."),
    Backend("rf3", "evaluator", ("rf3",), ("rf3",), "daemon",
            "The isolated runner retains inference engines and their RNG state."),
    Backend("esmfold", "evaluator", ("esmfold",), ("esmfold",), "daemon",
            "The model owner retains the ESMFold v1 model."),
    Backend("esmfold2", "evaluator", ("esmfold2",), ("esmfold2",), "daemon",
            "Biohub ESMFold2 uses an independent interpreter and checkpoint family."),
    Backend("chai1", "evaluator", ("chai1",), ("chai1",), "daemon",
            "The owner caches Chai components and preserves CPU offload between calls."),
    Backend("boltz", "evaluator", ("boltz",), ("boltz",), "daemon",
            "The owner retains Boltz 1; this is not a Boltz 2 model substitution."),
    Backend("protenix", "evaluator", ("protenix",), ("protenix",), "daemon",
            "The owner retains the model; prediction and result dumping are one request."),
    Backend("opendde", "evaluator", ("opendde",), ("opendde",), "daemon",
            "OpenDDE uses its own interpreter and chemistry assets."),
    Backend("rhofold", "evaluator", ("rhofold",), ("rhofold",), "daemon",
            "The owner retains the RNA model and performs device transfers."),
    *(Backend(name, "evaluator", (name,), (name,), "daemon",
              "Single RNA chain; explicit CPU preparation, device inference and CPU output.")
      for name in RNA_BACKENDS),
    Backend("sequence", "evaluator", ("sequence",), (), "local",
            "Sequence-only evaluation without structures or synthetic confidence."),
    Backend("dummy", "evaluator", ("dummy",), (), "local", "No model weights."),
    Backend("rosetta_crystal", "evaluator", ("rosetta_crystal",), ("pyrosetta",), "local",
            "PyRosetta scoring has no neural model weights."),
    Backend("ligandmpnn", "mutator",
            ("ligandmpnn", "grouped_ligandmpnn", "ProteinMPNN", "MultistateProteinMPNN",
             "multistate_proteinmpnn", "FarApartTiedMultitargetLigandMPNN",
             "far_apart_tied_multitarget_ligandmpnn"),
            ("ligandmpnn",), "daemon", "The owner retains the selected MPNN model subtype."),
    Backend("dynamicmpnn", "mutator", ("dynamicmpnn",), ("dynamicmpnn",), "daemon",
            "Requires a manually supplied resident backend_factory, source and checkpoint."),
    Backend("surfdesign", "mutator", ("surfdesign",), ("surfdesign",), "daemon",
            "The surface-design runtime persists in its isolated interpreter."),
    Backend("surfpro", "mutator", ("surfpro",), ("surfpro",), "daemon",
            "The surface-design runtime persists in its isolated interpreter."),
    Backend("carbondesign", "mutator", ("carbondesign",), ("carbondesign",), "daemon",
            "The owner retains its model and feature assets."),
    Backend("esmif", "mutator", ("esmif",), ("esmif",), "daemon",
            "The owner retains ESM-IF1; the coordinator reconstructs only its alphabet."),
    Backend("esm2", "mutator", ("esm2",), ("esm2",), "daemon",
            "The owner retains ESM2; the coordinator reconstructs only its alphabet."),
    Backend("esm3", "mutator", ("esm3",), ("esm3",), "daemon",
            "The owner retains the selected ESM3 model; prompt preparation stays local."),
    Backend("dplm", "mutator", ("dplm",), ("dplm",), "daemon",
            "The DPLM runner retains its model and sample-bank policy."),
    Backend("nampnn", "mutator", ("nampnn",), ("nampnn",), "daemon",
            "NA-MPNN preserves mixed protein, DNA and RNA design semantics."),
    Backend("local_mutation", "mutator", ("mcmc", "pore_blocker_mcmc", "xmix", "motifbreaker"),
            (), "local", "These sequence proposal policies have no model weights."),
    Backend("rfd3", "initializer", ("rfd3",), ("rfd3",), "daemon",
            "The RFD3 engine uses an isolated Foundry interpreter."),
    Backend("dplm_initializer", "initializer", ("dplm",), ("dplm",), "daemon",
            "Initialization uses the configured DPLM runner."),
    Backend("rapidock", "processor", ("RAPiDock",), ("rapidock",), "daemon",
            "The owner retains score and confidence models using a verified source runtime."),
    Backend("chemnet_placer", "processor", ("chemnet_placer",), (), "daemon",
            "The owner retains a manually installed PLACER model; inputs are request-scoped."),
    Backend("docking_tools", "processor",
            ("autodock_gpu", "autodock_vina", "haddock3_nucleic", "HADDOCK3_nucleic"),
            ("autodock-gpu", "autodock-vina", "haddock3"), "tool",
            "These docking executables do not expose persistent neural weights."),
    Backend("symmetry_canonicalizer", "processor", ("symmetry_canonicalizer",),
            ("rpxdock",), "tool", "RPXDock geometry and score tables are distinct from RAPiDock."),
    Backend("geometry", "processor",
            ("all_alanine", "ligand_fpocket_alignment", "ligand_centroid_alignment",
             "dimer_interface_placer", "hotspot_reference_align"),
            (), "local", "Geometry operations have no neural model weights."),
)


def get_backend(name: str, *, component: str | None = None) -> Backend:
    records = [record for record in BACKENDS
               if component is None or component == record.component]
    matches = [record for record in records if name == record.name]
    if not matches:
        matches = [record for record in records if name in record.aliases]
    if len(matches) != 1:
        raise ValueError(f"Unknown or ambiguous backend {name!r}; specify its component.")
    return matches[0]


def catalog_payload() -> list[dict]:
    return [asdict(record) for record in BACKENDS]

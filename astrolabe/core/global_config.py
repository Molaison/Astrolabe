# astrolabe/core/global_config.py
from dataclasses import dataclass, field
import sys
from types import ModuleType

DEFAULT_MUTATION_FRACTION = 0.1


constants_names = [
    "seed",
    "num_tokens",
    "PARSED_FILE",
    "predefined_statement",
    "segments",
    "named_files",
    "exclude_AA",
    "covalent_bonds",
    "protomers",
    "mutable_protomers",
    "segment_manager",
    "initial_temperature",
    "out_dir",
    "raw_model_results_dir",
    "is_debug",
    "use_metropolis_acceptance",
    "stage_manager",
    "loss_computer",
    "predockers",
    "mutators",
    "initializer_result",
    "evaluators",
    "decisions",
    "evaluation_journal",
    "resource_owner",
    "obabel_path",
    "molfile_to_params",
    "dssp_path",
    "usalign_path",
    "ananas_path",
    "enable_pyrosetta",
    "skip_complex_poses",
    "allow_mmcif_pdb_fallback",
    "per_residue_scores",
    "pre_eval_hooks",
    "protenix_constraint",
    "binding_site",
    "dynamic_binding_site",
    "dynamic_binding_site_enabled",
    "chai_restraints",
    "channel",
    "cyclization",
    "ligand_resnames",
    "extra_residue_params",
    "nonstandard_aliases",
    "user_ccd",
    "structure_start",
    # Add more constant names here as needed
]

variable_names = [
    "step",
    "ignored_complexes",
    "negative_design_complexes",
    "mutation_num",
    "pore_bottleneck_activation_index",
]

class RuntimeState:
    """Storage for one run. Legacy module access points to one current instance."""

    def __init__(self):
        self.constants = {name: None for name in constants_names}
        self.initialized_constants = {name: False for name in constants_names}
        self.variables = {name: None for name in variable_names}

    def init_const(self, name, value):
        if name not in constants_names:
            raise ValueError(f"'{name}' is not a recognized constant name")
        if self.initialized_constants[name]:
            if self.constants[name] == value:
                return
            raise RuntimeError(
                f"Constant '{name}' is already initialized. Call `global_config.reset()` before re-initializing constants."
            )
        self.constants[name] = value
        self.initialized_constants[name] = True

    def update_const(self, name, value):
        if name not in constants_names:
            raise ValueError(f"'{name}' is not a recognized constant name")
        if not self.initialized_constants[name]:
            raise RuntimeError(f"Constant '{name}' has not been initialized yet")
        self.constants[name] = value

    def get_const(self, name):
        if name not in constants_names:
            raise ValueError(f"'{name}' is not a recognized constant name")
        if not self.initialized_constants[name]:
            raise RuntimeError(f"Constant '{name}' has not been initialized yet")
        return self.constants[name]

    def set_variable(self, name, value):
        if name not in variable_names:
            raise ValueError(f"'{name}' is not a recognized variable name")
        self.variables[name] = value

    def get_variable(self, name):
        if name not in variable_names:
            raise ValueError(f"'{name}' is not a recognized variable name")
        return self.variables[name]


_current_state = RuntimeState()


@dataclass(frozen=True)
class RuntimeContext:
    """Explicit access to a particular run store, including across reset()."""

    state: RuntimeState = field(default_factory=lambda: _current_state)
    _MISSING = object()

    def get_const(self, name: str, default=_MISSING):
        try:
            return self.state.get_const(name)
        except RuntimeError:
            if default is not self._MISSING:
                return default
            raise

    @property
    def protomers(self):
        return self.get_const("protomers", {})

    @property
    def mutable_protomers(self):
        return self.get_const("mutable_protomers", {})

    @property
    def mutators(self):
        return self.get_const("mutators", {})

    @property
    def evaluators(self):
        return self.get_const("evaluators", {})

    @property
    def stage_manager(self):
        return self.get_const("stage_manager", None)

    @property
    def step(self):
        value = self.state.get_variable("step")
        return 0 if value is None else value

    def set_variable(self, name: str, value) -> None:
        self.state.set_variable(name, value)

    def get_variable(self, name: str):
        return self.state.get_variable(name)


def build_runtime_context(state: RuntimeState | None = None) -> RuntimeContext:
    return RuntimeContext(_current_state if state is None else state)


def init_const(name, value):
    return _current_state.init_const(name, value)


def update_const(name, value):
    return _current_state.update_const(name, value)


def set_variable(name, value):
    return _current_state.set_variable(name, value)


def get_const(name):
    return _current_state.get_const(name)


def get_variable(name):
    return _current_state.get_variable(name)


def __getattr__(name):
    if name in constants_names:
        return get_const(name)
    if name in variable_names:
        return get_variable(name)
    # Compatibility with historical test monkeypatches, not a second store.
    if name == "_initialized_constants":
        return _current_state.initialized_constants
    if name.startswith("_"):
        key = name[1:]
        if key in constants_names:
            return _current_state.constants[key]
        if key in variable_names:
            return _current_state.variables[key]
    raise AttributeError(f"module 'global_config' has no attribute '{name}'")


def reset():
    """Start a new legacy run; previously created contexts keep their store."""
    global _current_state
    _current_state = RuntimeState()


class _ConfigModule(ModuleType):
    """Forward old private-field monkeypatches into the current run store."""

    def __setattr__(self, name, value):
        key = name[1:] if name.startswith("_") else None
        if key in constants_names:
            _current_state.constants[key] = value
        elif key in variable_names:
            _current_state.variables[key] = value
        elif name in constants_names or name in variable_names:
            raise AttributeError("Use init_const/update_const or set_variable")
        else:
            super().__setattr__(name, value)


sys.modules[__name__].__class__ = _ConfigModule

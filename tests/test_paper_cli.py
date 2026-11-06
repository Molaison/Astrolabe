from __future__ import annotations

import contextlib
import csv
import io
import json
import random
from types import SimpleNamespace
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from astrolabe import cli, paper
from astrolabe.configuration import (
    dispatch_selected_search,
    register_search_factory,
)
from astrolabe.design.repeats import RepeatRun, materialize_repeat, plan_repeat_runs
from astrolabe.shell_completion import complete_words
import run


MIGRATED_SEARCH_PROFILES = {
    "rna-oligomer-4",
    "rna-oligomer-5",
    "bivalent-rna",
    "bivalent-rna-archived",
    "protein-binder-dna-archived",
    "protein-binder-rna-archived",
}


def invoke(arguments):
    return invoke_capture(arguments)[0]


def invoke_capture(arguments):
    ns = cli._build_parser().parse_args(["paper", *arguments])
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        result = ns._handler(ns)
    return result, output.getvalue()


def test_every_paper_protocol_has_a_composable_configuration():
    for name in paper.PROTOCOLS:
        ns = cli._build_parser().parse_args(["paper", "show", name, "--config-env", "env"])
        cfg = paper._compose(ns)
        assert cfg.paper_protocol
        assert cfg.steps >= 1
        if name not in MIGRATED_SEARCH_PROFILES:
            assert cfg.get("search") is None
            assert cfg.get("repeat_plan") is None


def test_paper_preview_and_last_override_need_no_models(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Preview must not enter model execution")

    monkeypatch.setattr(cli, "_run_from_config", forbidden)
    inputs = tmp_path / "input.yaml"
    inputs.write_text("seqs:\n  B: FAD\ntypes:\n  B: c\npaper_protocol:\n  supplied_input: local\n")
    dump = tmp_path / "filled.yaml"
    assert invoke(["run", "ligand-binder-af3", "--inputs", str(inputs),
                   "--seed", "17", "--overrides", "steps=7", "--dump-config", str(dump),
                   "--config-env", "env", "--dry-run"]) == 0
    cfg = OmegaConf.load(dump)
    assert cfg.seed == 17 and cfg.steps == 7
    assert cfg.paper_protocol.supplied_input == "local"
    assert cfg.seqs.B == "FAD"


def test_deprecated_paper_engine_requires_registered_search_selection(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        cli,
        "_run_from_config",
        lambda *args, **kwargs: pytest.fail("deprecated engine was routed"),
    )

    legacy = tmp_path / "legacy.yaml"
    legacy.write_text("paper_protocol:\n  engine: rna_beam\n")
    with pytest.raises(ValueError, match="deprecated.*search.algorithm"):
        invoke(
            [
                "run",
                "evaluate-af3",
                "--inputs",
                str(legacy),
                "--config-env",
                "env",
                "--dry-run",
            ]
        )

    empty = tmp_path / "empty.yaml"
    empty.write_text("paper_protocol:\n  engine: ''\n")
    with pytest.raises(ValueError, match="deprecated.*search.algorithm"):
        invoke(
            [
                "run",
                "evaluate-af3",
                "--inputs",
                str(empty),
                "--config-env",
                "env",
                "--dry-run",
            ]
        )

    selected = tmp_path / "selected.yaml"
    selected.write_text(
        "paper_protocol:\n"
        "  engine: rna_beam\n"
        "search:\n"
        "  algorithm: branching\n"
        "  owner: archived_refinement\n"
    )
    assert invoke(
        [
            "run",
            "rna-oligomer-4",
            "--inputs",
            str(selected),
            "--config-env",
            "env",
            "--dry-run",
        ]
    ) == 0

    trajectory = tmp_path / "trajectory.yaml"
    trajectory.write_text("paper_protocol:\n  engine: trajectory\n")
    assert invoke(
        [
            "run",
            "evaluate-af3",
            "--inputs",
            str(trajectory),
            "--config-env",
            "env",
            "--dry-run",
        ]
    ) == 0


def test_paper_run_reaches_existing_engine_and_resolves_input_paths(tmp_path, monkeypatch):
    source = tmp_path / "input"
    source.mkdir()
    inputs = source / "fixed.yaml"
    inputs.write_text("seqs:\n  A: ACDEFG\n  B: +AGST\n")
    out = tmp_path / "results"
    seen = []
    previous = Path.cwd()

    def execute(cfg, *, autoselect_gpu):
        seen.append((Path.cwd(), cfg.seqs.A, cfg.seed, cfg.repeats, cfg.out_dir))

    monkeypatch.setattr(cli, "_run_from_config", execute)
    assert invoke(["run", "evaluate-af3", "--inputs", str(inputs), "--seed", "7",
                   "--num-runs", "5", "--out-dir", str(out), "--config-env", "env"]) == 0
    assert seen == [(source, "ACDEFG", 7, 5, str(out))]
    assert Path.cwd() == previous


def test_fixed_evaluation_rejects_random_sequence_generation(tmp_path, monkeypatch):
    inputs = tmp_path / "fixed.yaml"
    inputs.write_text("seqs:\n  A: 80-120\n  B: +AGST\n")
    monkeypatch.setattr(cli, "_run_from_config", lambda *a, **k: pytest.fail("random evaluation ran"))
    with pytest.raises(ValueError, match="literal sequences"):
        invoke(["run", "evaluate-af3", "--inputs", str(inputs), "--config-env", "env"])


def test_bivalent_archived_starts_use_their_original_seeds(tmp_path, monkeypatch):
    import astrolabe.rna_beam as beam
    seen = []
    engine_calls = []

    def execute(cfg, repeat):
        seen.append(
            (
                Path(cfg.out_dir).name,
                repeat.index,
                repeat.seed,
                cfg.search.start.sequence,
            )
        )
        return {
            "budget_complete": True,
            "repeat": repeat.index,
            "seed": repeat.seed,
        }

    def run_from_config(cfg, *, autoselect_gpu):
        engine_calls.append(cfg)
        run.run_repeats(cfg)

    monkeypatch.setattr(beam, "run_configured_refinement", execute)
    monkeypatch.setattr(beam, "run_rna_beam", lambda *a, **k: pytest.fail("old RNA routing ran"))
    monkeypatch.setattr(beam, "run_bivalent_rna_beam", lambda *a, **k: pytest.fail("old RNA routing ran"))
    monkeypatch.setattr(cli, "_run_from_config", run_from_config)
    monkeypatch.setattr(run, "_run_pipeline", lambda cfg: pytest.fail("search used generic pipeline"))
    rc, stdout = invoke_capture(
        [
            "run",
            "bivalent-rna-archived",
            "--num-runs",
            "3",
            "--out-dir",
            str(tmp_path),
            "--config-env",
            "env",
        ]
    )
    assert rc == 0
    assert len(engine_calls) == 1
    assert seen == [
        ("001_seed_67101", 0, 67101, "UCGGCAGCUGCCAUCUGACU"),
        ("002_seed_67102", 1, 67102, "CCCAACAUGGUCCCAGGGCG"),
        ("003_seed_67103", 2, 67103, "CACUGUUUCGGUAAGUCGCA"),
    ]
    assert [
        json.loads(line)
        for line in stdout.splitlines()
        if line.startswith("{")
    ] == [
        {"budget_complete": True, "repeat": 0, "seed": 67101},
        {"budget_complete": True, "repeat": 1, "seed": 67102},
        {"budget_complete": True, "repeat": 2, "seed": 67103},
    ]


def test_nucleic_target_initial_sequences_match_original_seeded_generator(tmp_path, monkeypatch):
    inputs = tmp_path / "target.yaml"
    inputs.write_text("seqs:\n  B: +ACGU\nhomologies:\n  B: homology.json\n")
    seen = []
    engine_calls = []
    monkeypatch.setenv("GH_WEIGHT_PROTENIX_V1", "/unused/model.pt")

    def run_from_config(cfg, *, autoselect_gpu):
        engine_calls.append(cfg)
        run.run_repeats(cfg)

    monkeypatch.setattr(cli, "_run_from_config", run_from_config)
    monkeypatch.setattr(run, "_run_pipeline", lambda cfg: seen.append((cfg.seed, cfg.seqs.A)))
    invoke(["run", "protein-binder-rna-archived", "--inputs", str(inputs), "--seed", "1729",
            "--num-runs", "2", "--out-dir", str(tmp_path / "out"), "--config-env", "env"])
    assert len(engine_calls) == 1
    for (seed, sequence), expected_seed in zip(seen, (1729, 1730)):
        rng = random.Random(expected_seed)
        expected = "".join(rng.choices("ADEFGHIKLMNPQRSTVWY", k=rng.randint(60, 80)))
        assert seed == expected_seed and sequence == expected


def test_repeat_plan_owns_seed_precedence_output_and_initial_sequence(tmp_path):
    starts = [
        {"sequence": "AAAA", "seed": 101},
        {"sequence": "CCCC", "seed": 202},
    ]
    cfg = OmegaConf.create(
        {
            "task_name": "repeat-plan",
            "repeats": 2,
            "seed": 7,
            "out_dir": str(tmp_path),
            "search": {"initialization": "archived", "archived_starts": starts},
            "repeat_plan": {
                "seed_override": 41,
                "output": "{repeat:03d}_seed_{seed}",
            },
        }
    )
    runs = plan_repeat_runs(cfg)
    assert [(item.index, item.seed, item.start["sequence"], item.output.name) for item in runs] == [
        (0, 41, "AAAA", "001_seed_41"),
        (1, 42, "CCCC", "002_seed_42"),
    ]

    generated = plan_repeat_runs(
        OmegaConf.create(
            {
                "task_name": "generated",
                "repeats": 2,
                "seed": 9,
                "out_dir": str(tmp_path),
                "repeat_plan": {
                    "initial_sequence": {
                        "owner": "python_random",
                        "target": "A",
                        "alphabet": "AC",
                        "length": [3, 3],
                    }
                },
            }
        )
    )
    materialized = materialize_repeat(
        OmegaConf.create(
            {
                "task_name": "generated",
                "repeats": 2,
                "seed": 9,
                "out_dir": str(tmp_path),
                "seqs": {"A": "???"},
                "repeat_plan": {
                    "initial_sequence": {
                        "owner": "python_random",
                        "target": "A",
                        "alphabet": "AC",
                        "length": [3, 3],
                    }
                },
            }
        ),
        generated[0],
    )
    expected_rng = random.Random(9)
    expected_rng.randint(3, 3)
    expected = "".join(expected_rng.choices("AC", k=3))
    assert generated[0].seed == 9
    assert materialized.seed == 9
    assert materialized.seqs.A == expected


@pytest.mark.parametrize(
    ("search", "message"),
    [
        (None, "explicit search route"),
        ({}, "explicit search.algorithm"),
        ({"algorithm": "test"}, "explicit search.owner"),
        ({"owner": "test"}, "explicit search.algorithm"),
    ],
)
def test_materialize_repeat_start_requires_explicit_search_route(
    tmp_path: Path,
    search: dict[str, str] | None,
    message: str,
) -> None:
    cfg = {
        "task_name": "repeat-start-route",
        "repeats": 1,
        "seed": 7,
        "out_dir": str(tmp_path),
    }
    if search is not None:
        cfg["search"] = search
    repeat = RepeatRun(
        index=0,
        total=1,
        seed=7,
        start={"sequence": "AAAA"},
    )

    with pytest.raises(ValueError, match=message):
        materialize_repeat(OmegaConf.create(cfg), repeat)


def test_materialize_repeat_preserves_an_explicit_route_for_start(
    tmp_path: Path,
) -> None:
    repeat = RepeatRun(
        index=0,
        total=1,
        seed=7,
        start={"sequence": "AAAA"},
    )
    cfg = OmegaConf.create(
        {
            "task_name": "repeat-start-route",
            "repeats": 1,
            "seed": 7,
            "out_dir": str(tmp_path),
            "search": {"algorithm": "test", "owner": "test"},
        }
    )

    materialized = materialize_repeat(cfg, repeat)

    assert materialized.search.algorithm == "test"
    assert materialized.search.owner == "test"
    assert materialized.search.start.sequence == "AAAA"
    assert materialized.search.archived_start == 0


def test_selected_search_lookup_is_owner_qualified_and_never_falls_back():
    repeat = SimpleNamespace()
    assert dispatch_selected_search(OmegaConf.create({"steps": 1}), repeat) is False
    with pytest.raises(ValueError, match="Unknown selected search factory"):
        dispatch_selected_search(
            OmegaConf.create({"search": {"algorithm": "branching", "owner": "missing"}}),
            repeat,
        )

    seen = []
    key_algorithm = "test-algorithm"
    key_owner = "test-owner"
    register_search_factory(
        key_algorithm,
        key_owner,
        lambda cfg, item: seen.append((cfg.search.algorithm, item)),
    )
    selected = OmegaConf.create(
        {"search": {"algorithm": key_algorithm, "owner": key_owner}}
    )
    assert dispatch_selected_search(selected, repeat) is True
    assert seen == [(key_algorithm, repeat)]

    rejected = []
    invalid_algorithm = "invalid-summary-format"
    register_search_factory(
        invalid_algorithm,
        key_owner,
        lambda cfg, item: rejected.append(item),
    )
    with pytest.raises(ValueError, match="summary_format.*json"):
        dispatch_selected_search(
            OmegaConf.create(
                {
                    "search": {
                        "algorithm": invalid_algorithm,
                        "owner": key_owner,
                        "summary_format": "yaml",
                    }
                }
            ),
            repeat,
        )
    assert rejected == []


def test_six_migrated_profiles_have_canonical_search_or_repeat_fields():
    for name in MIGRATED_SEARCH_PROFILES:
        ns = cli._build_parser().parse_args(
            ["paper", "show", name, "--config-env", "env"]
        )
        cfg = paper._compose(ns)
        assert cfg.get("beam_search") is None
        assert cfg.paper_protocol.get("engine") is None
        if name.startswith(("rna-oligomer", "bivalent-rna")):
            assert cfg.search.algorithm == "branching"
            assert cfg.search.owner == "archived_refinement"
            assert cfg.search.measurement in {"confidence", "bivalent_geometry"}
            assert cfg.search.summary_format == "json"
        else:
            assert cfg.repeat_plan.initial_sequence.owner == "python_random"


def test_input_bundles_extract_through_public_command(tmp_path):
    root = Path(paper.__file__).resolve().parents[1] / "resources"
    assert invoke(["prepare-inputs", "--bundle", "cyclic", "--out-dir", str(tmp_path)]) == 0
    manifest = json.loads((root / "paper_inputs_cyclic.json").read_text())
    assert all((tmp_path / item["path"]).is_file() for item in manifest["files"])


def test_paper_fixed_completion_does_not_read_configuration_files(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Static paper completion must not scan the filesystem")

    monkeypatch.setattr(Path, "glob", forbidden)
    values = {item.value for item in complete_words(["paper", "run", "rna-"], cword=2)}
    assert values == {"rna-aptamer", "rna-oligomer-4", "rna-oligomer-5"}
    values = {item.value for item in complete_words(["paper", "run", "bivalent-rna", "--"], cword=3)}
    assert {"--data-dir", "--inputs", "--seed", "--dry-run"} <= values


def test_parent_redesign_fixes_only_strict_heavy_atom_contacts(tmp_path):
    parent = tmp_path / "parent.pdb"
    atoms = [("ALA", "A", 1, 0.0), ("SER", "A", 2, 10.0), ("GLY", "B", 1, 3.0)]
    lines = [f"ATOM  {i:5d}  CA  {res} {chain}{pos:4d}    {x:8.3f}{0.:8.3f}{0.:8.3f}  1.00 90.00           C\n"
             for i, (res, chain, pos, x) in enumerate(atoms, 1)]
    parent.write_text("".join(lines) + "END\n")
    ns = cli._build_parser().parse_args(["paper", "show", "solublempnn-control", "--config-env", "env"])
    cfg = paper._compose(ns)
    paper._prepare_parent_redesign(cfg, parent)
    assert cfg.seqs.A == "+A S"
    assert cfg.seqs.B == "+G"
    assert cfg.paper_protocol.fixed_parent_interface_positions == [1]
    assert cfg.initializer.structure.source == str(parent)


def test_explicit_framework_switch_changes_homology_and_database_subset():
    ns = cli._build_parser().parse_args(["paper", "run", "antibody-framework",
                                        "--framework", "scfv4d5_vhvl", "--config-env", "env"])
    cfg = paper._compose(ns)
    paper._framework(cfg, ns)
    assert cfg.pre_eval_hooks.af3_antibody_msa_cache.af3_turboab_db_subset == "IgG"
    assert cfg.homologies.A.framework_name == "scfv_vh_linker_vl__rfantibody_4d5"


def test_baseline_cli_preview_exposes_native_assets_without_execution(tmp_path, monkeypatch):
    import subprocess
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("preview executed"))
    native = tmp_path / "launcher.yaml"
    native.write_text("native_config: target.yaml\nnum_designs: 1000\nbudget: 80\n")
    output = tmp_path / "baseline"
    assert invoke(["baseline", "boltzgen", "--inputs", str(native), "--out-dir", str(output), "--dry-run"]) == 0
    assert not output.exists()
    choices = {item.value for item in complete_words(["paper", "baseline", "bind"], cword=2)}
    assert choices == {"bindcraft", "bindcraft-dual"}


def test_statistics_commands_write_results_with_explicit_columns_and_seed(tmp_path):
    rows = [dict(unit_id="one", method="Astrolabe-AF3", group=ligand,
                 confidence=.9, ptm=.8, plddt=.9, ddg=-45, dsasa=500,
                 ligand_hbonds=3, sc=.7, rg_ratio=1.1,
                 condition="dual", molecule_type="minibinder", ddg_ab=-45, ddg_ac=-40)
            for ligand in ("FAD", "IAI", "OQO", "SAM")]
    records = tmp_path / "records.csv"
    with records.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    common = ["--records", str(records), "--column", "iptm=confidence"]
    ranking = tmp_path / "rank"
    assert invoke(["analyze", "ligand-rank", *common, "--out-dir", str(ranking)]) == 0
    summary = json.loads((ranking / "ligand_ranking.json").read_text())
    assert summary["fixed_design_units"] == 4
    assert summary["columns"]["iptm"] == "confidence"
    sensitivity = tmp_path / "sensitivity"
    assert invoke(["analyze", "sensitivity", "--endpoint", "ligand-binder", *common,
                   "--planned", str(records), "--out-dir", str(sensitivity),
                   "--bootstrap-samples", "20", "--seed", "0"]) == 0
    ddg = tmp_path / "ddg"
    assert invoke(["analyze", "ddg", "--records", str(records), "--out-dir", str(ddg),
                   "--bootstrap-samples", "20", "--seed", "0"]) == 0
    summary = json.loads((ddg / "ddg_bootstrap.json").read_text())
    assert summary["parameters"]["seed"] == 0
    assert all(row["median_signed_delta"] == 5 for row in summary["groups"])

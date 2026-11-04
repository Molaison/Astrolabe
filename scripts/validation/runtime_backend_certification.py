#!/usr/bin/env python3
"""Inventory builtin registries and emit a scoped, evidence-qualified matrix.

This command imports no model/framework, downloads nothing, allocates no GPU,
and runs no certification. Registration is not evidence of backend execution.
Unknown/missing registry entries fail closed instead of silently disappearing.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess


REGISTRIES = {
    'evaluator': ('astrolabe/evaluators/utils.py', 'register_evaluator_factory'),
    'mutator': ('astrolabe/mutators/base.py', 'register_mutator_factory'),
}
COMMON_IDENTITY = [
    'actual source file digests and local patches', 'all checkpoints and auxiliary assets',
    'Python/framework/native library versions', 'effective scientific configuration and numerical settings',
    'input structures/sequences/topology/masks and exact RNG state/seed protocol',
    'allocated physical device identity and resource budget',
]
REAL_GATES = [
    'single pretrained request', 'two continuous requests and observed load count',
    'GPU tensor residency and real execution-boundary trace', 'input/config cache invalidation',
    'PDB/coordinate/score parity under a prespecified tolerance', 'RNG/proposal/acceptance/stage decision parity',
    'failure injection and observed process/VRAM recovery', 'repeated fixed-workload effective throughput',
]


def scope(name, **configuration):
    return {'variant': name, 'configuration_subset': configuration}


def entry(keys, factory, scopes, freeze, boundary, tests, *, pins=(), target='pretrained_gpu', note=''):
    return {'keys': keys.split(), 'factory': factory, 'scopes': scopes,
            'configuration_to_freeze': freeze, 'boundary': boundary,
            'available_tests': tests, 'vendor_gitlinks': list(pins),
            'target': target, 'note': note}


def catalog():
    mpnn_types = ['ligand_mpnn','protein_mpnn','soluble_mpnn',
                  'per_residue_label_membrane_mpnn','global_label_membrane_mpnn']
    mpnn_scopes = [scope(name, model_type=name) for name in mpnn_types]
    mpnn_freeze = ['exact checkpoint for selected model_type', 'seed and constructor RNG',
                  'temperature, masks, ties, atom context, batch/sample counts',
                  'synchronization_policy and proposal_policy']
    result = {
      'evaluator': [
        entry('alphafold3','_build_alphafold3_evaluator',
          [scope('native AF3 configured model_dir')],
          ['model_dir','num_recycles','num_samples','rng_seeds','flash_attention_implementation','CCD/MSA/templates/topology'],
          'owner CPU features; worker continuous JAX forward and D2H; owner CPU extraction/scoring',
          ['tests/test_backend_stages_cpu.py','tests/test_af3_feature_codec_real.py'], pins=['vendor/alphafold3']),
        entry('alphafold2','_build_alphafold2_evaluator',
          [scope('default model_3_multimer_v3',model_id=3,suffix='multimer_v3'),
           scope('configured ptm model',suffix='ptm'),scope('other explicit model_id/suffix pairing')],
          ['model_id','suffix','data_dir/checkpoint','recycles','msa_clusters','seed','initial_guess'],
          'CPU prepare_only/process_features; audited fixed-parameter predict and D2H in worker',
          ['tests/test_af2_resident.py','tests/test_af2_broker_resident.py'], pins=['vendor/alphafold2'],
          note='Each model ID, suffix, source and checkpoint pairing is a separate certification.'),
        entry('rf3','_build_rf3_evaluator',[scope('Foundry configured RF3 checkpoint')],
          ['repo_path','ckpt_path','Python environment','effective config including seed'],
          'resident engine; native run pipeline remains composite; exact-config cache and post-init RNG replay',
          ['tests/test_rf3_resident.py','tests/test_rf3_evaluator.py']),
        entry('esmfold','_build_esmfold_evaluator',
          [scope('ESMFold1 default',model_name='esmfold_3B_v1'),scope('explicit local ESMFold1 .pt')],
          ['model_name/checkpoint','ESM trunk hub dependencies','device','torch numerical settings','ordered chain sequences'],
          'audited native atom14->37 and required D2H/RNG under lease; host PDB formatting; unknown helper composite fallback',
          ['tests/test_esmfold_output_boundary.py','tests/test_esmfold1_resident.py'],
          pins=['vendor/esm2','vendor/openfold_esm2'],
          note='f102 located a 2,097,152,000-byte invalid ZIP folding checkpoint; fair-esm/OpenFold distribution metadata and the expected audited ESM helper path are absent. No actual ESM import test or pretrained run is inferred. CPU native-output differential remains synthetic.'),
        entry('esmfold2','_build_esmfold2_evaluator',
          [scope('Biohub default',model_name='biohub/ESMFold2'),scope('configured Biohub snapshot/model name')],
          ['model_name and folding snapshot','nested ESMC snapshot','esmc_precision','num_loops','num_sampling_steps',
           'num_diffusion_samples','seed','lm_dropout','msa_max_depth','msa_column_mask_rate','chunk_size'],
          'CPU request preparation; native builder.fold and materialization remain composite',
          ['tests/test_esmfold2_biohub_wrapper.py','tests/test_esmfold2_resident_settings.py'],
          pins=['vendor/biohub_esm','vendor/biohub_transformers']),
        entry('chai1','_build_chai1_evaluator',
          [scope('default component CPU offload',component_residency='cpu'),
           scope('explicit component device residency',component_residency='device')],
          ['six component checkpoints','optional ESM2/conformer assets','component_residency','use_esm_embeddings',
           'num_trunk_recycles','num_diffn_timesteps','seed'],
          'CPU feature context; ESM and folding subgraphs separately admitted; native finalization still partly composite',
          ['tests/test_chai1_resident.py','tests/test_evaluator_lifecycle_contract.py'], pins=['vendor/chai_lab']),
        entry('boltz','_build_boltz_evaluator',[scope('Boltz1 / Boltz1.load_from_checkpoint')],
          ['boltz1_conf.ckpt or explicit Boltz1 checkpoint','recycling_steps','sampling_steps','diffusion_samples',
           'MSA inputs','precision=32 in current runner','Lightning environment'],
          'CPU YAML/MSA preparation; Lightning predict/dataloader/transfer/writer remains composite',
          ['tests/test_boltz1_resident.py','tests/test_boltz_wrapper.py'], pins=['vendor/boltz'],
          note='No Boltz2 registry or loader exists here; an arbitrary checkpoint is not Boltz2 support.'),
        entry('opendde','_build_opendde_evaluator',
          [scope('default',model_name='opendde_v1'),scope('explicit native model/checkpoint')],
          ['model_name','load_checkpoint_path','CCD/templates','num_cycles','num_steps','num_samples','dtype','seed',
           'foldcp_mode=single and one allocated device'],
          'CPU preprocessing; native loop/dataloader/model/output/cleanup stays in one composite lease',
          ['tests/test_opendde_resident.py','tests/test_opendde_wrapper.py'], pins=['vendor/opendde']),
        entry('mol-1 mol1','_build_mol1_evaluator',
          [scope('default',model_name='mol-1-base'),scope('explicit Mol-1 checkpoint/source')],
          ['actual mol/ source','checkpoint','CCD/cluster/obsolete assets','seeds','sample/recycle/precision config'],
          'independent Mol-1 adapter; protein input owner preparation and native composite resident worker',
          ['tests/test_mol1_resident.py'],
          note='Two aliases of one factory. Neither Protenix compatibility nor a constraint variant is inferred.'),
        entry('protenix','_build_protenix_evaluator',
          [scope(name,model_name=name) for name in ('protenix_base_default_v0.5.0','protenix_base_constraint_v0.5.0',
                                                  'protenix_base_default_v1.0.0','protenix-v2')],
          ['requested/effective model_name','identity_policy','use_constraints','checkpoint/source/env pairing',
           'seeds/samples/recycles','dtype','dynamic skip_amp','mc_dropout'],
          'CPU features/dumper; resident predict with original RNG/config handling and D2H',
          ['tests/test_protenix_resident.py','tests/test_protenix_wrapper_covalent_bonds.py'], pins=['vendor/protenix'],
          note='Constraint compat switching must be recorded as effective v0.5 constraint, never as requested v2 certification.'),
        entry('rhofold','_build_rhofold_evaluator',
          [scope('unrelaxed',relax_steps=0),scope('explicit OpenMM relaxation',relax_steps='configured > 0')],
          ['checkpoint','RNA-FM dependencies','seed','relax_steps','openmm_platform/properties and physical binding'],
          'CPU tokens; native model/RNA-FM plus sigmoid/softmax and D2H; CPU PDB/NPZ; relaxation separately admitted',
          ['tests/test_rhofold_resident.py','tests/test_rhofold_evaluator_compat.py'], pins=['vendor/rhofold']),
        entry('dummy','_build_dummy_evaluator',
          [scope('deterministic CPU without structures',write_structure=False),scope('optional PyRosetta structure output',write_structure=True)],
          ['complete Dummy config','actual PyRosetta env if write_structure=true'],
          'CPU owner; no inference weights or GPU', ['tests/test_workflow_benchmark.py'], target='cpu',
          note='Historical real CPU Dummy/MCMC workflow evidence applies only to its recorded no-structure fixture.'),
        entry('rosetta_crystal','_build_rosetta_crystal_evaluator',[scope('configured PyRosetta CPU scoring')],
          ['PyRosetta environment','residue params','reference structure content','scoring/repack/minimize/replicates/seed/mapping'],
          'CPU process environment; reference pose/score cache invalidated by content/config',
          ['tests/test_rosetta_crystal_evaluator_contract.py'], target='cpu',
          note='Fake pose/reference tests do not certify an actual PyRosetta workload.'),
      ],
      'mutator': [
        entry('ligandmpnn','_build_ligandmpnn_mutator',mpnn_scopes,mpnn_freeze,
          'bounded CPU PDB parse cache; native featurize/sample/score device graph; CPU result finalization',
          ['tests/test_mutator_resident_runtime.py','tests/test_ligandmpnn_device_binding.py'],
          note='All five model_type values require their own pretrained checkpoint/config evidence; ligand native_tie is rejected.'),
        entry('grouped_ligandmpnn','_build_grouped_ligandmpnn_mutator',mpnn_scopes,mpnn_freeze+['groups/shared fixed segments'],
          'same private MPNN runner; owner retains grouped probability and candidate selection semantics',
          ['tests/test_mutator_resident_runtime.py','tests/test_mpnn_sync_policy.py']),
        entry('proteinmpnn','_build_proteinmpnn_mutator',
          [scope('protein_mpnn ordinary',model_type='protein_mpnn'),
           scope('ProteinMPNN native joint tie',model_type='protein_mpnn',coupling='native_tie')],mpnn_freeze,
          'same private runner; native joint/autoregressive call stays atomic; owner retains selector',
          ['tests/test_mutator_resident_runtime.py','tests/test_mpnn_sync_policy.py'],
          note='Factory rejects non-protein_mpnn model_type; native_tie rejects probability-changing proposal transforms.'),
        entry('multistateproteinmpnn multistate_proteinmpnn faraparttiedmultitargetligandmpnn far_apart_tied_multitarget_ligandmpnn',
          '_build_multistate_proteinmpnn_mutator',
          [scope('ProteinMPNN legacy far-apart multistate',model_type='protein_mpnn',context_layout='far_apart',
                 coupling='legacy_order_conditioned_marginals')],mpnn_freeze+['reference/layout/chain conditioning'],
          'same native model/decode; owner preserves far-apart ordering and reference-conditioned proposals',
          ['tests/test_multistate_proteinmpnn.py','tests/test_mpnn_sync_policy.py'],
          note='All four aliases select the same ProteinMPNN factory; the LigandMPNN-named aliases do not introduce a ligand checkpoint family.'),
        entry('esm2','_build_esm2_mutator',[scope('esm2_t33_650M_UR50D')],
          ['650M local checkpoint','alphabet','masked positions','position_selector/proposal settings','RNG'],
          'owner tokenize/mask; private resident logits/softmax; CPU probability publication',
          ['tests/test_mutator_resident_runtime.py'],pins=['vendor/esm2']),
        entry('esmif','_build_esmif_mutator',[scope('esm_if1_gvp4_t16_142M_UR50')],
          ['142M local checkpoint','backbone coordinates','fixed masks','joint decoder RNG'],
          'CPU coordinates/masks; native encoder and entire incremental autoregressive decoder remain atomic',
          ['tests/test_mutator_resident_runtime.py'],pins=['vendor/esm2']),
        entry('esm3','_build_esm3_mutator',
          [scope('default',model_name='esm3-open'),scope('explicit local SDK model snapshot')],
          ['model_name/snapshot','SDK/source version','masked prompt','temperature','RNG'],
          'owner DTO; worker native SDK prompt/encode/logits; private model',
          ['tests/test_esm3_mutator.py','tests/test_mutator_resident_runtime.py'],pins=['vendor/esm3']),
        entry('dplm','_build_dplm_mutator',
          [scope('default',model_name='airkingbd/dplm_150m'),scope('explicit DPLM snapshot')],
          ['model_name/local snapshot','external Python/ByProt','sampling_strategy','max_iter','num_samples','temperature','response-pool consumption'],
          'resident child after real ready; native tokenize/diffusion/decode remains composite; finite response-pool semantics unchanged',
          ['tests/test_dplm_lifecycle.py','tests/test_dplm_runner_response_cache.py'],pins=['vendor/dplm']),
        entry('surfdesign','_build_surfdesign_mutator',[scope('configured SurfDesign experiment',checkpoint='best.ckpt')],
          ['experiment/config/checkpoint','PyMOL version','surface_mode','seed','max_iter','strategy','temperature'],
          'CPU PyMOL surface preparation/cache; resident refinement/decoder; CPU result materialization',
          ['tests/test_surfdesign_mutator.py'],pins=['vendor/surfdesign']),
        entry('surfpro','_build_surfpro_mutator',[scope('configured fairseq SurfPro',checkpoint='checkpoint_best.pt')],
          ['checkpoint','MSMS/pdb_to_xyzrn contents','SRU build','max_surface_vertices','seed','logit_temperature'],
          'CPU external surface tools/cache; resident native encoder plus complete autoregressive decoder',
          ['tests/test_surfpro_mutator.py'],pins=['vendor/surfpro','vendor/sru']),
        entry('carbondesign','_build_carbondesign_mutator',[scope('configured CarbonDesign/embedding checkpoints')],
          ['external repo','model_ckpt','model_config','model_features','LM embedding weights','batch_size','temperature','max_seq_len'],
          'resident main model; dataset/embedding transforms may use GPU, so loader/embedding/model stays composite',
          ['tests/test_mutator_additional_lifecycle.py']),
        entry('nampnn','_build_nampnn_mutator',
          [scope(f'native one-shot entropy seed / shared_tokens={shared}',extra_args={'seed':0},na_shared_tokens=shared) for shared in (0,1)] +
          [scope(f'audited resident fixed seed / shared_tokens={shared}',na_shared_tokens=shared) for shared in (0,1)],
          ['checkpoint','exact inference/run.py','na_shared_tokens','design_na_only','parse_na_only',
           'fixed masks/chains','native sample count','extra_args.seed'],
          'version-gated native constructor split; exact post-constructor RNG replay; native joint decode/output unchanged',
          ['tests/test_mutator_additional_lifecycle.py','tests/test_nampnn_mixed_design.py'],pins=['vendor/na_mpnn'],
          note='Resident AST requires inference/run.py SHA256 8d8937cceb84fbfc2e1b49b65e21b730a74e9dafab9e9c6720d1af75bda403d7 and explicit nonzero seed. Seed 0 remains one-shot.'),
        entry('dynamicmpnn','_build_dynamicmpnn_mutator',
          [scope('explicit backend_factory lifecycle'),scope('configured backend_callable'),scope('configured command_template')],
          ['external source/checkpoint','exact callback/CLI API','environment','temperature','num_samples','probability semantics'],
          'private bridge; native joint inference composite; only explicit factory may report observed model/load_count',
          ['tests/test_real_dynamicmpnn_adapter.py','tests/test_dynamic_mpnn_mutator.py'],
          note='Arbitrary callback/CLI route has unknown residency; do not substitute a newer native API or infer pretrained loading.'),
        entry('mcmc','_build_mcmc_mutator',[scope('configured RandomSubstitutionMutator')],
          ['MCMC mutation_method','select_positions','trajectory RNG','mutation count','acceptance/stage protocol'],
          'CPU trajectory owner; one trajectory retains causal serial MCMC decisions',
          ['tests/test_workflow_benchmark.py'],target='cpu',
          note='Historical real CPU Dummy/MCMC workload validated; no GPU or learned mutator.'),
        entry('pore_blocker_mcmc','_build_pore_blocker_mcmc_mutator',
          [scope('PoreBlockingSubstitutionMutator',mutation_method='frequency_adjusted',select_positions='plddt')],
          ['effective override configuration','position probabilities','trajectory RNG and acceptance'],
          'CPU trajectory owner; no model weights/GPU',[],target='cpu'),
        entry('motifbreaker','_build_motifbreaker_mutator',[scope('configured MotifBreakerMutator')],
          ['motif policy','fixed positions','trajectory RNG and acceptance'],
          'CPU mutation implementation; no model weights/GPU',[],target='cpu'),
        entry('xmix','_build_xmix_mutator',[scope('configured XMixMutator')],
          ['mixture/pool inputs','mutation settings','trajectory RNG and acceptance'],
          'CPU mutation implementation; no model weights/GPU',[],target='cpu'),
      ],
    }
    return result


def discover(root, kind):
    relative, decorator = REGISTRIES[kind]
    path = root / relative
    source = path.read_text(encoding='utf-8')
    result = {}
    for node in ast.parse(source).body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for value in node.decorator_list:
            if not (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                    and value.func.id == decorator):
                continue
            if len(value.args) != 1 or not isinstance(value.args[0], ast.Constant) or not isinstance(value.args[0].value, str):
                raise ValueError(f'{relative}:{value.lineno}: dynamic registry name needs manual audit')
            declared = value.args[0].value
            key = declared.strip().lower()
            if key in result:
                raise ValueError(f'duplicate normalized registry key: {kind}/{key}')
            result[key] = {'key':key,'declared_key':declared,'factory':node.name,
                           'source_path':relative,'source_line':node.lineno,
                           'source_sha256':hashlib.sha256(source.encode()).hexdigest()}
    return result


def _git(root, *arguments):
    result = subprocess.run(['git','-C',str(root),*arguments],text=True,capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else None


def _successful_remote_json(result):
    if (result.get('status') != 'exited' or result.get('exitCode') != 0
            or result.get('outputTruncated') is not False):
        raise ValueError('recorded preflight result is incomplete, failed or truncated')
    return json.loads(result['stdout'])


def f102_preflight(root):
    """Read immutable CPU/asset records; never treat readiness as GPU execution."""
    relative = Path('evidence/runtime-next/agent-b/f102-backend-preflight')
    directory = root/relative
    manifest = json.loads((directory/'manifest.json').read_text())
    provenance = {}
    for name, identity in manifest['files'].items():
        data = (directory/name).read_bytes()
        if len(data) != identity['bytes'] or hashlib.sha256(data).hexdigest() != identity['sha256']:
            raise ValueError(f'f102 recorded preflight evidence changed: {name}')
        provenance[str(relative/name)] = identity
    commands = json.loads((directory/'commands.json').read_text())
    compatibility_commands = json.loads((directory/'commands-42-46.json').read_text())

    def command_json(records, command_id, compatibility=False):
        matches = [record for record in records if record['id'] == command_id]
        if len(matches) != 1:
            raise ValueError(f'f102 command identity is missing or ambiguous: {command_id}')
        record = matches[0]
        if compatibility:
            record = record.get('poll_response', record['initial_response'])
        return _successful_remote_json(record['result'])

    cpu = json.loads((directory/'CPU_COMPATIBILITY.json').read_text())
    if (cpu['module_import'] != command_json(compatibility_commands, '45', True)
            or cpu['checkpoint_and_structure'] != command_json(compatibility_commands, '46', True)):
        raise ValueError('f102 CPU summary differs from its actual completed command output')
    parsed_record = json.loads((directory/'gpu-control-parse-true.json').read_text())
    parsed = _successful_remote_json(parsed_record['completed']['result'])
    protocol_path = Path('benchmarks/runtime/workloads/proteinmpnn_gpu_control.json')
    protocol = json.loads((root/protocol_path).read_text())
    weights = cpu['checkpoint_and_structure']
    if (weights['weights_sha256'] != protocol['assets']['checkpoint']['sha256']
            or weights['input_sha256'] != protocol['assets']['pdb']['sha256']
            or weights['missing_keys'] or weights['unexpected_keys']
            or weights['model_forward_calls'] != 0 or weights['cuda_initialized']
            or cpu['module_import']['cuda_initialized'] or parsed['cuda_initialized']
            or parsed['chain_counts'] != protocol['assets']['pdb']['chain_counts']
            or parsed['mask_sum'] != protocol['assets']['pdb']['residues']
            or parsed['mask_values'] != [1]
            or parsed['native_sequence_sha256'] != protocol['assets']['pdb']['native_sequence_sha256']
            or parsed['settings'] != protocol['numerical_settings']):
        raise ValueError('f102 CPU preflight does not match the recorded unsampled control assets/settings')
    observed_sources = {Path(path).name: value['sha256'] for path, value in cpu['module_import']['source'].items()}
    matching = all(observed_sources.get(Path(path).name) == sha
                   and (root/path).is_file() and hashlib.sha256((root/path).read_bytes()).hexdigest() == sha
                   for path, sha in protocol['native_sources'].items())
    main_weight = command_json(commands, '23')
    installation = command_json(commands, '38')
    archives = command_json(commands, '30')
    filesystem = command_json(commands, '27')
    runner = {
        'id':protocol['name'],
        'api':'build_runner("ligandmpnn", model_type="protein_mpnn")',
        'scope':protocol['scope'],
        'registry_certification':False,
        'registry_rows_not_certified':['mutator/ligandmpnn','mutator/proteinmpnn'],
        'comparison':'same candidate legacy wrapper versus existing private resident runner; both reuse one model per trial',
        'protocol':str(protocol_path),'protocol_sha256':hashlib.sha256((root/protocol_path).read_bytes()).hexdigest(),
        'harness':'benchmarks/runtime/gpu_mutator_control.py',
        'harness_sha256':hashlib.sha256((root/'benchmarks/runtime/gpu_mutator_control.py').read_bytes()).hexdigest(),
        'assets':protocol['assets'],
        'cpu_preflight':{
            'status':'DONE' if matching else 'PARTIAL','kind':'real_pretrained_weights_strict_cpu_architecture_load_and_native_input_parse',
            'native_sources_match_checkout':matching,'source_digests':protocol['native_sources'],
            'model_forward_calls':0,'gpu_execution':False,'cuda_initialized':False,
            'checkpoint_keys':weights['checkpoint_keys'],'strict_missing_keys':weights['missing_keys'],
            'strict_unexpected_keys':weights['unexpected_keys'],'state_tensor_count':weights['state_tensor_count'],
            'parameter_count':weights['parameter_count'],'parameter_dtypes':weights['parameter_dtypes'],'num_edges':weights['num_edges'],
            'environment':{key:value for key,value in cpu['module_import'].items() if key!='source'},
            'native_parse':parsed,
            'limitation':'Actual architecture load and parse only; zero forward/sample calls. This is not a completed public-runner constructor/decode, GPU measurement, registry selector/proposal/acceptance test or full workflow.'},
        'planned_gpu_control':{'status':'NOT_RUN','evidence':None,'interleaved_groups':len(protocol['groups']),
            'trials':sum(len(order) for order in protocol['mode_order']),
            'design_requests':sum(len(order) for order in protocol['mode_order'])*protocol['requests_per_trial'],
            'warmup_requests':protocol['warmup_requests'],'budget':protocol['budget'],
            'gate':'Final source/CPU freeze, launcher preparation and one new verified Slurm allocation are still required. Readiness never implies an executed GPU certificate.'},
        'performance_certification':{'status':'BLOCKED','effective_throughput_improvement_proved':False},
        'exclusions':['pretrained evaluator W1','full registry mutator selector/proposal/acceptance','MCMC or stage transitions',
                      'shared broker','other MPNN model types, grouped/native-tie/multistate variants','revision-attributed speedup'],
    }
    return {
        'status':'DONE','kind':'recorded_read_only_asset_and_actual_cpu_preflight','provenance':provenance,
        'backend_execution_certificate':False,
        'esmfold':{'main_checkpoint':main_weight,'installation_metadata':installation,
            'actual_esm_or_openfold_import_executed':False,'pretrained_execution':False,
            'blockers':['Located ESMFold1 main checkpoint is an invalid ZIP (2,097,152,000 bytes; BadZipFile).',
                        'fair-esm/OpenFold distribution metadata and expected audited ESM helper path were absent in the inspected SIF; import usability remains unverified.',
                        'No repaired/downloaded assets, complete source/environment pairing or pretrained ESMFold execution.']},
        'other_assets':{'filesystem_metadata':filesystem,'archive_metadata':archives,
            'qualification':'Directory metadata, readable ZIP directories and recorded sidecar digests only. No fresh full checksum/state-dict/inference validation is inferred, except the separately named ProteinMPNN CPU control.'},
        'runner_control':runner,
    }


def build_matrix(root):
    root = Path(root).resolve()
    specs = catalog()
    inventories = {kind:discover(root,kind) for kind in REGISTRIES}
    records = []
    for kind, groups in specs.items():
        keyed = {key:spec for spec in groups for key in spec['keys']}
        actual = inventories[kind]
        if set(keyed) != set(actual):
            raise ValueError(f'{kind} registry/catalog drift: unreviewed={sorted(set(actual)-set(keyed))}; '
                             f'missing={sorted(set(keyed)-set(actual))}')
        for key in sorted(actual):
            spec = copy.deepcopy(keyed[key])
            if actual[key]['factory'] != spec['factory']:
                raise ValueError(f'{kind}/{key}: factory changed; certification scope needs review')
            for path in spec['available_tests']:
                if not (root/path).is_file():
                    raise ValueError(f'{kind}/{key}: evidence/test path missing: {path}')
            is_gpu = spec['target'] == 'pretrained_gpu'
            record = {**actual[key],**spec,'kind':kind,'status':'BLOCKED' if is_gpu else 'PARTIAL',
                      'registered':True,'registration_is_certification':False,
                      'identity_required':COMMON_IDENTITY,
                      'configuration_state':'NOT_CONFIGURED_FOR_REAL_CERTIFICATION',
                      'pretrained_gpu':{'status':'BLOCKED' if is_gpu else 'NOT_APPLICABLE',
                          'measured':False,'evidence':None},
                      'real_backend_gates':{gate:'BLOCKED' if is_gpu else 'NOT_RUN_IN_THIS_GENERATION' for gate in REAL_GATES},
                      'evidence_kind':'available code/contract tests; not executed by this generator',
                      'source_references':{},
                      'blockers':(['No executed real GPU certificate supplied for this exact registry/configuration scope',
                                   'Complete source/environment/checkpoint pairing has not been certified for this registry scope',
                                   'Real frozen workload numerical/lifecycle/failure/throughput gates not run'] if is_gpu
                                  else ['No current full backend certification is run by this inventory tool'])}
            for path in spec['vendor_gitlinks']:
                value = _git(root,'ls-tree','HEAD',path)
                fields = value.split() if value else []
                record['source_references'][path] = fields[2] if fields and fields[0]=='160000' else None
            if not spec['vendor_gitlinks']:
                record['source_references']['external_or_repository_source'] = 'requires actual configured source digest; no vendor revision invented'
            for variant in record['scopes']:
                variant['certification_status'] = record['status']
                variant['execution_manifest'] = None
            record['performance_certification'] = {'status':'BLOCKED' if is_gpu else 'PARTIAL',
                'effective_throughput_improvement_proved':False}
            records.append(record)
    historical = root/'evidence/runtime-20261103/cpu-three-mode-02/report.json'
    hist = json.loads(historical.read_text())
    historical_evidence = {
        'kind':'historical_actual_cpu_workflow','status':'DONE','certified_scope':'recorded fixture only',
        'path':str(historical.relative_to(root)),'sha256':hashlib.sha256(historical.read_bytes()).hexdigest(),
        'fixture':'scripts/validation/fixtures/runtime_cpu.yaml','complete':hist['complete'],
        'completed_trajectories':sum(row['completed'] for row in hist['runs']),
        'scientific_scores_decisions_certified':hist['correctness']['certified'],
        'coordinate_parity':hist['correctness']['delivered_structure_parity'],
        'limitation':'Recorded Dummy/no-structure + CPU MCMC only; resident modes slower; not this HEAD rerun or learned-model/GPU evidence',
    }
    ci_path = root/'evidence/runtime-next/agent-b/cpu-ci-01-summary.json'
    ci = json.loads(ci_path.read_text()) if ci_path.is_file() else None
    torch_ci_path = root/'evidence/runtime-next/agent-b/torch-ci-01-summary.json'
    torch_ci = json.loads(torch_ci_path.read_text()) if torch_ci_path.is_file() else None
    for record in records:
        if (record['kind'],record['key']) in {('evaluator','dummy'),('mutator','mcmc')}:
            record['historical_evidence'] = historical_evidence
            if ci is not None:
                record['recorded_w0_ci_evidence'] = {
                    'kind':'actual_cpu_process_workflow','status':ci['status'],
                    'summary':str(ci_path.relative_to(root)),
                    'sha256':hashlib.sha256(ci_path.read_bytes()).hexdigest(),
                    'candidate':ci['candidate'],'baseline':ci['baseline'],
                    'protocol_sha256':ci['protocol_sha256'],'runs':ci['runs'],
                    'completed_trajectories':sum(row['trajectories'] for row in ci['counts'].values()),
                    'initial_evaluations':sum(row['initial_evaluations'] for row in ci['counts'].values()),
                    'mutation_attempts':sum(row['mutation_attempts'] for row in ci['counts'].values()),
                    'accepted_mutations':sum(row['accepted_mutations'] for row in ci['counts'].values()),
                    'limitation':'Exact W0 Dummy/no-structure + CPU MCMC at the recorded revisions only; all mutation attempts rejected; initial evaluations are not accepted scientific progress; no pretrained/GPU or other CPU backend upgrade'}
        if record['kind']=='evaluator' and record['key']=='esmfold':
            manifest_path=root/'evidence/runtime-next/agent-b/esmfold-output-run-manifest.json'
            manifest=json.loads(manifest_path.read_text())
            matching=all((root/path).is_file() and hashlib.sha256((root/path).read_bytes()).hexdigest()==sha
                         for path,sha in manifest['source_sha256'].items())
            record['local_output_evidence']={
                'kind':'synthetic_torch_cpu_with_source_extracted_native_helpers',
                'status':'DONE' if matching else 'PARTIAL','source_matches_checkout':matching,
                'pretrained':False,'gpu':False,'result':'63 passed, 1 process test deliberately deselected',
                'manifest':str(manifest_path.relative_to(root)),
                'report':'evidence/runtime-next/agent-b/esmfold-output-report.md',
                'limitation':'Output-only numeric/RNG/handler contract; no actual model, GPU transfer or effective throughput certification'}
            if torch_ci is not None:
                record['actual_cpu_process_evidence'] = {
                    'kind':'real_cpu_processes_with_fake_model_and_synthetic_local_checkpoint',
                    'status':torch_ci['regressions']['status'],
                    'summary':str(torch_ci_path.relative_to(root)),
                    'sha256':hashlib.sha256(torch_ci_path.read_bytes()).hexdigest(),
                    'head':torch_ci['head'],'run_id':torch_ci['run_id'],'job_id':torch_ci['job_id'],
                    'torch':torch_ci['torch'],'test_cases':torch_ci['regressions']['test_cases'],
                    'skips':torch_ci['regressions']['skips'],'required_suites':torch_ci['regressions']['required_suites'],
                    'limitation':'65 CPU tests including real resident process reuse and constructor precision/RNG; fake ESM model/weights and synthetic native outputs; no pretrained/GPU or throughput certification'}
    preflight = f102_preflight(root)
    for record in records:
        if record['kind']=='evaluator' and record['key']=='esmfold':
            record['f102_asset_preflight'] = preflight['esmfold']
            record['blockers'] = preflight['esmfold']['blockers'] + record['blockers']
        if record['kind']=='mutator' and record['key'] in {'ligandmpnn','proteinmpnn'}:
            record['related_runner_preflight'] = {
                'control_id':preflight['runner_control']['id'],'registry_certification':False,
                'limitation':'Actual protein_mpnn CPU architecture/input compatibility only; public runner GPU control has not executed and does not cover this registry selector.'}
    return {
        'schema_version':2,'head':_git(root,'rev-parse','HEAD'),
        'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'Builtin factories only; external plugin registrations/configurations require separate enumeration and certification',
        'status':{'inventory':'DONE','pretrained_gpu_certification':'BLOCKED','full_backend_runtime_certification':'PARTIAL'},
        'counts':{kind:{'registry_keys':len(inventories[kind]),
                        'factories':len({r['factory'] for r in inventories[kind].values()})} for kind in REGISTRIES},
        'registry_status_counts':{state:sum(record['status']==state for record in records) for state in ('DONE','PARTIAL','BLOCKED')},
        'interpretation':'This command validates inventory and evidence provenance only. It never upgrades a registered key, alias, CPU fixture, or source pin into pretrained/GPU certification.',
        'f102_asset_preflight':{key:value for key,value in preflight.items() if key!='runner_control'},
        'runner_controls':[preflight['runner_control']],
        'records':records,
    }


def markdown(matrix):
    lines = ['# Runtime backend certification matrix','',
        f"Registry inventory HEAD: `{matrix['head']}`. Generated by `scripts/validation/runtime_backend_certification.py`.",'',
        '**DONE:** all builtin registry keys are enumerated with factory/variant/config scope. **PARTIAL:** CPU and synthetic contract evidence exists only for the explicitly named scopes. **BLOCKED:** every learned/pretrained GPU registry scope lacks its complete real numerical/lifecycle/performance certificate in this evidence set. Located assets and a CPU-compatible pretrained checkpoint do not constitute GPU execution.','',
        'Generation does not execute any backend. Vendor pins identify source references; actual installed files, local patches, checkpoints and environment must be fingerprinted for a real run. Alias rows share a factory, not a broader scientific certification.','',
        '## Exact registry coverage','',
        '| Kind | Keys | Factories |','|---|---:|---:|']
    for kind,count in matrix['counts'].items():
        lines.append(f"| {kind} | {count['registry_keys']} | {count['factories']} |")
    counts=matrix['registry_status_counts']
    lines += ['',f"Registry status: **{counts['DONE']} DONE / {counts['PARTIAL']} PARTIAL / {counts['BLOCKED']} BLOCKED**. These counts do not include narrower preflight or contract scopes."]
    for kind in REGISTRIES:
        lines += ['',f'## {kind.title()} matrix','',
                  '| Actual normalized key | Variant/configuration scope | Execution boundary | Real backend status |',
                  '|---|---|---|---|']
        for r in matrix['records']:
            if r['kind'] != kind: continue
            variants='; '.join(v['variant'] for v in r['scopes'])
            state='BLOCKED: pretrained/GPU' if r['target']=='pretrained_gpu' else 'PARTIAL: CPU only; GPU N/A'
            if 'historical_evidence' in r: state+='; recorded CPU fixture DONE'
            if 'recorded_w0_ci_evidence' in r: state+='; exact W0 CPU CI DONE'
            if 'local_output_evidence' in r: state+='; synthetic native-output CPU contract '+r['local_output_evidence']['status']
            if 'actual_cpu_process_evidence' in r: state+='; CPU fake-weights process CI DONE'
            lines.append(f"| `{r['key']}` | {variants} | {r['boundary']} | {state} |")
        lines += ['', '### Configuration and evidence limits','']
        for r in matrix['records']:
            if r['kind'] != kind or r['key'] != r['keys'][0]: continue
            keys=', '.join('`'+key+'`' for key in r['keys'])
            details='; '.join(r['configuration_to_freeze'])
            refs='; '.join(f'`{path}` @ `{sha}`' for path,sha in r['source_references'].items())
            lines.append(f"- {keys}: freeze {details}. Source: {refs}. {r['note']}")
    control=matrix['runner_controls'][0]
    cpu=control['cpu_preflight']
    lines += ['', '## f102 actual assets and CPU preflight','',
        'Immutable commands, failed discovery attempts, full bounded responses, summaries and file digests are preserved under `evidence/runtime-next/agent-b/f102-backend-preflight/`. The generator verifies their digest manifest and checks the CPU summary against the completed native command outputs. These are recorded observations, not commands executed by matrix generation.','',
        '| Scope | Actual evidence | Certification limit |','|---|---|---|',
        '| ESMFold1 | Located folding checkpoint is 2,097,152,000 bytes, has no readable ZIP directory and raises `BadZipFile`. `fair-esm`/`openfold` distribution metadata and the expected audited ESM helper path were absent in the inspected SIF. | W1 remains BLOCKED. Actual `import esm`/`import openfold` was not tested; absence of metadata is not an observed import exception. No repair/download or pretrained inference. |',
        f"| ProteinMPNN v48_020 | Real checkpoint freshly hashed; strict CPU architecture load accepted {cpu['state_tensor_count']} state tensors, {cpu['parameter_count']:,} float32 parameters, zero missing/unexpected keys. | Zero forward/sample calls; CUDA remained uninitialized. Architecture compatibility is not full public-runner or registry certification. |",
        '| 4GYT input | Full byte-identical 531,114-byte PDB, A177 + B177; native CPU parse with `parse_atoms_with_zero_occupancy=True`, input mask all ones, frozen native sequence digest and numerical settings. | 354 residues retained. The all-ones design mask follows the explicit full A/B design configuration; native returned design output does not itself contain a mask. |',
        '| AF2 / AF3 / Protenix / Boltz / ESM trunk | Existing path/size metadata, selected ZIP-directory readability and historical sidecar digests are recorded. | No new full checksum/state-dict/inference certification; installed Boltz 2.2.1 metadata does not add a Boltz2 registry loader. |','',
        '### Separate public-runner control: GPU NOT_RUN','',
        f"`{control['api']}` is the sole planned runner API. Protocol: `{control['protocol']}` (SHA-256 `{control['protocol_sha256']}`). It compares the same candidate's existing legacy wrapper with its existing private resident runner; both construct once and reuse the model for three sequential requests. This does not certify the complete registry `ligandmpnn` or `proteinmpnn` scientific selector/proposal/acceptance path.",'',
        'The fixed plan contains five alternating groups, seeds 17–21, ten model constructions, thirty requests and no warmup; complete 4GYT AB354, temperature 0.3, omit C, batch 1×1, all native kwargs and scientific gates are explicit. The maximum allocation is one GPU, four CPUs, 32 GiB, 900 seconds / 0.25 GPU-hours. Final source/CPU freeze, launcher preparation and a newly verified Slurm allocation are still required. No successful GPU result or speedup is recorded here.','',
        'The task-specific entrypoints are documented in `benchmarks/runtime/gpu_mutator_control.md` and `benchmarks/runtime/gpu_slurm_launcher.md`. Later raw GPU evidence can qualify only this exact runner control. Evaluator W1, shared broker, MCMC/stage transitions, other model types, grouped/tied/multistate behavior and revision-attributed speedup remain excluded.','',
        '## Evidence interpretation','',
        '- The preserved actual CPU Dummy/MCMC report is `evidence/runtime-20261103/cpu-three-mode-02/report.json`: 27 completed trajectories, exact recorded score/decision agreement, no structure-coordinate certificate. Resident modes were slower. It is historical evidence, not a current HEAD or GPU rerun.',
        '- The task W0 CPU CI summary is `evidence/runtime-next/agent-b/cpu-ci-01-summary.json` (canonical source under `evidence/runtime-next/ci/`). Candidate 6620da6 versus baseline d66 ran 30 trials / 90 trajectories / 270 evaluations in five interleaved groups, with confirmed release. Its 90 initial evaluations are not accepted scientific progress: all 180 mutation attempts were rejected. This only updates the exact Dummy/MCMC W0 scope; no structure-coordinate, learned-model, other CPU-backend or performance certificate is inferred. The full raw archive is delivered by the main integration report.',
        '- ESMFold1 output evidence is `evidence/runtime-next/agent-b/esmfold-output-report.md`: real CPU torch tensors and pinned native helper excerpts, exact PDB/score/RNG checks, synthetic inputs, no pretrained inference. Requested host-payload reduction is not actual GPU transfer measurement or effective throughput. Independent C output-only CPU timings were slower by 9.31% / 6.68% for the two fixed shapes (`evidence/runtime-next/agent-c/review-b-output-comparison.json`); negative results are retained. KEEP applies only to resource lifecycle/boundary correctness and requested payload reduction; performance certification remains BLOCKED.',
        '- The ESMFold CPU process CI at `db937f9706270cb109e799cda29f52d2872726b5` completed all four focused suites: 65 passed, zero skips/deselections, torch 2.7.1+cpu with cuda_build=null. `evidence/runtime-next/agent-b/torch-ci-01-summary.json` records source/run/job/artifact identity. This closes the local process-test limitation for the recorded fake-model/synthetic-checkpoint scope, including one resident load and constructor precision/RNG behavior. It does not certify a pretrained ESMFold model or GPU.',
        '- Existing MPNN contract tests use an actual CPU architecture with synthetic checkpoints; the separate f102 readiness record above uses the actual pretrained v48_020 checkpoint but performs no sampling. Neither certifies a real GPU registry workload. NA-MPNN/Surf/DPLM lifecycle fixtures and mock loaders do not certify their pretrained variants. Tests listed in JSON are available verification entry points, not proof this generator executed them.',
        '- Each real variant must pass single request, continuous requests/load count, device boundary/residency, cache invalidation, numerical/PDB parity, RNG/proposal/acceptance/stage parity, failure/recovery, and repeated fixed-workload effective throughput. Every exact seed/sample/recycle/precision/constraint setting remains frozen.',
        '- Boltz is Boltz1; no Boltz2 is registered. Mol-1 has independent aliases and is not certified by Protenix. Protenix requested/effective constraint model must be recorded separately. Legacy FarApartTiedMultitargetLigandMPNN aliases require protein_mpnn. NA-MPNN seed 0 remains one-shot.',
        '', '## Regenerate without models or GPU','', '```bash',
        'python scripts/validation/runtime_backend_certification.py --output /tmp/runtime-backend-certification.json --markdown /tmp/runtime-backend-certification.md',
        '```','',
        'The tool fails on registry/catalog drift, changed factory ownership or missing named evidence/test files. JSON records exact factory locations, source digests, variants, required configuration, pins, evidence classification and all unrun gates. It does not download weights, use shared GPUs or change configuration.','',
        '## Real backend verification entry','',
        'After selecting an existing supported configuration and provisioning an explicitly authorized allocation with valid local pretrained assets, use the repository workflow benchmark. W1 ESMFold1 remains blocked by the located invalid folding checkpoint and unverified usable ESM/OpenFold installation; a source pin or output fixture is not a model installation.','',
        '```bash',
        'python -m astrolabe --config-dir config run --standalone /absolute/path/frozen-workload.yaml --no-chdir',
        'python scripts/validation/runtime_workflow_benchmark.py --config /absolute/path/frozen-workload.yaml --config-dir config --out /absolute/path/new-empty-benchmark-directory --repetitions 5 --walltime-budget APPROVED_SECONDS --require-gpu',
        '```','',
        'Inherit the actual approved allocation, replace APPROVED_SECONDS with its real budget, and record every effective config/source/weight/environment identity. Planning-only output, successful imports, mock transport or a CPU speed result must not mark these GPU rows DONE.','']
    return '\n'.join(line.rstrip() for line in lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output',type=Path)
    parser.add_argument('--markdown',type=Path)
    args=parser.parse_args()
    try:
        value=build_matrix(args.repo_root)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f'certification inventory rejected: {exc}\n')
    encoded=json.dumps(value,indent=2,sort_keys=True)+'\n'
    if args.output: args.output.write_text(encoded,encoding='utf-8')
    else: print(encoded,end='')
    if args.markdown: args.markdown.write_text(markdown(value),encoding='utf-8')

if __name__=='__main__': main()

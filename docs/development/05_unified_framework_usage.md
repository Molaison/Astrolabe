# 统一框架目标接口与用法草图（非已实现）
这是完整目标的用法草图，未表示全部科学路径已接通。已有可运行内核和 CPU
组合示例见 [当前实现](06_unified_framework_implementation.md)；现有常规入口仍可用。

Proposed signatures only; no runtime or backend qualification. `04_unified_framework_sketch.py` is canonical. CallKey is journal-private; callers never construct it. Existing `Astrolabe(cfg).run()` remains the ordinary entry. All scientific settings below are illustrative, not defaults.

## Short ordinary use
```python
from astrolabe import Astrolabe
Astrolabe(composed_cfg).run()  # selected initializer/evaluator/mutator + stage
```
```sh
astrolabe run -t monomer --dry-run
astrolabe run -t monomer
```
Existing task composition selects its components and stage; dry-run shows selected effective options without loading models. The proposed framework adds no required operation graph or transport setup to this call.

## Advanced explicit transform/conditioning composition
```yaml
system: case.yaml
initializer: {type: structure, source: input.cif}
components:
  fold: {family: evaluator, type: alphafold3, options: {num_recycles: 3}}
  relax: {family: processor, type: rosetta_minimize}
  choose: {family: artifact_selection, type: indexed}
  guess: {family: conditioning, type: af3_initial_guess}
  sites: {family: selector, type: interface}
  pack: {family: packing, type: mapped_structure, options: {ties: ties.json}}
  design: {family: mutator, type: ligandmpnn}
  quality: {family: measurement, type: confidence}
  rank: {family: objective, type: expression, options: {expression: mean_confidence}}
sampling: {draw: {temperature: 0.2, mutation_count: 1}}
stages: [{trial: {type: repeat_design, rounds: 2, components: [fold, relax, choose, guess, sites, pack, design, quality, rank]}}]
search: {type: beam, width: 4, evaluation_budget: 20}
report: {columns: [candidate, parents, stage, objective, accepted]}
```
Descriptors validate this combination against the actual selected adapters before loading: a missing packing/conditioning capability rejects the recipe. Configuration does not assert these proposed seams already exist. `repeat_design` is reusable composition, not a task-specific engine. Preserve each archived recipe's own ordering/options/accounting.

## Concrete typed custom composition
The example below specifies the operation chain; signatures are in 04_unified_framework_sketch.py. An owning conditioning plugin defines `InitialGuessRequest(candidate: Candidate, artifact: Artifact)` and registers `Operation[InitialGuessRequest, Conditioning]`. Its bind checks atom/chain mapping and records the actual mechanism; it is not a global intent variant. Configuration lowering supplies `system`, `parts`, `runtime`, `initial`, policy and protocol values, with fixed assembly/tie/weight/sample definitions (this example requires one assembly, one artifact, one packed state with weight 1, and one proposal). All aliases used below are declared above.
```python
# Within a registered Trial, using 04_unified_framework_sketch.py types; runtime objects are injected.
def op(alias: str, request_type: type[I], result_type: type[O]) -> Operation[I, O]:
    return parts.operation(alias, request_type, result_type)
fold = op('fold', PredictRequest, Evidence)
relax = op('relax', TransformRequest, ArtifactBank)
choose = op('choose', ChooseArtifactRequest, Artifact)
guess = op('guess', InitialGuessRequest, Conditioning)
sites = op('sites', SelectRequest, Selection)
pack = op('pack', PackRequest, PackedStates)
design = op('design', ProposeRequest, Proposals)
quality = op('quality', MeasureRequest, Measurement)
rank = op('rank', AssessRequest, Assessment)
requirements = parts.requirements(('fold','relax','choose','guess','sites','pack','design','quality','rank'))
def evaluate(label, candidate, conditioning, sample):
    e = runtime.call(label+'/predict', fold, PredictRequest(system, candidate, assemblies, conditioning, sources, (sample,)))
    m = runtime.call(label+'/measure', quality, MeasureRequest(system, candidate, (EvidenceUse(e, 'current', candidate.identity),), ()))
    e = e.with_measurements((m,))
    a = runtime.call(label+'/assess', rank, AssessRequest(candidate, e, runtime.history, objective_revision))
    return Scored(candidate, e, a)
# Canonical run calls runtime.preflight(requirements) once BEFORE root/model load.
# Runtime binds root work/history; Trial.root calls evaluate("root", candidate, (), root_sample).
# Trial.perform(work, runtime) contains this deterministic bounded composition:
current = work.parents[0]
proposals = []
for round_index in range(2):
    label = f'design/{round_index}'
    base = EvidenceUse(current.evidence, 'base_guidance', current.candidate.identity)
    bank = runtime.call(label+'/relax', relax, TransformRequest(current.evidence.artifacts[0]))
    artifact = runtime.call(label+'/choose', choose, ChooseArtifactRequest(bank, 0))
    selected = runtime.call(label+'/select', sites, SelectRequest(system, current.candidate, (base,)))
    packed = runtime.call(label+'/pack', pack, PackRequest((artifact,), (selected,), ties, state_weights))
    proposed = runtime.call(label+'/propose', design, ProposeRequest(system, tuple(p.candidate for p in work.parents), current.candidate, (base,), (selected,), packed, parts.sampling('draw'), round_samples[round_index]))
    # Explicit single-child policy; bank/fanout recipes use an owning selection operation.
    proposal, = proposed.items
    proposals.append(proposal)
    conditioning = runtime.call(label+'/condition', guess, InitialGuessRequest(proposal.candidate, artifact))
    current = evaluate(label, proposal.candidate, (conditioning,), round_samples[round_index])
# Return TrialOutcome(work.identity, 'completed', tuple(proposals), (current,), attempt_ids).
# Runtime supplies attempt_ids from this work's journal; no manual reserve/tell here.
```
Configuration supplies root/round sample identities and validates artifact multiplicity, assemblies, packing weights/ties, and proposal cardinality; never infer these from first-result convenience. `artifacts[0]` is an explicit configured single-artifact restriction here, not a framework rule. A multi-state recipe selects/packs its full bank explicitly. Runtime history is the fixed work-entry snapshot.

## Measurement and extension semantics
`measurement_view(metric, runtime.data).between(left, right, left_axis='left_site', right_axis='right_site').reduce(reduction)` selects ordered mapped axes. Unique names distinguish equal kinds; subject/mapping mismatch fails. Partial/unavailable data uses explicit status/masks, never zero. Region-in-complex selection differs from extracting a region and recomputing its SASA. Raw measurements remain immutable; reductions produce derived identity/provenance.
New operations register descriptors, typed request/result codecs, requirements and private adapters at their actual family owner. Adding a chemical conditioner or gradient method does not edit Runtime, Trial or an Intent enum. A real gradient adapter owns live tensors and bounded lifetime internally; absent capability fails preflight. No speculative universal gradient session is exposed. Independent tables have shared typed sampling operations; joint/AR/direct proposals return Proposals. Packing, probability mixture and log fusion are distinct registered operations, with applied-transform identity and decoding-boundary checks.

## Two-round crash/resume contract
1. Preflight records every possible component/capability and codecs before loading. Root, work-entry candidate/history/RNG and policy checkpoint are durable.
2. Round 0 completes: each named call durably commits typed output and RNG receipt before returning. Budget reservations/issue/terminal transitions share one runtime owner.
3. Round 1 prediction is issued; process dies before terminal commit. Journal says unknown; its protocol budget remains charged/reserved. Resume repeats deterministic control flow from the work-entry checkpoint, reuses completed outputs and applies their recorded post-RNG states.
4. The same round-1 prediction key blocks, never resends. Runtime reconciliation may accept verified terminal output. Otherwise it remains unknown; explicit retry creates a new linked attempt under the retry policy, preserving the original. A reused work/round slot with different prepared inputs fails identity validation.
5. Policy tell/checkpoint applies each work outcome once. Trial does no external I/O or randomness outside registered operations. Codecs, trial/version/options, scientific contract, mappings, sources and seeds are resume-validated. This is operational replay of control flow, not inference replay.

## Full-family disposition and output
Existing initializer, processor/hook, selector, conditioning, evaluator, mutator, loss/per-residue and policy owners supply typed operations/descriptors. Existing dispatch-only families gain owning registrations; discovery derives from them. Sampling/fusion/packing/reducers have scientific owners, not core task branches. Model daemons/execution adapters and installation/provider catalogs remain below these interfaces; no second scheduler or inventory. Input transforms return typed PredictRequest variants at their owner; coordinate transforms return ArtifactBank; selectors return mapped Selection; metrics return Measurement; objective/gate operations return Assessment or their own typed results. Every existing family gets a migration disposition, including those absent from this example.
Candidate changes invalidate affected coupled-assembly evidence for the new assignment, not history, model weights or source resources. Unchanged projected inputs require explicit evidence rebinding; replicates stay distinct. Objective/history revisions refresh aggregates/gates. Relaxed coordinates retain artifact lineage; original confidence is never relabeled. Public app/CLI and pruned effective config remain; default summaries show selected scalars, detailed records reference arrays. Migrate actual callers to one controller and remove duplicate loops per slice; keep scientific compatibility adapters until their last consumer migrates.

## Constructible data and option ownership
`SelectedOptionsSchema(owner, version, json_schema, effective_options_json)` carries readable schema and validated selected options directly. The owning registry supplies it without storing a file or loading a model; dry-run displays this object. No schema digest lookup is necessary.
`Codec[T]` belongs to the typed family owner and encodes canonical bytes, decodes them and enumerates referenced Blobs; codec identity includes format version. `DataStore` alone hashes/stores/resolves those bytes. Operation descriptors carry request/result codecs; system/domain/chemistry/index-map/array/RNG owners register their own codecs in that same store's codec resolution, not a second component catalog.
For example, configuration construction receives typed `allowed_values` and `topology` objects with their owner codecs: `restriction_ref = data.put(allowed_values, allowed_values_codec)`; `domain = Domain(variable_id, fixed_token_indices, fixed_atom_ids, restriction_ref)`; `topology_ref = data.put(topology, topology_codec)`. Build `System(system_key, variables, occurrences, topology_ref, (domain,))` with its validated canonical identity; occurrence maps and chemical graphs follow the same put/get path. No caller fabricates an opaque reference without supplying its encoded value or resolving an existing resource.
`put` is memory-only; metadata inspection/dry-run never calls `persist`. Before committing a completed call or checkpoint, Runtime persists the result/receipt and their referenced closure, then commits journal references. Reload checks digests and registered codec versions; unresolved references/codecs fail rather than return fabricated values. `measurement_view(metric, data)` resolves mapped arrays through that owner. Blob paths/transport remain private; no DSL or new scheduler is introduced.

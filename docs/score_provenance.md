# Strict score provenance

Legacy score contribution flattening and expression evaluation retain their existing behavior. A contribution only enters configured strict evidence queries if its producer explicitly supplies `measurement_snapshot`, `input_snapshot`, `subject`, and `relation`. Omitted provenance fails closed. The IDs identify actual producer inputs and measurements; the query consumer never stamps a contribution with the evaluator result currently being scored.

A structure measurement uses `relation: measured_on`, `subject: predicted_structure` or `reference_structure`, and the same concrete identity for `measurement_snapshot` and `input_snapshot`. A mutator's proposal distribution evaluated using an accepted/reference structure uses `subject: proposal`, `relation: proposal_on_reference`, and identifies that reference in `input_snapshot`. Its measurement snapshot is supplied by the producer and must not be inferred from a later prediction. `ScoreQuery` defaults to `predicted_structure` and `measured_on`; reference measurements and proposal distributions require the corresponding explicit query subject and relation as well as the correct snapshot.

The decision owner stores immutable runtime records by their producer-provided measurement identity, without maintaining a second accepted pointer. A repeated identical record may be reused; conflicting evidence for the same measurement fails. `StandardScores` evaluator fields remain captured under that assessment's ID. The code validates identities and relations but cannot establish whether a producer told the truth about what it measured: adapters must populate the metadata from their actual input and measurement. Existing producers without that metadata remain available to legacy expressions but are not yet qualified for strict evidence queries.

`StandardScores.per_proto_scores` is exposed as `per_proto_score` with `observation` and `design_position` axes; named `per_proto_score_terms` are exposed as `per_proto_term_<name>`. The canonical position labels are `<protomer>:<one-based-position>`, and selectors can use `mapping: auto`. These already-aggregated vectors do not retain physical chain, context, or sample provenance, so the adapter does not invent those labels. When several vectors are present, selecting one observation or supplying explicit observation weights is required.

For cross-context position selection, use `observations` instead of a single `query` and explicitly choose all three aggregation dimensions:

```yaml
loss:
  score_selector_queries:
    shared_interface:
      observations:
        - context: AB
          weight: 1.0
          query:
            name: pae
            source: evaluator
            entity: AB
            entity_scope: oligo
            axes: [position]
            select:
              position: ["AB:A:1:A", "AB:A:2:A"]
              target_position: ["AB:B:1:B"]
          mapping: auto
        - context: AC
          weight: 2.0
          query:
            name: pae
            source: evaluator
            entity: AC
            entity_scope: oligo
            axes: [position]
            select:
              position: ["AC:A:1:A", "AC:A:2:A"]
              target_position: ["AC:B:1:C"]
          mapping: auto
      aggregation: {reduction: mean, count: context, missing: error}
      comparison: lt
      threshold: 5.0
```

Use `score("shared_interface")` as the mutator position selector. Each observation performs its own directed query before mapping onto SegmentManager variables; pair matrices are never stacked or symmetrized. `count: context` reduces repeated physical copies within an observation before applying its context weight; `count: physical` counts each weighted physical occurrence. Repeated copies require explicit `position_weights` covering the observation's labels, or `position_weights: equal`. `missing: error` requires every positively weighted context to cover the same canonical variables; `missing: ignore` divides only by weights actually present for that variable. `reduction: sum` returns the weighted sum instead of dividing. Zero-weight contexts do not affect the denominator. All source evidence must still pass its own snapshot, subject, axis, and finite-value checks.

An entity can name both an oligomer and a protomer (for example `A`). `ScoreSpec.entity_scope` and the optional `ScoreQuery.entity_scope` distinguish `oligo` and `protomer`; filtering an ambiguous unscoped record fails closed. Unscoped position records without enough labels to filter a subset also fail closed rather than retaining positions from another context.

When a merged accepted assessment carries immutable records whose snapshots are their original per-oligomer assessment IDs, set `query.snapshot: member` on each observation. The selector resolves that value through the actual accepted assessment's `member_assessment_ids[context]`; it does not relabel a member record as the aggregate. Missing members, a non-accepted aggregate, or mismatched context fail closed. Use the default `last_accepted` for records measured directly on the aggregate assessment.

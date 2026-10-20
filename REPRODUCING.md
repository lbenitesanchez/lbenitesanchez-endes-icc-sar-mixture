# Reproducing the scientific analysis

For the current English table-and-figure workflow, start with [simplified/README.md](simplified/README.md). Use `simplified/verify_inputs.py` followed by `simplified/run_all.py --output-root ../endes-mixture-outputs`. The 35 reviewed CSVs in `simplified/inputs/` are selected by default. This regenerates artifacts from certified derived inputs; it does not fit models. The instructions below cover the complete source-data preparation, model-fitting, correction and scoring pipeline.

Run from the repository root. All survey records, auxiliary files, native fits, sampler state and generated output must be outside this checkout. The published protocols describe the archived analysis; a new run must retain its own source, executable and input hashes rather than pretending to be the archived execution.

## 1 External inputs and the original runs

Read `reproduction/DATA_INPUTS.md`. The workers require a private reference study with the historical scientific freeze files, auxiliary template, geography, real-data basis and input/reference checksums. The simplified CSV bundle does not supply those original arrays or complete fitting reference trees. Do not execute the historical `comparison_setup.py` preservation step against this checkout: that legacy step also expects the excluded editorial archive. Use the recorded external scientific reference tree.

The original model, generation and scoring interfaces are documented by:

```sh
python reproduction/compile_runtime.py --help
python reproduction/sikov_worker.py --help
python reproduction/real_portable_prepare.py --help
python reproduction/real_portable_worker.py --help
python revision_20261015/code/numerical_cohorts_portable.py --help
```

Order: materialize/hash-check the reference inputs; compile the unchanged Stan model; reconstruct real inputs; generate development and evaluation populations; complete development and freeze the 75/25 recipe; fit all planned principal, actual-data, graph and B300 cases; complete the sparse57 and development NUTS cohorts. Keep evaluation truth separate from fitting inputs. The portable cohort runner supports a new private `--protocol` registration with the same scientific constants and current source/executable hashes.

## 2 Refit REML and bootstrap with the recorded models and seeds

The correction inventory expects this fixed private layout:

```text
PRIVATE_RUN_ROOT/
  iaos_reproduction_2026-10-13/sikov/
  iaos_revision_2026-10-15/numerical_cohorts/
```

These must be complete runs, including model inputs, original prediction/status files and unchanged ICC/NUTS components. `PRIVATE_RUN_ROOT` is their configurable parent. Select a new external `CORRECTION_ROOT` and never overwrite the archived runs.

```sh
python revision_20261015/code/reml_correction/refit_cohorts.py inventory --help
python revision_20261015/code/reml_correction/refit_cohorts.py run --help
python revision_20261015/code/reml_correction/resolve_refit_stages.py --help
```

Inventory binds inputs, method, graph, B and endpoint seeds. Dispatch stage one for `principal_256`, `development_32`, `sparse57`, `observed_graph_64`, `B300_subset_8`, `real_temporal` and `real_territorial`. Register stage-two selection using only failed central certificates/rejected bootstrap refits; freeze sources before dispatch. Run the final `refit_cohorts_stage2_v2.py` only for that complete selection. Compare stages, resolve the native tree, and run both `verify_corrected_fits.py` and `verify_refit_cohorts.py` before locking predictions. Retain all planned populations and original random streams.

Stage-one and stage-two protocols are in `revision_20261015/protocols/reml_correction_20261018.json` and `reml_correction_stage2_20261018.json`. The wrappers, validators and stage-resolution CLIs contain the executable argument contracts. No outcome scoring belongs before the complete native prediction lock.

## 3 Mixtures, scoring and numerical tables

Use the frozen preparation adapter for `prepare` and `verify-lock`. Use the final `recalculate_corrected_results.py` for `score` and `export`; the preparation snapshot's historical score action is not the final selection adapter. `score_eligibility_adapter_amendment_20261018.json` records the original five-candidate restriction.

Every action needs `--repo`, `--old-root`, `--old-numerical-root`, `--corrected-root`, `--native-lock` and `--output`. Scoring additionally needs `--reference-root`, `--exe` and `--cmdstan`. Supply the external archived/reference trees; use fresh output directories. Seal all prediction components and mixture distributions before opening outcomes. ICC/NUTS predictions are reused without changing their draws. Score all populations, retain numerical diagnostics and keep Holm3/Holm12 unchanged.

`export` also needs the historical aggregate reference tree materialized under the `expected/current_freeze/data` and revision/reference paths expected by the code. The complete original aggregate reference trees are not supplied by the simplified bundle. The code-only numerical contract preserves the original table matrices but returns empty narrative fields; it does not build documents. `sikov_score.py` and `real_score.py` generate the main numerical CSVs before this formatting/export step.

Run final `audit_corrected_scores_v3.py` against the completed private prediction/scoring trees and exported numerical layer. Its synthetic fixture validator is `validate_score_auditor_v3.py`. Keep all scoring and verification receipts external to this checkout. Stationarity certification is not proof of a global optimum or model adequacy.

## 4 Code-only compatibility

The model/scoring mathematics and retained table functions are unchanged. Paths, private-reference routing, editorial omissions and numerical-only formatting extraction are explicit adaptations in `provenance/code_only_sources.json` and the associated parity receipts. Unknown source changes are rejected. The repository includes only the 35 reviewed reproduction CSVs described in [simplified/INPUTS.md](simplified/INPUTS.md), in addition to the sources and metadata. It contains no manuscript, supplement document, cover letter, raw survey records or chains.

## Historical supplement results

The simplified interface covers article Tables 1–5/Figures 1–3 and supplement Tables S1–S9/S23/Figures S1–S2. The **13 remaining tables and 4 figures** belong to historical supporting analyses. They have no `simplified/.../generate.py` entry point and are excluded from `simplified/run_all.py`.

Use the following original sources to locate their computations. These are pipeline components requiring a prepared external study and reference tree, rather than standalone table generators. Follow the legacy stages and argument contracts in [reproduction/run_all.py](reproduction/run_all.py), [reproduction/legacy_runner.py](reproduction/legacy_runner.py) and [external input requirements](reproduction/DATA_INPUTS.md). Historical CUT8 and benchmark results retain their original roles; they do not add validation populations to the current mixture analysis.

| Current supplement number | Original computation source |
| --- | --- |
| Tables S10, S13 | [sim_design_diagnostics.py](src/estudio_publicacion/fase2_2026-10-07/scripts/sim_design_diagnostics.py) |
| Tables S11, S14, S15 | [validacion_score_external.py](src/estudio_publicacion/fase2_2026-10-07/scripts/validacion_score_external.py) |
| Tables S12, S18 | [phase2_application_summary.py](src/estudio_publicacion/fase2_2026-10-07/scripts/phase2_application_summary.py) |
| Tables S16, S17 | [diseno_fit_cut.py](src/estudio_publicacion/fase2_2026-10-07/scripts/diseno_fit_cut.py) and the CUT8/CUT16 comparison stage |
| Tables S19–S21 | [benchmark_finalize.py](src/estudio_publicacion/scripts/competitors/benchmark_finalize.py) |
| Table S22 | [benchmark_spatial_evaluate.py](src/estudio_publicacion/scripts/competitors/benchmark_spatial_evaluate.py) |
| Figures S3–S6 | [phase2_figures.py](src/estudio_publicacion/fase2_2026-10-07/scripts/phase2_figures.py), routed by [legacy_figures.py](reproduction/legacy_figures.py) |

The numerical table formatting is in `supplement_tables` in [reproduction/article_coverage.py](reproduction/article_coverage.py). The legacy figure wrapper uses older labels; use the figure stems to match the current numbering:

| Current figure | Original figure stem | Legacy wrapper label |
| --- | --- | --- |
| Figure S3 | `design_diagnostics` | `Figure_A1` |
| Figure S4 | `external_contrasts` | `Figure_A2` |
| Figure S5 | `application2025_ICC_capitals` | `Figure_A3` |
| Figure S6 | `application2025_capitals` | `Figure_S1` |

# ENDES ICC–SAR mixture

Code for estimating district childhood anaemia and stunting in Peru with a fixed **75% ICC beta + 25% full-graph SAR** distributional mixture. The current workflow includes the same-model REML and parametric-bootstrap corrections used in the revised analysis.

**Start with the [English table-and-figure guide](simplified/README.md).** It provides one entry point for each current result and reuses the original statistical modules.

## Contents

| What you want to do | Start here |
| --- | --- |
| Regenerate a particular table or figure | [Table-and-figure guide](simplified/README.md#tables-and-figures) |
| Regenerate all current outputs | [Run all entry point](simplified/run_all.py) |
| Understand the proposed mixture | [Fixed 75/25 mixture interface](simplified/model/mixture_75_25.py) and the original [`mixture_prediction`](src/estudio_publicacion/sikov_2026-10-10/scripts/improve_candidates.py) |
| Prepare the external derived inputs | [Simplified input contract](simplified/INPUTS.md) |
| Re-estimate the complete analysis from source data | [Full reproduction instructions](REPRODUCING.md) and [data requirements](reproduction/DATA_INPUTS.md) |
| Cite the code | [CITATION.cff](CITATION.cff) |

## Quick start

Run these commands from the repository root:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r simplified/requirements.txt
python tools/check_code_release.py
```

The simplified workflow requires an external input folder containing the registered derived CSVs and reference tables. Their contents are not distributed in this repository. See [INPUTS.md](simplified/INPUTS.md) before running the analysis.

```sh
export IAOS_SIMPLE_DATA_ROOT="/external/path/simplified_inputs"
export IAOS_SIMPLE_OUTPUT_ROOT="/external/path/simplified_outputs"
python simplified/verify_inputs.py
python simplified/run_all.py
```

Replace both paths with your own folders. Keep the input and output folders separate and outside the checkout. You can also supply `--data-root` and `--output-root` directly. Use `--help` to inspect an entry point without providing data.

For **Figure 1** and **Table 3**, run:

```sh
python simplified/article/figure_1/generate.py
python simplified/article/table_3/generate.py
```

These commands regenerate outputs from certified derived inputs. The full fitting workflow, including NUTS, REML and bootstrap, is described in [REPRODUCING.md](REPRODUCING.md).

## Browse the current outputs

| Article tables | Article figures |
| --- | --- |
| [Table 1: aggregate performance](simplified/article/table_1/generate.py) | [Figure 1: RMSE and coverage](simplified/article/figure_1/generate.py) |
| [Table 2: primary Holm3 contrasts](simplified/article/table_2/generate.py) | [Figure 2: paired differences from ICC](simplified/article/figure_2/generate.py) |
| [Table 3: incremental comparison with ICC](simplified/article/table_3/generate.py) | [Figure 3: sparse57 sensitivity](simplified/article/figure_3/generate.py) |
| [Table 4: sparse57 sensitivity](simplified/article/table_4/generate.py) | |
| [Table 5: secondary Holm12 contrasts](simplified/article/table_5/generate.py) | |

The [supplement entry points](simplified/README.md#tables-and-figures) cover Tables S1–S9 and S23, and Figures S1–S2. Historical supporting experiments remain with the original analysis sources. Earlier Spanish entry-point paths are retained for compatibility; the English paths above are the recommended starting points.

## Original analysis and full re-estimation

All original statistical sources remain in their existing locations.

| Component | Entry points |
| --- | --- |
| Original FH, SAR and bootstrap models | `src/estudio_publicacion/sikov_2026-10-10/scripts/sikov_model.py`, `sikov_bootstrap.py`, `sikov_published_reference.py`, `sikov_published_fit.py` |
| ICC beta and NUTS | `src/estudio_publicacion/scripts/competitors/beta_power*.py` and `.stan` sources |
| Primary simulations, development and graph/B300 sensitivity | `reproduction/sikov_worker.py`, `sikov_score.py`; original `comparison_*`, `improve_*` and sensitivity scripts |
| Actual-data preparation, fitting and evaluation | `reproduction/real_portable_prepare.py`, `real_portable_worker.py`, `real_score.py` |
| Independent sparse57 and development NUTS cohorts | `revision_20261015/code/numerical_cohorts_portable.py` |
| REML/bootstrap correction | `revision_20261015/code/reml_correction/refit_cohorts.py`, `stable_reml_solver.py`, final `refit_cohorts_stage2_v2.py`, `stable_reml_solver_stage2_v2.py` |
| Corrected mixtures and scoring | frozen `preparation_adapter_19d7a57c.py` for prepare/verify-lock; `recalculate_corrected_results.py` for score/export |
| Incremental ICC contrasts and cohort summaries | `revision_20261015/code/incremental_icc.py`, `cohort_extensions.py` |
| Independent numerical verification | `verify_corrected_fits.py`, `verify_refit_cohorts.py`, final `audit_corrected_scores_v3.py` in the correction directory |

The final second-stage solver and runner names end in `_stage2_v2.py`; the final independent score auditor ends in `_v3.py`. Earlier versions are retained as provenance.

R and CmdStan are additional requirements for data preparation and NUTS. The complete workflow uses the dependencies recorded in `requirements.lock.txt`; the historical environment has its own `reproduction/requirements.legacy.lock.txt`. A full rerun requires external survey, auxiliary, geographic, reference and model-input files, previous states where specified, and a compiled Stan executable. Seeds alone do not guarantee bit-identical finite MCMC trajectories on another platform.

The primary cohort contains 256 populations; the development cohort contains 32. The independent sparse57 sensitivity contains 256 populations and the observed-graph sensitivity 64. NUTS reuses the development populations, and B300 uses an eight-population subset. The cohorts retain their separate roles and the evaluation uses the fixed 75/25 recipe.

## Publication scope and provenance

This repository distributes source code, dependency specifications, scientific protocols, checksums and instructions. The article, supplementary document, cover letter, title page, survey records, stored result tables, figures, fits and chains are excluded.

The original statistical formulas, seeds, likelihoods, parameter bounds, graphs, bootstrap sizes, mixture weights and multiplicity rules are preserved. Portable routing and numerical formatting adaptations are recorded under [provenance/](provenance/). Source checks accept only documented identities and adaptations.

Authors: **Luis Benites, Rocio Maehara and Alejandro Aybar-Flores**, Universidad del Pacífico, Peru. Author identifiers are recorded in [CITATION.cff](CITATION.cff). No software license is assigned in this repository.

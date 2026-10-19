# Simplified workflow: one entry point per table and figure

This is the recommended reader-facing interface for the current **75% ICC beta + 25% full-graph SAR** proposal. It reuses the original statistical modules. The internal names `Selected` and `ICC_SAR_w75` refer to the same fixed recipe.

## Execution order

From the repository root, prepare the environment:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r simplified/requirements.txt
```

Set two folders **outside the repository**:

```sh
export IAOS_SIMPLE_DATA_ROOT="/external/path/simplified_inputs"
export IAOS_SIMPLE_OUTPUT_ROOT="/external/path/simplified_outputs"
python simplified/verify_inputs.py
python simplified/run_all.py
```

Replace the example paths with your own folders. The input folder must contain the registered derived data and reference tables described in [INPUTS.md](INPUTS.md). An existing verified local simplified package can serve as the input folder. The output folder must be separate from the input folder and outside the checkout.

The order is **install → verify inputs → regenerate outputs → inspect verification reports**. You can pass `--data-root` and `--output-root` instead of setting environment variables. `--help` displays the available options without requiring data.

## Figure 1 and Table 3

After setting the variables above:

```sh
python simplified/article/figure_1/generate.py
python simplified/article/table_3/generate.py
```

Outputs are written under the configured external output folder:

```text
simplified_outputs/
  article/figure_1/output/Figure_1.png
  article/table_3/output/Table_3.csv
```

Each result also includes a `verification.json` report identifying its sources, inputs and outputs. The entry points are independent: you can generate Table 3 directly without first generating Tables 1 and 2.

To regenerate one group:

```sh
python simplified/run_all.py --only article
python simplified/run_all.py --only supplement
```

## Tables and figures

The commands below assume that the terminal is at the repository root and that both external-folder variables are set.

| Result | Entry point | Main computation or plot |
| --- | --- | --- |
| Table 1 | [article/table_1/generate.py](article/table_1/generate.py) | Aggregate performance across the primary populations |
| Table 2 | [article/table_2/generate.py](article/table_2/generate.py) | Primary IS95 contrasts and the Holm3 family |
| Table 3 | [article/table_3/generate.py](article/table_3/generate.py) | Paired proposal–ICC differences, MCSE and relative changes |
| Table 4 | [article/table_4/generate.py](article/table_4/generate.py) | Independent sparse57 sensitivity |
| Table 5 | [article/table_5/generate.py](article/table_5/generate.py) | Secondary contrasts and the Holm12 family |
| Figure 1 | [article/figure_1/generate.py](article/figure_1/generate.py) | RMSE and coverage by context and endpoint |
| Figure 2 | [article/figure_2/generate.py](article/figure_2/generate.py) | Paired differences from ICC |
| Figure 3 | [article/figure_3/generate.py](article/figure_3/generate.py) | Sparse57 contrasts |
| Table S1 | [supplement/table_s1/generate.py](supplement/table_s1/generate.py) | Laplace/NUTS development and leave-one-population-out checks |
| Table S2 | [supplement/table_s2/generate.py](supplement/table_s2/generate.py) | Incremental comparison by context, endpoint and support |
| Table S3 | [supplement/table_s3/generate.py](supplement/table_s3/generate.py) | Sparse57 sensitivity by district support |
| Table S4 | [supplement/table_s4/generate.py](supplement/table_s4/generate.py) | Computational diagnostics |
| Table S5 | [supplement/table_s5/generate.py](supplement/table_s5/generate.py) | Common projection to [0,1] |
| Table S6 | [supplement/table_s6/generate.py](supplement/table_s6/generate.py) | Bias-corrected Gaussian intervals |
| Table S7 | [supplement/table_s7/generate.py](supplement/table_s7/generate.py) | Bootstrap-size sensitivity |
| Table S8 | [supplement/table_s8/generate.py](supplement/table_s8/generate.py) | Poverty-stratum contrasts |
| Table S9 | [supplement/table_s9/generate.py](supplement/table_s9/generate.py) | Observed-graph mechanism sensitivity |
| Table S23 | [supplement/table_s23/generate.py](supplement/table_s23/generate.py) | REML/bootstrap correction counts |
| Figure S1 | [supplement/figure_s1/generate.py](supplement/figure_s1/generate.py) | IS95 before and after the common projection |
| Figure S2 | [supplement/figure_s2/generate.py](supplement/figure_s2/generate.py) | Descriptive district application map |

For example, run `python simplified/supplement/table_s1/generate.py`. The numbering follows the current article and supplement. [artifacts.json](artifacts.json) maps the English artifact identifiers to their entry points and original sources. Historical CUT8 experiments remain in the original analysis sources with their separate roles.

## Where the proposal is implemented

[model/mixture_75_25.py](model/mixture_75_25.py) provides a fixed-weight interface to the original [`mixture_prediction`](../src/estudio_publicacion/sikov_2026-10-10/scripts/improve_candidates.py). The original function calculates quantiles of the mixture distribution and retains uncertainty arising from disagreement between component means.

This interface accepts ICC and SAR components that have already been estimated over the same districts, endpoints and ordering. It does not fit the components. Inspect its help for the required NPZ arrays:

```sh
python simplified/model/mixture_75_25.py --help
```

## What this workflow reproduces

The 20 entry points regenerate tables and figures from **derived inputs produced by the corrected fits**. Table 3 and Table S2 recalculate paired statistics from population-level metrics; the other tables format certified aggregates. Figures are drawn again using the original plotting statements. Reference tables are read only after computation to verify the result.

Before running, the workflow checks 18 original source files and 35 external inputs against [input_hashes.json](input_hashes.json). A temporary view outside the checkout preserves the relative paths required by the analysis adapters and is removed when execution finishes. All generated artifacts stay in the external output folder.

This interface does not repeat NUTS, REML or bootstrap fitting. For full re-estimation from source data, follow [REPRODUCING.md](../REPRODUCING.md) and [DATA_INPUTS.md](../reproduction/DATA_INPUTS.md), retaining the prescribed input checks, prediction locks and validation order.

The repository contains source code, instructions and hashes. It excludes data records, saved tables, images and article documents. Older Spanish entry-point paths remain available for compatibility; use the English paths in this guide for new work.

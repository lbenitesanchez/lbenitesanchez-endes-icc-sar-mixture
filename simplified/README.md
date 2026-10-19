# Simplified workflow: one entry point per table and figure

This is the recommended reader-facing interface for the **75% ICC beta + 25% full-graph SAR** proposal. It reuses the original statistical modules. The internal names `Selected` and `ICC_SAR_w75` refer to the same fixed recipe.

It covers **20 outputs: article Tables 1–5 and Figures 1–3; supplement Tables S1–S9 and S23, and Figures S1–S2**. Historical supplement Tables S10–S22 and Figures S3–S6 have no simplified entry points; use the [original-source map](../REPRODUCING.md#historical-supplement-results).

**A fresh clone is not sufficient to generate numerical outputs.** Obtain the complete [external input bundle](INPUTS.md#obtaining-the-input-bundle) first. The commands here regenerate outputs from corrected derived inputs; they do not rerun statistical model fits.

## Execution order

Follow the root [Quick start](../README.md#quick-start) to clone the repository. From the repository root, prepare the environment once with **Python 3.12** (validated with **3.12.14**):

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r simplified/requirements.txt
```

Set two folders **outside the repository** and verify the inputs:

```sh
export IAOS_SIMPLE_DATA_ROOT="/external/path/simplified_inputs"
export IAOS_SIMPLE_OUTPUT_ROOT="/external/path/simplified_outputs"
python simplified/verify_inputs.py
```

Replace both example paths with your own folders. The input root must directly contain the `datos/` and `referencias/` files described in [INPUTS.md](INPUTS.md). The output folder must be separate from the input folder, outside the checkout, and either new, empty or already managed by this workflow.

The order is **install → obtain and verify all 35 inputs → choose one result, a group or all 20 → inspect verification reports**. All 35 CSVs are required even for a single table or figure. The simplified route uses Python only; R and CmdStan are additional requirements for full re-estimation. `--help` displays the options without requiring data.

## Run one result, a group or all 20

After setting the variables and verifying inputs above, choose the command you need. For example, generate Figure 1 or Table 3 directly:

```sh
python simplified/article/figure_1/generate.py
python simplified/article/table_3/generate.py
```

These create the following files under `IAOS_SIMPLE_OUTPUT_ROOT`:

```text
article/figure_1/output/Figure_1.png
article/figure_1/output/Figure_1.svg
article/table_3/output/Table_3.csv
```

Each result includes a `verification.json` report. Entry points are independent: Table 3 does not require generating Tables 1 and 2 first.

Alternatively, choose one group or all 20 supported outputs:

```sh
python simplified/run_all.py --only article
python simplified/run_all.py --only supplement
python simplified/run_all.py
```

The article group has 8 outputs; the supplement group has 12. Run only the command corresponding to your choice. To select one result through the batch runner, use its identifier, for example `python simplified/run_all.py --only Article_Table_3`.

You can pass `--data-root` and `--output-root` directly instead of setting environment variables. For example:

```sh
python simplified/article/table_3/generate.py \
  --data-root "/external/path/simplified_inputs" \
  --output-root "/external/path/simplified_outputs"
```

## Tables and figures

The commands below assume that the terminal is at the repository root, the Python environment is activated, and both external-folder variables are set. Every output path is relative to `IAOS_SIMPLE_OUTPUT_ROOT`. Figure entries also create the corresponding `.svg` file.

| Result and purpose | Command | Main output |
| --- | --- | --- |
| [Table 1](article/table_1/generate.py): Aggregate estimator performance | `python simplified/article/table_1/generate.py` | `article/table_1/output/Table_1.csv` |
| [Table 2](article/table_2/generate.py): Primary Holm3 contrasts | `python simplified/article/table_2/generate.py` | `article/table_2/output/Table_2.csv` |
| [Table 3](article/table_3/generate.py): Incremental comparison with ICC | `python simplified/article/table_3/generate.py` | `article/table_3/output/Table_3.csv` |
| [Table 4](article/table_4/generate.py): Independent sparse57 sensitivity | `python simplified/article/table_4/generate.py` | `article/table_4/output/Table_4.csv` |
| [Table 5](article/table_5/generate.py): Secondary Holm12 contrasts | `python simplified/article/table_5/generate.py` | `article/table_5/output/Table_5.csv` |
| [Figure 1](article/figure_1/generate.py): RMSE and interval coverage | `python simplified/article/figure_1/generate.py` | `article/figure_1/output/Figure_1.png` |
| [Figure 2](article/figure_2/generate.py): Paired differences against ICC | `python simplified/article/figure_2/generate.py` | `article/figure_2/output/Figure_2.png` |
| [Figure 3](article/figure_3/generate.py): Sparse57 contrasts | `python simplified/article/figure_3/generate.py` | `article/figure_3/output/Figure_3.png` |
| [Table S1](supplement/table_s1/generate.py): Laplace and NUTS development and leave-one-population-out checks | `python simplified/supplement/table_s1/generate.py` | `supplement/table_s1/output/Table_S1.csv` |
| [Table S2](supplement/table_s2/generate.py): Comparisons by context and support | `python simplified/supplement/table_s2/generate.py` | `supplement/table_s2/output/Table_S2.csv` |
| [Table S3](supplement/table_s3/generate.py): Sparse57 sensitivity by district support | `python simplified/supplement/table_s3/generate.py` | `supplement/table_s3/output/Table_S3.csv` |
| [Table S4](supplement/table_s4/generate.py): Computational diagnostics | `python simplified/supplement/table_s4/generate.py` | `supplement/table_s4/output/Table_S4.csv` |
| [Table S5](supplement/table_s5/generate.py): Common probability-bound projection | `python simplified/supplement/table_s5/generate.py` | `supplement/table_s5/output/Table_S5.csv` |
| [Table S6](supplement/table_s6/generate.py): Bias-corrected Gaussian intervals | `python simplified/supplement/table_s6/generate.py` | `supplement/table_s6/output/Table_S6.csv` |
| [Table S7](supplement/table_s7/generate.py): Bootstrap-size sensitivity | `python simplified/supplement/table_s7/generate.py` | `supplement/table_s7/output/Table_S7.csv` |
| [Table S8](supplement/table_s8/generate.py): Poverty-stratum contrasts | `python simplified/supplement/table_s8/generate.py` | `supplement/table_s8/output/Table_S8.csv` |
| [Table S9](supplement/table_s9/generate.py): Graph-mechanism sensitivity | `python simplified/supplement/table_s9/generate.py` | `supplement/table_s9/output/Table_S9.csv` |
| [Table S23](supplement/table_s23/generate.py): Numerical-correction accounting | `python simplified/supplement/table_s23/generate.py` | `supplement/table_s23/output/Table_S23.csv` |
| [Figure S1](supplement/figure_s1/generate.py): Interval score and common projection | `python simplified/supplement/figure_s1/generate.py` | `supplement/figure_s1/output/Figure_S1.png` |
| [Figure S2](supplement/figure_s2/generate.py): Descriptive district application map | `python simplified/supplement/figure_s2/generate.py` | `supplement/figure_s2/output/Figure_S2.png` |

[artifacts.json](artifacts.json) maps the English artifact identifiers to entry points and original sources. Numbering follows the current article and supplement. Use the [historical supplement map](../REPRODUCING.md#historical-supplement-results) for S10–S22 and Figures S3–S6.

## Check the generated results

Each table is checked cell by cell against its external reference CSV. Each figure is checked against the registered plotted-value fingerprint; its report also records whether the PNG is byte-identical to the archived image. Rendering differences across environments can affect PNG bytes, so inspect that field separately from the plotted-value check.

Read `verification.json` beside each result. At the output root, `EXECUTION_VERIFICATION.json` records the **latest invocation only**: running a single result after a batch replaces that batch summary. Use `run_all.py` in one invocation when you want a report covering all 20 outputs. Previously generated per-result files remain in their own folders.

## Where the proposal is implemented

[model/mixture_75_25.py](model/mixture_75_25.py) provides a fixed-weight interface to the original [`mixture_prediction`](../src/estudio_publicacion/sikov_2026-10-10/scripts/improve_candidates.py). The original function calculates quantiles of the mixture distribution and retains uncertainty arising from disagreement between component means.

This interface accepts ICC and SAR components already estimated over the same districts, endpoints and ordering. It does not fit the components. Inspect its help for the required NPZ arrays:

```sh
python simplified/model/mixture_75_25.py --help
```

## What this workflow reproduces

The 20 entry points regenerate tables and figures from **derived inputs produced by the corrected fits**. Table 3 and Table S2 recalculate paired statistics from population-level metrics; the other tables format certified aggregates. Figures are drawn again using the original plotting statements. Reference tables are read only after computation to verify the result.

Before running, the workflow checks 18 original source files and 35 external inputs against [input_hashes.json](input_hashes.json). A temporary view outside the checkout preserves the relative paths required by the analysis adapters and is removed when execution finishes. All generated artifacts stay in the external output folder.

This interface does not repeat NUTS, REML or bootstrap fitting. For full re-estimation from source data, follow [REPRODUCING.md](../REPRODUCING.md) and [DATA_INPUTS.md](../reproduction/DATA_INPUTS.md), retaining the prescribed input checks, prediction locks and validation order.

The repository contains source code, instructions and hashes. It excludes data records, saved tables, images and article documents. Older Spanish entry-point paths remain available for compatibility; use the English paths in this guide for new work.

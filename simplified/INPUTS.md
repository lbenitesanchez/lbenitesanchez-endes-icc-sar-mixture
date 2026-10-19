# External inputs for the simplified workflow

`--data-root` identifies an external input folder containing the frozen derived data and reference CSV files. Their exact relative paths and SHA-256 hashes are registered in [input_hashes.json](input_hashes.json); their contents are not distributed on GitHub.

The verified local simplified package can be used as the input folder. Keep that package outside the checkout. Its historical directory names are part of the input contract, so retain them when providing the files:

```text
EXTERNAL_INPUTS/
  datos/
    desarrollo/
    diagnosticos/
    incremental/
    mapa/
    sikov/
    sparse57/
  referencias/
    Articulo/
    Suplemento/
```

The contract contains **20 derived-data CSVs and 15 reference-table CSVs**. Derived inputs include simulated-population metrics, corrected aggregates, diagnostics, district predictions, map coordinates and observation masks. Reference tables are the final exported tables and are used only to verify outputs after computation.

`sources_sha256` verifies **18 source files already present in this repository**. Additional copies of those sources are not required beside the inputs. The hashes bind the interface to its recorded numerical revision; they prevent silently substituting aggregates from another run. A new full re-estimation must retain its own provenance and pass the validation chain described in [REPRODUCING.md](../REPRODUCING.md).

## Verify without generating outputs

From the repository root:

```sh
python simplified/verify_inputs.py --data-root /external/path/simplified_inputs
```

## Regenerate all current outputs

```sh
python simplified/run_all.py \
  --data-root /external/path/simplified_inputs \
  --output-root /external/path/simplified_outputs
```

Replace the example paths with your own folders. The output folder must be outside the checkout, separate from the input folder, and either new or managed by this workflow. The program checks the inputs before creating outputs.

Source-data preparation, model fitting and validation require the additional external files specified in [reproduction/DATA_INPUTS.md](../reproduction/DATA_INPUTS.md). The simplified workflow covers regeneration from certified derived inputs; it does not replace the full estimation pipeline.

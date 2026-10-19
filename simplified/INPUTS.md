# External inputs for the simplified workflow

`--data-root` identifies an external input folder containing the frozen derived data and reference CSV files. Their exact relative paths and SHA-256 hashes are registered in [input_hashes.json](input_hashes.json); their contents are not distributed on GitHub.

## Obtaining the input bundle

Cloning this repository and installing Python packages does not create the required inputs. **All 35 registered CSVs are required for every simplified entry point**, including a single table or figure. There is no public download of this bundle in this repository.

For access enquiries, contact the corresponding author, Luis Benites, at [le.benitess@up.edu.pe](mailto:le.benitess@up.edu.pe), and identify the repository revision and `simplified/input_hashes.json` contract. Availability and any access conditions must be confirmed with the authors. The required subset consists of the `datos/` and `referencias/` files listed in that contract; the article, supplement, cover letter, raw survey records and saved model fits are not needed for this simplified route.

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

## Regenerate all 20 supported outputs

```sh
python simplified/run_all.py \
  --data-root /external/path/simplified_inputs \
  --output-root /external/path/simplified_outputs
```

Replace the example paths with your own folders. The output folder must be outside the checkout, separate from the input folder, and either new or managed by this workflow. The program checks the inputs before creating outputs.

Source-data preparation, model fitting and validation require the additional external files specified in [reproduction/DATA_INPUTS.md](../reproduction/DATA_INPUTS.md). The simplified workflow covers regeneration from certified derived inputs; it does not replace the full estimation pipeline.

## Missing files or a hash mismatch

Point `--data-root` to the folder directly containing `datos/` and `referencias/`, retain the exact historical filenames, and supply the complete 35-file bundle. A hash mismatch means the file does not match the recorded revision; obtain the matching bundle before continuing. A new full re-estimation needs a separately registered provenance chain and may produce different values; its outputs are not interchangeable with this frozen input contract.

# Included inputs for the simplified workflow

The complete **35-file reproduction bundle is included in [inputs/](inputs/)**: 20 derived-data CSVs and 15 numerical reference tables. Its exact relative paths and SHA-256 hashes remain registered in [input_hashes.json](input_hashes.json). The files are copied without changing values or bytes.

## Obtaining the input bundle

Clone or download this repository. The English entry points automatically use `simplified/inputs/`, so no separate download, data request or input-path configuration is required. The older `external_inputs_sha256` JSON key is retained for compatibility; the files it identifies are now bundled.

**The principal ENDES survey microdata are not included.** These CSVs contain derived simulation metrics, computational diagnostics, aggregate comparisons, model-based district estimates, district-capital coordinates and numerical reference tables. They allow regeneration of the 20 supported outputs, rather than re-estimation of NUTS, REML or bootstrap models. Read the [data dictionary, privacy review and INEI attribution](inputs/README.md).

The original folder names are retained so the frozen analysis paths and hashes still match:

```text
simplified/inputs/
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

All 35 CSVs are checked for every entry point, even when generating one result. Reference tables are used only after computation to verify the generated table; they are not copied to create the output. `sources_sha256` also verifies 18 original source files already present in this repository.

## Verify without generating outputs

From the repository root:

```sh
unset IAOS_SIMPLE_DATA_ROOT
python simplified/verify_inputs.py
```

`unset` clears an input override from an earlier session. The default verifier checks the included bundle and reports `input_location: bundled`.

## Regenerate all 20 supported outputs

```sh
python simplified/run_all.py --output-root "../endes-mixture-outputs"
```

The output folder must be outside the checkout and either new, empty or already managed by this workflow. No generated images or new result files are written into the repository.

## Optional external input copy

An existing verified local package or a separate copy of these 35 CSVs can still serve as the input root:

```sh
python simplified/verify_inputs.py --data-root "/external/path/simplified_inputs"
python simplified/run_all.py \
  --data-root "/external/path/simplified_inputs" \
  --output-root "/external/path/simplified_outputs"
```

An external root must directly contain `datos/` and `referencias/`, be outside the checkout, and not overlap the output folder. `IAOS_SIMPLE_DATA_ROOT` is an alternative override. The only accepted input root inside the checkout is the registered `simplified/inputs/` directory.

## Missing files or a hash mismatch

Restore the matching files from this repository revision. Preserve the historical filenames and the complete 35-file bundle. A hash mismatch means the file does not match the recorded numerical revision. New full model fits require their own source, executable and input provenance and may produce different results; their outputs are not interchangeable with this frozen input contract.

Full source-data preparation and model fitting still require the external files specified in [reproduction/DATA_INPUTS.md](../reproduction/DATA_INPUTS.md). The included CSVs do not replace those original data or fitting reference trees.

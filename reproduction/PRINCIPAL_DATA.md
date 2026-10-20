# Principal data and model fitting from original inputs

The principal database was reviewed separately from the 35 derived reproduction CSVs. **It is not published in this repository.** The linked child and household records contain age, sex, sample/district identifiers and individual anaemia/stunting responses. The prepared district fitting files also retain direct proportions and sample counts; some endpoints are based on one child. Releasing those files unchanged can disclose an individual health response. Suppressing or altering those values would change the inputs used in the article.

Two reviewed **district covariate CSVs** are included in [inputs/](inputs/), with their original values and provenance caveats. They contain no individual ENDES responses. The [principal-data review](../provenance/principal_data_review.json) records the publication decision.

**Current scope:** the 35 simplified CSVs allow direct regeneration of 15 tables and 5 figures. A new full model run still requires independently acquired survey files and complete scientific source/reference trees. **GitHub plus the six raw SAV files alone is not currently a self-contained full model-fitting package.** The commands below document the actual pipeline and state its prerequisites; they are not evidence that a reader-only full run has been completed.

## 1. Acquire the official survey data locally

Obtain data through the [INEI microdata portal](https://proyectos.inei.gob.pe/microdatos/), using the agency's current access conditions. Keep the downloads and all reconstructed person-level records in an external private working directory.

| Year | Official acquisition route | Interpretation |
| --- | --- | --- |
| 2019 | [Module 64](https://proyectos.inei.gob.pe/iinei/srienaho/descarga/SPSS/691-Modulo64.zip) and [Module 74](https://proyectos.inei.gob.pe/iinei/srienaho/descarga/SPSS/691-Modulo74.zip) | Routes recorded by the analysis; availability was checked with HEAD requests. |
| 2024 | [Current household module 1629](https://proyectos.inei.gob.pe/iinei/srienaho/descarga/SPSS/968-Modulo1629.zip) and [current weight/height/anaemia module 1638](https://proyectos.inei.gob.pe/iinei/srienaho/descarga/SPSS/968-Modulo1638.zip) | Current official alternatives found in INEI's module index. Byte identity with the archived inputs must be checked after acquisition. |
| 2024, archived route | [Informacion Complementaria ENDES 2024](https://www.inei.gob.pe/media/MenuRecursivo/publicaciones_digitales/Est/Lib2016/Informacion_Complementaria_Endes_2024.zip) | Original recorded URL; current availability was not confirmed during this review. |

Extract and arrange the relevant files beneath an external study root. When necessary, use the year-specific filenames shown below; do not modify the SAV contents:

```text
PRIVATE_RAW_STUDY/
  datos/raw/ENDES2019/Modulo64/RECH0.SAV
  datos/raw/ENDES2019/Modulo64/RECH1.sav
  datos/raw/ENDES2019/Modulo74/RECH6.SAV
  fase2_2026-10-07/datos/private_external_raw/RECH0_2024.SAV
  fase2_2026-10-07/datos/private_external_raw/RECH1_2024.SAV
  fase2_2026-10-07/datos/private_external_raw/RECH6_2024.SAV
```

The expected six sizes and SHA-256 hashes are in [private_csv_checksums.json](../expected/private_csv_checksums.json). After obtaining the files, run this from the repository root:

```sh
python reproduction/check_principal_inputs.py --data-root "/external/path/PRIVATE_RAW_STUDY"
```

The checker reads file bytes for identity verification and checks the two public auxiliaries. It does not export individual records, prepare outcomes or fit models. A changed archive is not accepted as the historical data merely because its filenames match. Passing this check certifies file identities only, not completeness of the full pipeline prerequisites.

## 2. Install the fitting environment

Use Python 3.12 and the repository's [requirements.lock.txt](../requirements.lock.txt). The simplified plotting requirements alone are insufficient. The fitting workflow also requires a C++ toolchain, CmdStan (the recorded local runtime used 2.40.0), and R with `haven`, `survey`, `readxl`, `jsonlite` and `digest`. Install dependencies using their official instructions; keep the Stan runtime outside the checkout.

```sh
python -m pip install -r requirements.lock.txt
export RLIB="/external/path/R-library"
Rscript -e 'lib <- Sys.getenv("RLIB"); dir.create(lib, recursive=TRUE, showWarnings=FALSE); install.packages(c("haven", "survey", "readxl", "jsonlite", "digest"), lib=lib, repos="https://cloud.r-project.org")'
```

## 3. Supply the matching scientific reference prerequisites

Before preparing or fitting, the historical pipeline requires the following externally reviewed inputs:

- A scientific source-study tree with matching `protocol_freeze.json`, the published-SAR addendum and observed-graph freeze metadata. These files are not supplied by `src/estudio_publicacion/` alone.
- The matching scientific reference tree, input hashes and preparation comparison references.
- Geographic auxiliaries, 13 recorded census queries and every aggregate table referenced by those queries.
- The historical phase-two protocol/access/partition/prediction-lock metadata. Replaying these records does not create a new independent 2024 validation sample.
- Original preparation comparison data, including the district direct-outcome basis. Keep these sensitive records external and private.

The two public auxiliary CSVs do not replace this complete reference structure. Do not create placeholder freezes, skip source/hash/convergence gates, or interpret a failed prerequisite check as a successful model run. The package currently does not provide an automated public-only reconstruction of all these reference artifacts.

## 4. Compile and prepare with the complete prerequisites

The following variable names identify separate, actual directories. Set them to your own matching paths before running any command:

```sh
export REPO="$PWD"
export WORK="/external/path/new_model_run"
export RAW="/external/path/PRIVATE_RAW_STUDY"
export SOURCE_STUDY="/external/path/reviewed_scientific_source_study"
export EXPECTED="/external/path/reviewed_aggregate_reference_tree"
export REFERENCE="/external/path/new_private_reference_tree"
export CMDSTAN="/external/path/cmdstan-2.40.0"
export RSCRIPT="/absolute/path/to/Rscript"
export RLIB="/external/path/R-library"
```

`SOURCE_STUDY` must include the full scientific protocol metadata required above. `EXPECTED` must be reviewed for the materializer's aggregate-only input contract; do not point it at an unrestricted private study folder. `RLIB` must identify the same R package library populated in step 2. `REFERENCE` and `WORK` must be new external destinations. The materializer rejects raw/private files in the claimed public source/reference inputs and links separately acquired SAV files only into the private destination.

```sh
python reproduction/compile_runtime.py \
  --source-root "$REPO/src/estudio_publicacion" \
  --cmdstan "$CMDSTAN" --runtime-root "$WORK/runtime"

python reproduction/real_materialize_reference.py \
  --public-study-root "$SOURCE_STUDY" --expected-study-root "$EXPECTED" \
  --raw-study-root "$RAW" --reference-root "$REFERENCE" \
  --catalogue "$REPO/expected/private_csv_checksums.json"

python reproduction/real_portable_prepare.py \
  --source-root "$SOURCE_STUDY" --reference-root "$REFERENCE" \
  --run-root "$WORK/sikov" --catalogue "$REPO/expected/private_csv_checksums.json" \
  --public-aux-csv "$REPO/reproduction/inputs/aux_all_districts.csv" \
  --rscript "$RSCRIPT" --r-library "$RLIB"
```

Inspect the preparation receipt and `real_basis_diagnostic.json`. Previous portable preparation matched the raw/direct/auxiliary quantities but disclosed a difference in the historical spectral basis. The basis is not uniquely determined in the disconnected graph; do not label a new preparation bit-identical without its actual comparison results. [real_current_basis_lock.py](real_current_basis_lock.py) supports an explicitly declared geometry-only target before any fits; that target is not bundled here. Do not substitute a completed fitted state for fresh estimation.

## 5. Fit, correct and validate

Once preparation and its gates pass, the initial temporal and territorial fits use these entry points:

```sh
python reproduction/real_portable_worker.py --action fit --scope temporal \
  --source-root "$SOURCE_STUDY" --reference-root "$REFERENCE" \
  --run-root "$WORK/sikov" --catalogue "$REPO/expected/private_csv_checksums.json" \
  --exe "$WORK/runtime/beta_power_hybrid" --cmdstan "$CMDSTAN"

python reproduction/real_portable_worker.py --action fit --scope territorial \
  --source-root "$SOURCE_STUDY" --reference-root "$REFERENCE" \
  --run-root "$WORK/sikov" --catalogue "$REPO/expected/private_csv_checksums.json" \
  --exe "$WORK/runtime/beta_power_hybrid" --cmdstan "$CMDSTAN"
```

These create new model fits and retained sampler diagnostics; they are not final corrected article results by themselves. Continue with the same-model REML/bootstrap correction, native prediction locking, mixture calculation, scoring and independent verification in [REPRODUCING.md](../REPRODUCING.md#2-refit-reml-and-bootstrap-with-the-recorded-models-and-seeds), in that order. The simulation/development/sparse57/graph cohorts have their own prescribed roles and are documented there. Preserve the 75/25 weights, seeds, model definitions and multiplicity families.

A full fitting run can be reported as reproduced only after the completed diagnostics, correction and scoring checks pass. The present data-publication review did not run new NUTS/REML/bootstrap fits or establish full reader-only numerical reproduction.

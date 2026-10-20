# Reviewed reproduction inputs

This folder contains the **35 CSVs needed to regenerate the 20 outputs supported by the simplified workflow**. The files retain the original hashes in [input_hashes.json](../input_hashes.json). The exact inventory, column names, row counts, review summary and source attribution are recorded in [provenance/public_inputs.json](../../provenance/public_inputs.json).

These are **derived research inputs**, rather than the principal ENDES database. Original survey records, household or person identifiers, respondent locations, raw individual outcomes, native fits and chains are not included. The review examined all rows and columns and traced the district files to their model and geographic sources; it detected no personal sensitive content in these frozen files.

## What is included

| Location | Content and unit of observation | Role |
| --- | --- | --- |
| `datos/desarrollo/` | Candidate rankings and leave-one-simulated-population-out diagnostics | Development summaries |
| `datos/diagnosticos/` | Method/cohort counts, elapsed times and numerical diagnostics | Tables S4 and S23 |
| `datos/incremental/` | Metrics per simulated population, method, outcome and district-support group, plus their paired summaries | Recalculate proposal–ICC comparisons |
| `datos/sparse57/` | Paired summaries from independent simulated sparse57 populations | Sparse-support sensitivity |
| `datos/sikov/` except `aplicacion2019/` | Simulated and aggregate comparison scores, graph/B300/poverty sensitivities and aggregate external contrasts | Performance and sensitivity outputs |
| `datos/mapa/area_frame.csv` | 1,874 district codes and district-capital coordinates | Geographic plotting locations |
| `datos/mapa/district_availability_for_map.csv` | One row per district, with endpoint-availability booleans | Indicate availability of district fitting estimates |
| `datos/sikov/resultados/aplicacion2019/matched_application2019_districts.csv` | 1,874 districts × 11 modeled methods; district estimates, intervals, capital coordinates and derived poverty stratum | Descriptive application figure |
| `referencias/Articulo/`, `referencias/Suplemento/` | 15 final numerical reference-table matrices | Verification after table computation |

The bundle has 6,420,838 bytes across 35 CSVs. Reference tables and derived summaries are intentionally included to make the numerical regeneration checks available to readers. They do not include article or supplement text.

## Interpreting the district fields

- `ubigeo` is the public district code, not a respondent identifier. Treat it as a six-character string and retain leading zeros.
- `latitude` and `longitude` are decimal-degree coordinates of **district legal capitals**, not homes, survey clusters or respondent GPS points. The same capital coordinates are repeated across modeled methods.
- `p_anemia`, `p_stunting` and their `lo95_`/`hi95_` columns are model-based district estimates and uncertainty intervals on the proportion scale, not individual health measurements or direct empirical district proportions. The application plot converts these to percentages. Gaussian model estimates or bounds are not automatically truncated to [0,1].
- `anemia_observed` and `stunting_observed` indicate availability of usable district fitting estimates. They provide no respondent identifiers or respondent counts.
- `poverty_stratum` is an ordered, derived district poverty stratum, not a household income record.
- `replication` and `held_replication` index simulated populations. In computational diagnostics, counts refer to fits, bootstrap draws or simulation cases, rather than surveyed persons.

These descriptive district predictions are **research outputs and are not official INEI district statistics**. The bundle does not add independent real-data validation or alter the analysis conclusions.

## Source attribution and redistribution review

The research uses ENDES and historical district auxiliaries from the **Instituto Nacional de Estadística e Informática (INEI), Peru**:

- [ENDES official portal](https://proyectos.inei.gob.pe/endes/): survey source for the authors' derived analysis. Original survey microdata are not redistributed here.
- [Directorio Nacional de Municipalidades Provinciales, Distritales y de Centros Poblados 2019](https://www.inei.gob.pe/media/MenuRecursivo/publicaciones_digitales/Est/Lib1653/Libro.pdf): source of district-capital coordinates. Its credits page expressly permits complete or partial reproduction when INEI is acknowledged. All 1,874 published capital-coordinate pairs were independently matched to this directory.
- [Mapa de Pobreza Provincial y Distrital 2018](https://www.gob.pe/institucion/inei/informes-publicaciones/3204872-mapa-de-pobreza-provincial-y-distrital-2018) and [Census 2017 aggregate query service](https://censos2017.inei.gob.pe/redatam/): historical auxiliary sources used in the authors' analysis.

The other CSVs are the authors' generated simulation results, diagnostics, aggregate comparisons, district predictions, availability flags and numerical reference tables. The content/provenance review identified no redistribution blocker for these 35 files. Original source publications and contact-containing directories or workbooks are not bundled. No new formal license is assigned to official sources or to the repository by this review.

## Run

From the repository root, after installing [the Python dependencies](../README.md#execution-order):

```sh
unset IAOS_SIMPLE_DATA_ROOT
python simplified/verify_inputs.py
python simplified/run_all.py --output-root "../endes-mixture-outputs"
```

The program verifies the included data and original sources, regenerates the outputs and writes verification reports to the external output folder. It does not refit models. See the [per-table and per-figure commands](../README.md#tables-and-figures).

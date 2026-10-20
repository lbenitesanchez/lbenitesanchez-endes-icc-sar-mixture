# Public district covariates for model preparation

These **two district-level auxiliary CSVs** were reviewed separately from the 35 simplified reproduction inputs. They contain 1,874 district rows each and no ENDES respondent records, individual health responses or ENDES respondent counts. See the [review and exact hashes](../../provenance/principal_data_review.json).

| File | Columns | Recorded role and provenance |
| --- | --- | --- |
| `aux_all_districts.csv` | 22 | Aggregate auxiliary compilation used by the portable preparation route. Its source-status columns disclose an author-supplied compilation, unverified extraction and Redatam queries not independently re-executed. It is not presented as a new independent official census extraction. |
| `aux_sikov14_official.csv` | 54 | Historical district census, geography and poverty covariates assembled by [`align_aux_sikov.py`](../../src/estudio_publicacion/sikov_2026-10-10/scripts/align_aux_sikov.py) without reading ENDES responses. Counts are district census aggregates, rather than ENDES sample counts. |

Values and bytes are unchanged from the recorded fitting auxiliaries. District identifiers and names refer to public administrative areas; geographic coordinates represent district capitals. Source attribution: **Instituto Nacional de Estadística e Informática (INEI), Peru**, historical Census 2017, district poverty and geographic publications. The [existing source-attribution note](../../simplified/inputs/README.md#source-attribution-and-redistribution-review) identifies the official publications. No new formal license is assigned by this release.

The **principal child/household database and the direct-outcome fitting basis are excluded**: they contain individual health information or small district cells that can disclose an individual response. These two covariate files do not replace the survey outcomes or the full scientific reference trees required by the historical fitting pipeline.

For official acquisition routes, local input verification, model commands and the remaining prerequisites, read [PRINCIPAL_DATA.md](../PRINCIPAL_DATA.md). For immediate regeneration of the 20 supported tables and figures, use the [simplified workflow](../../simplified/README.md).

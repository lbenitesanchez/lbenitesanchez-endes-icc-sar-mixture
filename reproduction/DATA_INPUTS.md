# External input requirements

This code-only repository contains no input datasets, survey rows, geography arrays, fitted states, chains or saved result tables. Keep them outside the checkout and pass explicit paths to the preparation and fitting CLIs.

## Official survey acquisition records

The original acquisition records used these INEI URLs. They are provenance pointers; later downloads must be checked against the archived file identities.

- ENDES 2019 module 64: https://proyectos.inei.gob.pe/iinei/srienaho/descarga/SPSS/691-Modulo64.zip
- ENDES 2019 module 74: https://proyectos.inei.gob.pe/iinei/srienaho/descarga/SPSS/691-Modulo74.zip
- ENDES 2024: https://www.inei.gob.pe/media/MenuRecursivo/publicaciones_digitales/Est/Lib2016/Informacion_Complementaria_Endes_2024.zip
- Historical ENDES 2025 support: https://www.inei.gob.pe/media/MenuRecursivo/publicaciones_digitales/Est/Lib2089/informacion_complementaria.zip

Expected 2019 files in the private reference study are `datos/raw/ENDES2019/Modulo64/RECH0.SAV`, `RECH1.sav` in the same directory and `datos/raw/ENDES2019/Modulo74/RECH6.SAV`. The 2024 files are `fase2_2026-10-07/datos/private_external_raw/RECH0_2024.SAV`, `RECH1_2024.SAV` and `RECH6_2024.SAV`. The 2025 support uses `datos/raw/ENDES2025/Base_Datos/RECH0_2025.SAV`, `RECH1_2025.SAV` and `RECH6_2025.SAV`.

Only file/schema/checksum metadata is supplied in `expected/private_csv_checksums.json` and `expected/raw_2025_checksums.json`. Access to official data is separate from permission to redistribute it.

## Auxiliary and reference trees

Actual-data preparation additionally needs the recorded district census predictors, geography and locked real-data basis. The author auxiliary route takes explicit `--author-aux-root` or `ENDES_AUX_ROOT` and the workbooks `consolidado_nuevo_completo_con_directos.xlsx` and `consolidado_censo_ubigeo_actualizado.xlsx`. The portable route accepts an external, checksum-verified district auxiliary CSV; that CSV is not bundled here and is not a new independent census extraction.

Simulation generation requires the original synthetic auxiliary template and scientific reference freezes. The correction stage requires the recorded original input/prediction/status trees. These are external prerequisites, not files created by installing Python dependencies. Historical freeze records that contain private filesystem locations remain in the private reference tree, read via `--reference-root`; they are not published.

To rerun exactly the archived comparison, obtain/materialize the corresponding reference inputs and verify their recorded hashes. To create a new run on another platform, register its source/executable/input identities before generation and report resulting differences honestly. The scientific code, seed rules and input preparation routines are available here; this repository is not a self-contained reproduction archive.

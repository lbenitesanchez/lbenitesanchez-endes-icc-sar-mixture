#!/usr/bin/env python3
"""Build an allowlisted public reference tree; no raw units or sampler draws."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import shutil
from legacy_runner import PHASE, metadata_files


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-root', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--resume', action='store_true', help='Verify and finish an incomplete identical copy')
    a = p.parse_args()
    source = a.reference_root.resolve()
    target = a.destination.resolve()
    if target.exists() and (not a.resume or (target/'LEGACY_REFERENCE_MANIFEST.json').exists()):
        raise FileExistsError('Preserve the existing public reference tree')
    relatives = set(metadata_files(source))
    relatives.update(Path('datos/processed')/name for name in
                     ('aux_all_districts.csv', 'areas_2019_6_59.csv', 'areas_2025_6_59.csv'))
    relatives.update((Path('datos/auxiliares/aux_all_districts.csv'), Path('datos/provenance/source_manifest.json')))
    for folder in ('resultados/resultados_comparadores/final_evaluation',
                   'resultados/resultados_comparadores/spatial_evaluation',
                   PHASE+'/resultados/design_simulation/final', PHASE+'/resultados/externa_final'):
        for f in (source/folder).glob('*.csv'):
            # External district-level response score details remain private.
            if folder.endswith('externa_final') and f.name not in ('summary.csv', 'contrasts_Holm8.csv'):
                continue
            relatives.add(f.relative_to(source))
    relatives.add(Path(PHASE)/'resultados/application2025_summary.csv')
    relatives.add(Path(PHASE)/'resultados/diseno/2019_full_official/cut16/cut8_comparison.json')
    for folder, names in {
        'application2025_baseline': ('predictions.csv', 'status.json'),
        'application2025_designcut8': ('district_predictions.csv', 'completion.json'),
        'externa_temporal_2019_baseline': ('predictions.csv', 'status.json'),
        'externa_temporal_2019_designcut8': ('district_predictions.csv', 'completion.json'),
        'externa_temporal_2019_hogg': ('predictions.csv', 'status_recovered.json'),
        'externa_territorial_2024_baseline': ('predictions.csv', 'status.json'),
        'externa_territorial_2024_designcut8': ('predictions.csv', 'completion.json'),
        'externa_territorial_2024_hogg': ('predictions.csv', 'status_recovered.json', 'status.json')}.items():
        relatives.update(Path(PHASE)/'resultados'/folder/name for name in names)
    for relative in list(relatives):
        if relative.is_absolute() or '..' in relative.parts or relative.suffix not in {'.csv', '.json'}:
            raise ValueError('Reference allowlist contains an unsafe/non-aggregate file: '+str(relative))
    forbidden_columns = {'child_id','household_id','HHID','psu_key','HV001','HV021','HC0','HC53','HC70',
                         'weight_anemia','weight_stunting'}
    records = []
    for relative in sorted(relatives):
        original = source/relative
        if not original.exists():
            alternate = source/'paquete_comparadores_publicacion_2026-10-06_entrega'/relative
            if alternate.exists():
                original = alternate
            else:
                continue
        if original.suffix == '.csv':
            with original.open(newline='',encoding='utf-8-sig') as stream:
                headers = next(csv.reader(stream), [])
            if forbidden_columns.intersection(headers):
                raise ValueError('Individual-record columns rejected: '+str(relative))
            if {'y_anemia','y_stunting'}.intersection(headers):
                # These explicitly allowlisted area files store district
                # ratios/variances, rather than children's binary responses.
                if relative.parent != Path('datos/processed') or not relative.name.startswith('areas_'):
                    raise ValueError('Unrecognized outcome-level data: '+str(relative))
                with original.open(newline='',encoding='utf-8-sig') as stream:
                    rows=list(csv.DictReader(stream))
                if len({r['ubigeo']for r in rows})!=len(rows) or not all(r['level']=='district' for r in rows):
                    raise ValueError('Expected unique aggregated district rows: '+str(relative))
        dest = target/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if hashlib.sha256(dest.read_bytes()).digest()!=hashlib.sha256(original.read_bytes()).digest():
                raise ValueError('Existing reference bytes differ: '+str(relative))
        else:
            shutil.copyfile(original,dest)
        records.append({'path':str(relative),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'bytes':dest.stat().st_size})
    (target/'LEGACY_REFERENCE_MANIFEST.json').write_text(json.dumps({
        'status':'PUBLIC_AGGREGATE_AND_PROTOCOL_REFERENCE', 'files':records,
        'individual_records_included':False,'raw_SAV_or_RDS_included':False,
        'posterior_draws_or_sampler_chains_included':False,
        'use':'Historical numerical targets and frozen protocol metadata; never substitute historical predictions for fresh refitting.'},indent=2)+'\n')
    print(json.dumps({'status':'PASS_ALLOWLIST_AND_CSV_PRIVACY', 'files':len(records),'destination':str(target)},indent=2))


if __name__ == '__main__':
    main()

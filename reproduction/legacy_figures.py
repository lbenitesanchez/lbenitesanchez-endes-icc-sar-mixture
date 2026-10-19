#!/usr/bin/env python3
"""Regenerate the four legacy figures from fresh completed results.

Uses the unchanged phase2_figures.py CLI. Figure A1 uses CUT40 aggregates;
A2 uses the WHO2024 external Holm8 contrasts; A3/S1 use the fresh 2025
ICC/CUT8 district predictions. CUT16/S4/S4a are tables, not bitmap figures.
Pixel equality is reported separately from numerical-result agreement and
does not replace visual inspection or article-table verification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

PHASE='fase2_2026-10-07'
FIGURES={
 'design_diagnostics':'Figure_A1',
 'external_contrasts':'Figure_A2',
 'application2025_ICC_capitals':'Figure_A3',
 'application2025_capitals':'Figure_S1',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study-root',type=Path,required=True)
    p.add_argument('--reference-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--export-dir',type=Path)
    p.add_argument('--inspect-existing',action='store_true',help='Inspect existing generated figures without recreating them')
    a=p.parse_args();study=a.study_root.resolve();script=study/PHASE/'scripts/phase2_figures.py'
    proof=json.loads((study.parent/'legacy_preparation.json').read_text())
    expected=proof['source_sha256'][PHASE+'/scripts/phase2_figures.py']
    if sha(script)!=expected:
        raise RuntimeError('Prepared figure source changed')
    if not a.inspect_existing:
        subprocess.run([sys.executable,str(script),'--only','all'],cwd=study,check=True)
    from PIL import Image
    import numpy as np
    records=[]
    for stem,label in FIGURES.items():
        actual=study/PHASE/'figuras'/f'{stem}.png'
        reference=a.reference_root/PHASE/'figuras'/f'{stem}.png'
        if not actual.is_file():
            raise FileNotFoundError('Fresh figure missing: '+str(actual))
        with Image.open(actual) as image:
            pixel=np.asarray(image.convert('RGBA'))
            size=list(image.size)
            dpi=list(image.info.get('dpi',()))
        record={'figure':label,'fresh_path':str(actual),'sha256':sha(actual),'pixels':size,'dpi':dpi}
        if reference.exists():
            with Image.open(reference)as image:
                old=np.asarray(image.convert('RGBA'))
                record['reference_pixels']=list(image.size)
            record['same_canvas_dimensions']=pixel.shape==old.shape
            if pixel.shape==old.shape:
                delta=np.abs(pixel.astype(np.int16)-old.astype(np.int16))
                record.update(exact_pixels_equal=bool(np.array_equal(pixel,old)),
                              mean_absolute_channel_difference=float(delta.mean()),
                              different_pixels=int(np.any(delta!=0,axis=2).sum()))
            record['reference_sha256']=sha(reference)
        else:
            record['reference_image_missing']=True
        if a.export_dir:
            a.export_dir.mkdir(parents=True,exist_ok=True)
            for suffix in ('png','svg'):
                source=actual.with_suffix('.'+suffix)
                destination=a.export_dir/(label+'.'+suffix)
                if destination.exists():
                    raise FileExistsError('Preserve the previous exported figure: '+str(destination))
                shutil.copyfile(source,destination)
        records.append(record)
    result={'status':'FOUR_FRESH_LEGACY_FIGURES_CREATED_VISUAL_QA_PENDING',
            'figure_source_sha256':expected,'figures':records,'model_fits_dispatched':False,
            'old_predictions_substituted':False,'CUT16_and_S4_are_tables':True,
            'numerical_comparison_required_separately':'legacy_compare.py',
            'pixel_or_file_hash_identity_is_not_the_only_reproduction_criterion':True,
            'article_reproduction_confirmed':False}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'figures':len(records),'output':str(a.output)},indent=2))


if __name__=='__main__':main()

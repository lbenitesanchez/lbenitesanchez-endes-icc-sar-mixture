#!/usr/bin/env python3
"""Refresh current public component/source hashes after final editorial integration.

This records source identity; it does not rerun production computations. The
historical dense/numerical checks retain their declared snapshot and scope.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,difflib,hashlib,json,re

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')

def refresh_public_manifests(public_root,analysis_root=None):
    public=Path(public_root).resolve();analysis=Path(analysis_root).resolve() if analysis_root is not None else None
    prov=public/'datos/provenance';source_file=prov/'analysis_public_source_manifest.json';component_file=prov/'analysis_component_manifest.json'
    source=read(source_file);before=read(component_file)
    before_sha=sha(component_file);prior_hashes={x['path']:x['sha256'] for x in before['files']}
    now=datetime.now(timezone.utc).isoformat()
    paths={x['path'] for x in source['source_files']}
    paths.update('codigo/'+name for name in ['06_build_documents.py','08_results_narrative.py','09_continuity_diagnostic.py'] if (public/'codigo'/name).exists())
    entries={x['path']:x for x in source['source_files']}
    for rel in sorted(paths):
        p=public/rel;e=entries.get(rel,{'path':rel,'model_math_API_changed':False})
        e['public_source_sha256']=sha(p)
        if analysis is not None:
            original=analysis/'scripts'/p.name
            if not original.exists():raise FileNotFoundError(original.name)
            e['analysis_source_sha256']=sha(original)
            e['changes']=list(difflib.unified_diff(original.read_text().splitlines(),p.read_text().splitlines(),n=3))
        entries[rel]=e
    if analysis is not None:
        original=(analysis/'scripts/models.py').read_text()
        adapted=original.replace('supplied survey residual degrees of freedom','supplied df parameter (may be a domain stability proxy)')
        if (public/'codigo/models.py').read_text()!=adapted:
            raise AssertionError('Public model differs beyond the audited descriptive df label')
    frozen=public/'codigo/provenance_models/models_full_v1.py'
    if sha(frozen)!='50d1ac0446d4a351a7cb50aa7a7146b1701575498b8764a4bd3c073d77cfbb02':
        raise AssertionError('Historical Gaussian/bootstrap model source changed')
    source['source_files']=[entries[p] for p in sorted(entries)]
    source['current_component_refreshed_utc']=now
    source['editorial_refresh_scope']='Current public source hashes/differences after final presentation and document integration; no model arithmetic change or new production run claimed.'
    write(source_file,source)
    validated=prov/'public_analysis_validation.json';v=read(validated)
    validated_model=prior_hashes.get('codigo/models.py')
    v['verification_scope']={
        'historical_checks':'Fresh-workspace wrapper, four dense models, partial and unsampled domains, acquisition guards and raw reconstruction checks were run before final figure/document editorial integration.',
        'validated_core_models_sha256':validated_model,
        'current_core_models_sha256':sha(public/'codigo/models.py'),
        'core_models_unchanged_from_validated_snapshot':validated_model==sha(public/'codigo/models.py'),
        'current_summary_source_sha256':sha(public/'codigo/04_summarize_simulations.py'),
        'subsequent_changes':'Final summary legends, typography, manuscript wording/format and source packaging. Core numerical validation remains applicable; full simulations were not repeated for these changes.',
        'current_component_refresh_utc':now}
    if not v['verification_scope']['core_models_unchanged_from_validated_snapshot']:
        raise AssertionError('Dense checks would require new validation of changed core model')
    write(validated,v)
    # Audit after refreshing source differences so new metadata cannot quietly
    # reintroduce a personal filesystem path from the original working source.
    personal=[];raw=[]
    personal_pattern=re.compile(r"/(?:Users)/[^/\s]+/|[A-Za-z]:\\Users\\[^\\\s]+\\")
    text_suffixes={'.json','.jsonl','.md','.txt','.log','.py','.r','.csv','.sh'}
    for p in public.rglob('*'):
        if not p.is_file():continue
        rel=str(p.relative_to(public))
        if p.suffix.lower() in {'.sav','.rds','.zip'}:raw.append(rel)
        if p.suffix.lower() in text_suffixes and personal_pattern.search(p.read_text(errors='replace')):
            personal.append(rel)
    if personal or raw:raise AssertionError({'personal_path_files':personal,'raw_record_files':raw})
    write(prov/'public_component_privacy_audit.json',{'audit_utc':now,'after_source_provenance_refresh':True,
          'current_workstation_user_paths_excluded':True,'raw_SAV_RDS_ZIP_excluded':True,
          'scope':'Text components checked for macOS/Windows user-directory paths; actual SAV/RDS/ZIP records prohibited. Archived third-party source README retains its published Linux example paths. Final archive audit is separate.'})
    # The data-only component remains unchanged. Current analysis component
    # covers code/data/readmes; the final package_manifest covers result/docx.
    current=[]
    for p in sorted(public.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts:continue
        rel=p.relative_to(public)
        include=rel.parts[0] in ['codigo','datos'] or str(rel) in ['README_ANALISIS_ES_EN.md','README_DATOS_ES_EN.md','requirements.txt','reproducir_analisis.sh','validacion/dense_equivalence.json']
        if include and p!=component_file:current.append({'path':str(rel),'sha256':sha(p),'bytes':p.stat().st_size})
    component={
        'scope':'Current portable code/data/readme component after final source integration. Results and documents are indexed in package_manifest.json.',
        'refreshed_utc':now,'files':current,'validation':v,
        'historical_component_snapshot':before.get('historical_component_snapshot',{'manifest_sha256_before_refresh':before_sha,'scope':'Earlier component snapshot; its source hashes are archival evidence, not assertions about updated presentation files.','earlier_summary_source_sha256':prior_hashes.get('codigo/04_summarize_simulations.py')}),
        'production_computations_repeated_for_editorial_changes':False}
    write(component_file,component)
    mismatch=[x['path'] for x in current if sha(public/x['path'])!=x['sha256']]
    if mismatch:raise AssertionError(mismatch)
    return {'refreshed':True,'component_files':len(current),'source_files':len(entries),'current_summary_sha256':sha(public/'codigo/04_summarize_simulations.py'),'core_models_unchanged_from_validated_snapshot':v['verification_scope']['core_models_unchanged_from_validated_snapshot'],'component_hashes_match':True,'privacy_after_refresh_passed':True}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--public-root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--analysis-root',type=Path,default=None,help='Optional local analysis source root for refreshing source differences; no private paths are saved.')
    args=p.parse_args();print(json.dumps(refresh_public_manifests(args.public_root,args.analysis_root),indent=2))

if __name__=='__main__':main()

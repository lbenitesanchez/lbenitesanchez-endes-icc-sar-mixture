"""Create a fresh reproduction tree; --execute runs the locked experiment.

No finished output tree is overwritten. Dependencies are shared through local
directory symlinks; statistical sources and recorded freeze hashes stay intact.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def spatial_driver(root, base, driver):
    """Use an explicit approved numerical amendment; preserve original drivers."""
    amendment = root/'resultados/competidores_hogg/hogg_spatial_numerical_amendment.json'
    if not amendment.exists():
        return [str(base/f'hogg_spatial_{driver}.py')]
    record = json.loads(amendment.read_text())
    if not record.get('approved_after_independent_audit_and_pilot'):
        raise ValueError('The optional spatial initializer has not passed its independent pilot/amendment.')
    adapter = base/'hogg_spatial_numeric.py'
    if hashlib.sha256(adapter.read_bytes()).hexdigest() != record['adapter_sha256']:
        raise ValueError('The spatial initializer differs from its approved numerical amendment.')
    return [str(adapter), '--driver', driver]


def copy_spatial_amendment_evidence(root, target, record):
    """Copy every referenced proof only after checking relative paths and SHA."""
    verified=[]
    for prefix in ('independent_audit', 'numerical_pilot'):
        rel=Path(record[prefix+'_path'])
        if rel.is_absolute() or '..' in rel.parts or not rel.parts:
            raise ValueError('Spatial amendment evidence must use safe relative paths.')
        source=(root/rel).resolve()
        if not source.is_relative_to(root.resolve()) or not source.is_file():
            raise ValueError('Spatial amendment evidence is missing or outside the package.')
        if hashlib.sha256(source.read_bytes()).hexdigest()!=record[prefix+'_sha256']:
            raise ValueError('Spatial amendment evidence differs from its approved SHA: '+prefix)
        destination=target/rel
        if not destination.resolve().is_relative_to(target.resolve()):
            raise ValueError('Spatial evidence destination is outside the fresh workspace.')
        verified.append((source,destination))
    for source,destination in verified:
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--workers', type=int, default=1)
    a = p.parse_args()
    # Reviewed archives may contain explicitly path-redacted operational copies.
    # Validate that public layer before making a fresh tree; do not rewrite or
    # pretend to match the archived original checksum of those different bytes.
    import public_copy_provenance
    if (ROOT/public_copy_provenance.MANIFEST).exists():
        public_copy_provenance.verify_manifest(ROOT)
        historical=json.loads((ROOT/'resultados/resultados_comparadores/spatial_result_contract.json').read_text())
        public_copy_provenance.verify_public_reference(ROOT,historical['task_file'],historical['task_sha256'])
    target = a.directory.resolve()
    if target.exists():
        raise FileExistsError('Choose a new reproduction directory; existing results are preserved.')
    if target == ROOT or target.is_relative_to(ROOT):
        raise ValueError('Use a sibling or separate directory outside the reviewed package.')
    target.mkdir(parents=True)
    for name in ('scripts', 'codigo', 'datos/processed', 'datos/auxiliares', 'datos/provenance'):
        shutil.copytree(ROOT / name, target / name)
    (target/'manuscrito').mkdir()
    for name in ['references.json','sections.json']:
        shutil.copy2(ROOT/'manuscrito'/name,target/'manuscrito'/name)
    for source in (ROOT/'manuscrito').glob('competitor*.json'):
        shutil.copy2(source,target/'manuscrito'/source.name)
    (target/'validacion').mkdir()
    for name in ['beta_power_independent_audit.json','competitor_protocol_independent_review.json',
                 'hogg_spatial_pilot_independent_review.json','competitor_population_rng_independence_audit.json']:
        shutil.copy2(ROOT/'validacion'/name,target/'validacion'/name)
    # Preserve source-only QA separately. The real arithmetic report must be
    # produced afresh from the new execution, rather than copying its old PASS.
    qa=ROOT/'resultados/resultados_comparadores/validation/independent_final_audit_qa.json'
    if qa.exists():
        target_qa=target/qa.relative_to(ROOT)
        target_qa.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(qa,target_qa)
    for name in ('.competitor_env', '.Rlib'):
        if (ROOT / name).exists():
            (target / name).symlink_to(ROOT / name, target_is_directory=True)
    for name in ('finite_v2', 'sparse_v2'):
        src = ROOT / 'resultados/simulaciones/source_snapshots' / name
        shutil.copytree(src, target / 'resultados/simulaciones/source_snapshots' / name)
    freezes = ['resultados/resultados_comparadores/protocol_freeze.json',
               'resultados/resultados_comparadores/candidate_frozen_manifest.json',
               'resultados/competidores_gao/gao_frozen_code_manifest.json',
               'resultados/competidores_hogg/hogg_sampler_freeze.json',
               'resultados/competidores_hogg/hogg_strong_arms_freeze.json',
               'resultados/competidores_hogg/hogg_spatial_freeze.json']
    for name in freezes:
        dst = target / name; dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dst)
    spatial_amendment='resultados/competidores_hogg/hogg_spatial_numerical_amendment.json'
    if (ROOT/spatial_amendment).exists():
        # Verify before publishing an executable recipe; an unapproved draft
        # must never authorize itself through a public reproduction wrapper.
        spatial_driver(ROOT, ROOT/'scripts/competitors', 'batch')
        copy_spatial_amendment_evidence(ROOT,target,json.loads((ROOT/spatial_amendment).read_text()))
        shutil.copy2(ROOT/spatial_amendment, target/spatial_amendment)
        for name in ['hogg_spatial_numerical_initializer_independent_review.json',
                     'hogg_spatial_numerical_initializer_independent_evidence.json']:
            shutil.copy2(ROOT/'validacion'/name, target/'validacion'/name)
    for name in ['final_result_contract.json','numerical_amendment_v2.json','candidate_frozen_manifest_v2.json',
                 'protocol_numerical_v2_approved.json']:
        shutil.copy2(ROOT/'resultados/resultados_comparadores'/name,target/'resultados/resultados_comparadores'/name)
    archive='resultados/resultados_comparadores/protocol_archive/v1_before_numerical_repair'
    if (ROOT/archive).exists():shutil.copytree(ROOT/archive,target/archive)
    spatial_history='resultados/resultados_comparadores/spatial_execution_contract_archive/v1_before_initializer_adapter'
    if (ROOT/spatial_history).exists():shutil.copytree(ROOT/spatial_history,target/spatial_history)
    (target/'fresh_operational_provenance.json').write_text(json.dumps({
        'approved_statistical_sources_and_evidence':'Copied byte-exact; source manifests retain approved SHA values',
        'historical_public_task_tables':'Not copied into the new execution tree',
        'new_execution_task_paths':'Generated locally by the unchanged task preparation/evaluation programs',
        'new_execution_contract':'Generated locally for the same statistical settings and fixed IDs; distinct from the historical contract',
        'replacement_population_IDs':False,'statistical_configuration_changed':False},indent=2))
    contract=json.loads((target/'resultados/resultados_comparadores/final_result_contract.json').read_text())
    primary_tag=next(t for t in contract['final_tags']if t.startswith('final_primary'))
    structural_tag=next(t for t in contract['final_tags']if t.startswith('final_structural'))
    py = target / '.competitor_env/bin/python'
    base = target / 'scripts/competitors'
    out = target / 'resultados/resultados_comparadores'
    commands = [
        [str(py), str(base/'benchmark_prepare_final.py'), '--workers', str(a.workers)],
        [str(py), str(base/'benchmark_run.py'), '--tag', primary_tag, '--workers', str(a.workers),
         '--scenarios', 'binary_informative_mar', 'sparse_srs_2025', 'sparse_psu_2025',
         '--replications', *map(str, range(10000, 10200))],
        [str(py), str(base/'benchmark_run.py'), '--tag', structural_tag, '--workers', str(a.workers),
         '--scenarios', 'sparse_srs_iid', 'sparse_psu_bym2', '--replications', *map(str, range(20000, 20100))],
        [str(py), str(base/'gao_dispatch_final.py'), '--tasks', str(out/'task_plan/gao_all1800.csv'),
         '--batch-label', 'public_reproduction_gao', '--workers', str(a.workers)],
        [str(py), str(base/'benchmark_collect.py'), '--tasks', str(out/'task_plan/gao_all1800.csv'),
         '--tag', 'final_gao_evaluation_v1'],
        [str(py), '-c', 'import sys;sys.path.insert(0,"scripts/competitors");import hogg_tsln as h;h.compile_models();import hogg_design_extension as e;e.compile_extension()'],
        [str(py), str(base/'hogg_benchmark_batch.py'), '--tasks', str(out/'task_plan/hogg_all1100.csv'),
         '--tag', 'public_reproduction_hogg', '--workers', str(a.workers)],
        [str(py), str(target/'scripts/public_competitors/hogg_collect_tasks.py'),
         str(target/'resultados/competidores_hogg/public_reproduction_hogg/task_output_contract.csv'),
         str(out/'task_plan/hogg_collector1100.csv')],
        [str(py), str(base/'benchmark_collect.py'), '--tasks',
         str(out/'task_plan/hogg_collector1100.csv'),
         '--tag', 'final_hogg_evaluation_v1'],
        [str(py), str(base/'benchmark_finalize.py')],
        [str(py), str(base/'benchmark_spatial_evaluate.py'), '--prepare-only'],
        *[[str(py), *spatial_driver(target, base, 'batch'), '--start', str(20000+25*k),
           '--stop', str(20000+25*(k+1)), '--tag', f'spatial_sensitivity_{k:02d}'] for k in range(4)],
        [str(py), str(base/'benchmark_schedule_spatial_evaluation.py')],
        [str(py), str(base/'benchmark_validate_final_results.py'), '--wait', '--poll-seconds', '30'],
        [str(py), str(base/'benchmark_boundary_audit.py'), '--wait', '--include-spatial'],
        [str(py), str(base/'benchmark_independent_final_audit.py')],
    ]
    (target/'reproduction_commands.json').write_text(json.dumps(commands, indent=2))
    print(f'Fresh tree: {target}. Commands saved; 800 populations, no replacement seeds.')
    if a.execute:
        if not py.exists() or not (target/'.Rlib').exists():
            raise RuntimeError('Install local Python/CmdStan and R/INLA dependencies in the package first.')
        for command in commands:
            subprocess.run(command, cwd=target, check=True)


if __name__ == '__main__':
    main()

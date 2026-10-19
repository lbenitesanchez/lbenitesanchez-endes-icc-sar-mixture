"""Code-only numerical table contract from external CSV/JSON aggregates.

This publication adapter retains original table-formatting mathematics. It
contains no manuscript text, document reader, bibliography or editorial QA.
No input data, generated matrices or private records are distributed here.
"""
from __future__ import annotations
import hashlib
import importlib
import json
from pathlib import Path, PurePosixPath
import sys

PHASE = 'fase2_2026-10-07'

SIKOV = 'sikov_2026-10-10'

MAIN_IDS = ('1','2','3','4','5','6','7','8','A1','A2','A3')

MAIN_CSV = (
 'resultados/published_sar_comparison/paired_balanced_contrasts_Holm3.csv',
 'resultados/published_sar_comparison/physical_projection_contrasts.csv',
 'resultados/published_sar_comparison/summary.csv',
 'resultados/external_comparison_Holm12/paired_balanced_contrasts_Holm12.csv',
 'validacion/sikov_B300_descriptive_balanced.csv',
 'resultados/published_b300_descriptive/summary.csv',
 'resultados/published_b300_descriptive/selected_reference_differences.csv',
 'resultados/poverty_stratum_descriptive/selected_reference_descriptive_differences.csv',
 'sensibilidad_sampled_graph/resultados/descriptive_summary.csv',
 'sensibilidad_sampled_graph/resultados/descriptive_paired_contrasts.csv')

LEGACY_CSV = (
 PHASE+'/resultados/design_simulation/final/summary.csv',
 PHASE+'/resultados/externa_final/contrasts_Holm8.csv',
 PHASE+'/resultados/application2025_summary.csv',
 PHASE+'/resultados/design_simulation/final/paired_diagnostics.csv',
 PHASE+'/resultados/externa_final/summary.csv',
 'resultados/resultados_comparadores/final_evaluation/primary_balanced_interval_score.csv',
 'resultados/resultados_comparadores/final_evaluation/additional_arms_summary.csv',
 'resultados/resultados_comparadores/final_evaluation/structural_summary.csv',
 'resultados/resultados_comparadores/spatial_evaluation/summary.csv',
 'resultados/resultados_comparadores/final_evaluation/combined_summary.csv',
 'resultados/resultados_comparadores/boundary_audit/population_boundaries.csv')

CUT16 = PHASE+'/resultados/diseno/2019_full_official/cut16/cut8_comparison.json'

FIGURES = (
 ('manuscript','1',SIKOV+'/resultados/figuras/matched_rmse_coverage.png'),
 ('manuscript','2',SIKOV+'/resultados/figuras/matched_interval_score.png'),
 ('manuscript','3',SIKOV+'/resultados/aplicacion2019/matched_application2019_pointmaps.png'),
 ('manuscript','A1',PHASE+'/figuras/design_diagnostics.png'),
 ('manuscript','A2',PHASE+'/figuras/external_contrasts.png'),
 ('manuscript','A3',PHASE+'/figuras/application2025_ICC_capitals.png'),
 ('supplement','S1',PHASE+'/figuras/application2025_capitals.png'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def safe(root, relative):
    rel = PurePosixPath(relative)
    if rel.is_absolute() or '..' in rel.parts or '\\' in relative:
        raise ValueError('Unsafe relative artifact path')
    root = Path(root).resolve();path = root / relative
    if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError('Missing, linked or escaped current artifact: '+relative)
    if path.stat().st_size > 100_000_000:
        raise ValueError('Oversized current artifact')
    return path

def kernel_manifest(repo):
    source = Path(repo)/'src/estudio_publicacion'
    return {p.relative_to(source).as_posix():sha(p) for p in sorted(source.rglob('*'))
            if p.is_file() and p.suffix.lower() in ('.py','.r','.stan')}

def formatting_manifest(repo):
    paths = ('reproduction/article_coverage.py','reproduction/legacy_compare.py',
             'src/estudio_publicacion/'+SIKOV+'/scripts/report_documents.py')
    return {rel:sha(safe(repo,rel)) for rel in paths}

def functions(repo):
    directory = str((Path(repo)/'reproduction').resolve())
    if directory not in sys.path:sys.path.insert(0,directory)
    coverage = importlib.import_module('article_coverage')
    if Path(coverage.__file__).resolve() != Path(directory)/'article_coverage.py':
        raise ValueError('Unexpected formatting adapter import')
    report, _ = coverage.load_report(Path(repo)/'src/estudio_publicacion')
    return coverage, report


def derive(repo, data_root):
    """Construct the same 22 numerical tables; omit all paper narrative text."""
    coverage, report = functions(repo)
    data_root = Path(data_root)
    data = coverage.main_inputs(data_root / SIKOV)
    main = coverage.main_tables(report, data) + coverage.appendix_tables(data_root)
    supplement = coverage.supplement_tables(data_root)
    tables = [dict(family=family, id=ident, rows=rows, cells=sum(map(len, rows)))
              for family, ids, rows_list in (
                  ('manuscript', MAIN_IDS, main),
                  ('supplement', coverage.SUPPLEMENT_IDS, supplement))
              for ident, rows in zip(ids, rows_list)]
    if len(main) != 11 or len(supplement) != 11 or sum(t['cells'] for t in tables) != 1488:
        raise ValueError('Aggregate formatting changed the 22-table/1488-cell contract')
    paths = sorted(set([SIKOV + '/' + relative for relative in MAIN_CSV] +
                       list(LEGACY_CSV) + [CUT16]))
    return {'tables': tables,
            'narratives': [], 'narrative_predicates': {},
            'dynamic_narrative_count': 0, 'input_empirical_narrative_count': 0,
            'legacy_empirical_values': {},
            'figure_paths': [{'family': family, 'id': ident, 'path': path}
                             for family, ident, path in FIGURES],
            'data_inputs_sha256': {relative: sha(safe(data_root, relative))
                                  for relative in paths},
            'code_only_export': True,
            'paper_text_or_document_outputs_included': False}

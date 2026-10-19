"""Scientific figures from complete, frozen comparator result tables."""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figuras'
ALIASES = {'GaoAuthorStrict': 'GaoStrict', 'GaoCommonContinuityX': 'GaoCommon', 'HoggTSLN': 'Hogg'}
ORDER = ['FH', 'MFH', 'SFH', 'MSFH', 'GaoStrict', 'GaoCommon', 'Hogg', 'BetaPowerHMCGraph']
LABELS = ['FH', 'MFH', 'SFH', 'MSFH', 'Gao strict', 'Gao common', 'Hogg generic', 'Beta power']
SCENARIOS = ['binary_informative_mar', 'sparse_srs_2025', 'sparse_psu_2025']
CONTEXT = ['Informative sampling', 'Sparse simple random sampling', 'Sparse primary units']
COLORS = ['#777777', '#777777', '#777777', '#777777', '#386898', '#386898', '#386898', '#A23D2A']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.labelsize': 10,
                     'axes.titlesize': 11, 'axes.spines.top': False, 'axes.spines.right': False})


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(OUT / f'{name}.{extension}', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def primary_figure(frame, metric, error, name):
    fig, axes = plt.subplots(3, 2, figsize=(7.5, 8.2), sharey=True, constrained_layout=True)
    for r, scenario in enumerate(SCENARIOS):
        for c, outcome in enumerate(('anemia', 'stunting')):
            ax = axes[r, c]
            part = frame[(frame.scenario == scenario) & (frame.outcome == outcome) & (frame.group == 'all')]
            part = part.set_index('method').loc[ORDER]
            if not (part.replications == 200).all() or not (part.mean_areas == 80).all():
                raise RuntimeError('An incomplete cohort cannot enter a publication figure')
            x, dx = 100 * part[metric].to_numpy(), 1.95996398454 * 100 * part[error].to_numpy()
            for k, colour in enumerate(COLORS):
                ax.errorbar(x[k], k, xerr=dx[k], fmt='o', color=colour, markersize=4,
                            capsize=2, linewidth=1, zorder=3)
            ax.set_yticks(np.arange(len(ORDER)), LABELS)
            ax.set_ylim(7.7, -.7)
            ax.grid(axis='x', color='#E3E3E3', linewidth=.6, zorder=0)
            ax.set_title(f'{CONTEXT[r]}\n{outcome.capitalize()}')
            ax.set_xlabel('RMSE in percentage points' if metric == 'rmse' else 'Coverage in percent')
            if metric == 'coverage':
                ax.axvline(95, color='black', linestyle='--', linewidth=.8)
                ax.set_xlim(max(0, 10*np.floor((np.min(x-dx)-3)/10)), 101)
            else:
                ax.set_xlim(left=0)
    save(fig, name)


def candidate_map(config):
    folder = ROOT / config['candidate_application_path']
    pred = pd.read_csv(folder / 'district_predictions.csv', dtype={'ubigeo': str})
    areas = pd.read_csv(ROOT / 'datos/processed/aux_all_districts.csv', dtype={'ubigeo': str})
    # Author extraction files may expose the coordinates under a different name;
    # do not guess a positional join or silently substitute another geography.
    if not {'latitude', 'longitude', 'ubigeo'}.issubset(areas):
        raise ValueError('Verified district coordinates unavailable')
    merged = pred.merge(areas[['ubigeo', 'latitude', 'longitude']], on='ubigeo', validate='one_to_one')
    if len(merged) != 1874: raise ValueError('Prediction geography incomplete')
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 7.7), constrained_layout=True)
    for r, outcome in enumerate(('anemia', 'stunting')):
        for c, quantity in enumerate(('mean', 'width')):
            ax = axes[r, c]
            values = 100 * (merged[f'p_mean_{outcome}'] if quantity == 'mean' else
                            merged[f'hi95_{outcome}'] - merged[f'lo95_{outcome}'])
            point = ax.scatter(merged.longitude, merged.latitude, c=values, s=3,
                               cmap='viridis' if quantity == 'mean' else 'magma', rasterized=True)
            ax.set_title(f'{outcome.capitalize()}\n' + ('Posterior mean' if quantity == 'mean' else '95% interval width'))
            ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
            ax.set_aspect('equal', adjustable='box')
            fig.colorbar(point, ax=ax, label='Percent' if quantity == 'mean' else 'Percentage points', shrink=.85)
    save(fig, 'competitor_candidate_uncertainty')


def application_contrasts(config):
    """Display the complete exploratory family, including richer information."""
    frame = pd.read_csv(ROOT / config['application_paired_path'])
    references = ORDER[:-1] + ['HoggAgeSex', 'HoggAgeSexSpatial']
    labels = dict(zip(ORDER[:-1], LABELS[:-1]))
    labels.update(HoggAgeSex='Hogg age/sex', HoggAgeSexSpatial='Hogg age/sex BYM2')
    expected = {(year, method, outcome) for year in (2019, 2025)
                for method in (references[:7] if year == 2019 else references)
                for outcome in ('anemia', 'stunting')}
    actual = list(frame[['year', 'reference', 'outcome']].itertuples(index=False, name=None))
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('The territorial contrast figure requires the complete declared family')
    values = frame[['binary_NLL_difference', 'binary_NLL_ci_low', 'binary_NLL_ci_high']].to_numpy()
    if not np.isfinite(values).all() or not (frame.bootstrap_repetitions == 2000).all():
        raise ValueError('Invalid territorial contrast values or bootstrap size')
    if not frame.multiplicity.eq('exploratory_unadjusted').all():
        raise ValueError('Territorial comparisons must retain their exploratory label')
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 8.2), sharex=True, constrained_layout=True)
    for r, year in enumerate((2019, 2025)):
        selected_references = references[:7] if year == 2019 else references
        for c, outcome in enumerate(('anemia', 'stunting')):
            ax = axes[r, c]
            part = frame[(frame.year == year) & (frame.outcome == outcome)].set_index('reference')
            expected_support = (178, 36) if year == 2019 else (184, 37)
            if not (part.districts == expected_support[0]).all() or not (part.provinces == expected_support[1]).all():
                raise ValueError('Territorial contrast support differs across methods')
            for k, reference in enumerate(selected_references):
                row = part.loc[reference]
                colour = '#A23D2A' if reference.startswith('HoggAgeSex') else '#386898'
                ax.plot([row.binary_NLL_ci_low, row.binary_NLL_ci_high], [k, k],
                        color=colour, linewidth=1.1)
                ax.plot(row.binary_NLL_difference, k, 'o', color=colour, markersize=4)
            ax.set_yticks(range(len(selected_references)), [labels[name] for name in selected_references])
            ax.set_ylim(len(selected_references)-.3, -.7)
            ax.axvline(0, color='black', linestyle='--', linewidth=.8)
            ax.grid(axis='x', color='#E3E3E3', linewidth=.6)
            ax.set_title(f'{year}: {outcome.capitalize()}\n{expected_support[0]} districts, {expected_support[1]} provinces')
            ax.set_xlabel('Beta minus comparator binary score')
    save(fig, 'competitor_application_contrasts')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', default='manuscrito/competitor_document_inputs.json')
    args = parser.parse_args(); config = json.loads((ROOT / args.config).read_text())
    frame = pd.read_csv(ROOT / config['summary_path']); frame['method'] = frame.method.replace(ALIASES)
    primary_figure(frame, 'rmse', 'rmse_mcse', 'competitor_primary_rmse')
    primary_figure(frame, 'coverage', 'coverage_mcse', 'competitor_primary_coverage')
    candidate_map(config)
    application_contrasts(config)
    print(json.dumps({'figures': 4, 'resolution_dpi': 300,
                      'primary_error_bars': '95% Monte Carlo intervals',
                      'application_error_bars': 'exploratory unadjusted paired province-bootstrap intervals'}))


if __name__ == '__main__':
    main()

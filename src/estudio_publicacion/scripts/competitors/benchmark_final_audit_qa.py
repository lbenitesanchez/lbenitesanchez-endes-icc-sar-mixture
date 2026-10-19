#!/usr/bin/env python3
"""Synthetic, hand-calculated tests for the independent final auditor."""
import json
import math
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import benchmark_independent_final_audit as audit


def run():
    theta = np.column_stack([np.r_[np.zeros(40), np.ones(40)], np.full(80, .5)])
    prediction = dict(mean=np.column_stack([np.full(80, .5), np.full(80, .6)]),
        lower=np.column_stack([np.full(80, .25), np.full(80, .4)]),
        upper=np.column_stack([np.full(80, .75), np.full(80, .8)]))
    masks = {'all': np.ones((80, 2), bool)}
    task = dict(scenario='synthetic_control', replication=1, method='Test')
    rows = pd.DataFrame(audit.independent_rows(task, theta, masks, prediction, [True, False], 2))
    anemia = rows[rows.outcome == 'anemia'].iloc[0]; stunting = rows[rows.outcome == 'stunting'].iloc[0]
    np.testing.assert_allclose(anemia[['bias', 'mse', 'mae', 'coverage', 'width', 'interval_score']].to_numpy(float), [0, .25, .5, 0, .5, 10.5])
    np.testing.assert_allclose(stunting[['bias', 'mse', 'mae', 'coverage', 'width', 'interval_score']].to_numpy(float), [.1, .01, .1, 1, .4, .4])
    assert not stunting.converged and stunting.areas == 80
    control = []
    for i in range(3):
        control.append(dict(scenario='control', replication=i, method='BetaPowerHMCGraph', protocol='test', group='all', outcome='anemia',
            bias=.1 * i, mse=.01 * (i + 1) ** 2, mae=.1 * (i + 1), coverage=.2 * (i + 1), width=.2 + .1 * i,
            interval_score=i + 1., areas=80, areas_requested=80, availability_rate=1., converged=i != 0, fit_seconds=1.))
    frame = pd.DataFrame(control); summary = audit.summaries(frame).iloc[0]
    np.testing.assert_allclose([summary.bias, summary.bias_mcse, summary.rmse, summary.rmse_mcse],
        [.1, .1 / math.sqrt(3), math.sqrt(.14 / 3), 7 / (600 * math.sqrt(.14 / 3))])
    reference = frame.copy(); reference['method'] = 'Reference'; reference['bias'] = 0.; reference['mse'] = .01
    reference['coverage'] = .2; reference['interval_score'] = .5
    pair = audit.contrasts(pd.concat([frame, reference]), ['Reference']).iloc[0]
    np.testing.assert_allclose([pair.bias_change, pair.bias_change_mcse, pair.interval_score_change,
        pair.interval_score_change_mcse, pair.log_rmse_ratio_mcse], [.1, .1 / math.sqrt(3), 1.5, 1 / math.sqrt(3), .25])
    difference, se = audit.balanced_statistics([[1, 3], [2, 4], [0, 2]])
    np.testing.assert_allclose([difference, se], [2, 1 / math.sqrt(3)])
    np.testing.assert_allclose(audit.adjusted_holm3([.01, .04, .03]), [.03, .06, .06])
    numeric = audit.prediction_boundaries(task, theta, masks, prediction, True)
    assert numeric[0]['truth_zero_requested'] == 40 and numeric[0]['truth_one_requested'] == 40
    assert numeric[0]['truth_zero_covered'] == numeric[0]['truth_one_covered'] == 0
    missing = {key: np.full((80, 2), np.nan) for key in prediction}
    unavailable = audit.prediction_boundaries(task, theta, masks, missing, False)[0]
    assert unavailable['requested_area_cells'] == 80 and unavailable['available_area_cells'] == 0
    actual_out = audit.OUT
    with tempfile.TemporaryDirectory() as temporary:
        synthetic_folder = Path(temporary) / 'predictions'
        synthetic_folder.mkdir()
        output = pd.DataFrame({'area': np.arange(80)})
        for column, name in enumerate(audit.OUTCOMES):
            for key, prefix in [('mean', 'p_mean'), ('lower', 'lo95'), ('upper', 'hi95')]:
                output[prefix + '_' + name] = prediction[key][:, column]
        output.to_csv(synthetic_folder / 'predictions.csv', index=False)
        # The descriptive boundary audit may use an exported numeric CSV even
        # when its sampler-status artifact is unavailable to the metric audit.
        csv_only = audit.read_prediction_csv(synthetic_folder)
        assert audit.prediction_boundaries(task, theta, masks, csv_only, True)[0]['available_area_cells'] == 80
        try:
            audit.read_prediction({'output_dir': str(synthetic_folder)})
        except FileNotFoundError:
            pass
        else:
            raise AssertionError('Metric audit must not fabricate a missing sampler status')
        (synthetic_folder / 'status.json').write_text(json.dumps({'endpoints': [{'success': True}, {'success': False}], 'fit_seconds': 7}))
        loaded, converged, seconds = audit.read_prediction({'output_dir': str(synthetic_folder)})
        assert converged == [True, False] and seconds == 7
        assert len(audit.independent_rows(task, theta, masks, loaded, converged, seconds)) == 2
        output['area'] = output.area.astype(float)
        output.loc[0, 'area'] = .25
        output.to_csv(synthetic_folder / 'predictions.csv', index=False)
        try:
            audit.read_prediction_csv(synthetic_folder)
        except ValueError:
            pass
        else:
            raise AssertionError('Fractional area identifiers cannot silently be truncated')
        audit.OUT = Path(temporary)
        try:
            for stage in ['neither', 'main_only', 'both']:
                if stage != 'neither':
                    path = audit.OUT / 'final_evaluation/final_completion.json'; path.parent.mkdir(exist_ok=True)
                    path.write_text(json.dumps({'evaluation_completed': True}))
                if stage == 'both':
                    path = audit.OUT / 'spatial_evaluation/completion.json'; path.parent.mkdir(exist_ok=True)
                    path.write_text(json.dumps({'evaluation_completed': True}))
                try:
                    audit.finished()
                except RuntimeError:
                    assert stage != 'both'
                else:
                    assert stage == 'both'
        finally:
            audit.OUT = actual_out
    proof = dict(passed=True, manual_point_interval_values_agree=True, between_population_MCSE_and_delta_RMSE_agree=True,
        paired_influence_and_interval_score_MCSE_agree=True, three_context_strata_formula_agrees=True,
        Holm3_manual_values_agree=True, finite_nonconvergence_retained=True,
        exact_boundary_missing_denominators_retained=True, both_final_markers_required=True,
        numeric_CSV_boundary_accounting_distinct_from_status_eligibility=True,
        missing_status_not_fabricated_and_finite_nonconvergence_retained=True,
        nonintegral_prediction_area_identifier_rejected=True,
        no_actual_truth_predictions_or_performance_read=True, no_fit_or_statistical_source_modified=True,
        independent_auditor_sha256=audit.sha256(Path(audit.__file__)), qa_sha256=audit.sha256(Path(__file__)))
    destination = audit.OUT / 'validation/independent_final_audit_qa.json'
    destination.write_text(json.dumps(proof, indent=2)); return proof


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))

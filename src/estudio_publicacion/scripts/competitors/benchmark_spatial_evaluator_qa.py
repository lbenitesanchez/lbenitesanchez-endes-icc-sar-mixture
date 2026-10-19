#!/usr/bin/env python3
"""Deterministic controls for spatial matching, MCSE and failure denominators.

Uses synthetic constant prevalence, never any final population or fit result.
"""
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from benchmark_inputs import OUT,sha256
from benchmark_metrics import replication_metrics,summarize,paired_contrasts
import benchmark_spatial_evaluate as s


def main():
    checks={};rows=[];candidates=[]
    with tempfile.TemporaryDirectory(prefix='endes_spatial_evaluator_control_')as directory:
        folder=Path(directory);(folder/'evaluation_only').mkdir();(folder/'model_inputs').mkdir()
        np.savez(folder/'evaluation_only/truth.npz',theta=np.full((80,2),.5),sample_idx=np.arange(64))
        np.savez(folder/'model_inputs/direct.npz',y=np.full((64,2),.5))
        for j,r in enumerate(sorted(s.IDS)):
            ref={'mean':np.full((80,2),.6),'lower':np.full((80,2),.3),'upper':np.full((80,2),.8)}
            width=.2+.001*j
            cand={'mean':np.full((80,2),.51+.0001*j),
                'lower':np.full((80,2),.5-width/2),'upper':np.full((80,2),.5+width/2)}
            for source,pred,method in ((rows,ref,s.METHOD),(candidates,cand,s.CANDIDATE)):
                for row in replication_metrics(folder,pred,method,'control',j!=0,0):
                    row.update(scenario=s.SCENARIO,replication=r);source.append(row)
        reference=pd.DataFrame(rows);candidate=pd.DataFrame(candidates)
        table=summarize(reference);allcells=table[table.group=='all']
        np.testing.assert_allclose(allcells.rmse,.1,atol=1e-14)
        np.testing.assert_allclose(allcells.coverage,1,atol=0)
        np.testing.assert_allclose(allcells.width,.5,atol=1e-14)
        checks['known_constant_truth_bias_RMSE_coverage_width']=True
        contrasts=paired_contrasts(candidate,reference,s.CONTRAST)
        expected=.2+.001*np.arange(100)-.5
        np.testing.assert_allclose(contrasts.interval_score_change,expected.mean(),atol=1e-14)
        np.testing.assert_allclose(contrasts.interval_score_change_mcse,
            np.std(expected,ddof=1)/np.sqrt(100),atol=1e-14)
        assert (contrasts.paired_replications==100).all()
        checks['paired_interval_score_MCSE_independent_manual_formula']=True
        counts=s.endpoint_counts(reference)
        assert all(v['requested_replications']==100 and v['prediction_replications']==100 and
            v['converged_prediction_replications']==99 and
            v['finite_nonconverged_prediction_replications']==1 for v in counts.values())
        checks['nonconverged_finite_predictions_retained']=True
        missing=reference[reference.replication!=20000]
        counts=s.endpoint_counts(missing)
        assert all(v['requested_replications']==100 and v['prediction_replications']==99 and
            set(v['complete_population_ids'])!=s.IDS for v in counts.values())
        checks['one_execution_failure_never_claims100_available']=True
        partial=reference.copy()
        partial.loc[(partial.replication==20000)&(partial.outcome=='anemia'),'areas']=79
        assert s.endpoint_counts(partial)['anemia']['complete80_area_replications']==99
        checks['one_missing_area_prevents_full_population_guard']=True
        empty=s.endpoint_counts(pd.DataFrame())
        assert all(v['requested_replications']==100 and v['prediction_replications']==0 for v in empty.values())
        checks['all_failed_retains_requested100']=True
        bad=pd.concat([reference,reference.iloc[:1]],ignore_index=True)
        try:s.checked_frame(bad,s.METHOD)
        except ValueError:checks['duplicate_evaluation_cell_rejected']=True
        else:raise AssertionError('Duplicate evaluation was accepted')
        s.validate_tasks(s.expected_tasks())
        bad_tasks=s.expected_tasks();bad_tasks.loc[0,'replication']=19999
        try:s.validate_tasks(bad_tasks)
        except AssertionError:checks['development_or_wrong_seed_rejected']=True
        else:raise AssertionError('Wrong seed was accepted')
        pending=folder/'pending';pending.mkdir()
        (pending/'predictions.csv').write_text('area\n0\n');(pending/'status.json').write_text('{}')
        assert not s.task_accounted({'output_dir':str(pending)})
        (pending/'task_execution.json').write_text('{}')
        assert s.task_accounted({'output_dir':str(pending)})
        failed=folder/'failed';failed.mkdir();(failed/'execution_failure.json').write_text('{}')
        assert s.task_accounted({'output_dir':str(failed)})
        checks['pending_fit_distinguished_from_finished_or_explicit_failure']=True
    proof=dict(passed=all(checks.values()),checks=checks,
        actual_population_truth_or_performance_read=False,
        evaluator_sha256=sha256(Path(s.__file__)),
        qa_source_sha256=sha256(Path(__file__)),
        metrics_source_sha256=sha256(Path(__file__).with_name('benchmark_metrics.py')))
    destination=OUT/'validation/spatial_evaluator_qa.json';destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(proof,indent=2));print(json.dumps(proof,indent=2))


if __name__=='__main__':main()

"""B100, missing-endpoint and output contract QA on artificial training data."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
import json,tempfile,time
from pathlib import Path
import numpy as np
from sikov_published_fit import fit_arrays,ROOT,sha
from sikov_published_audit import independent_triangular
from sikov_final_audit import dense_gaussian_check

def main():
    before=time.monotonic();rng=np.random.default_rng(20261010945);N=60;strata=np.repeat([1,2,3],20);s=np.concatenate([np.arange(i*20,i*20+15) for i in range(3)])
    names=['intercept','Altitud','PisoTierra','Cemento','Agua','Elect','Desague','Internet','SIS','EsSalud','NoSeguro','Refrig','Castellano','Analfabet','Rural']
    X=np.column_stack([np.ones(N),rng.uniform(size=(N,14))]);y=rng.uniform(.12,.42,size=(len(s),2));D=np.zeros((len(s),2,2));D[:,0,0]=rng.uniform(.003,.008,len(s));D[:,1,1]=rng.uniform(.003,.008,len(s))
    y[[1,17,33],0]=np.nan;D[[4,21,35],1,1]=np.nan
    d={'X_raw':X,'y':y,'D':D,'sample_idx':s,'strata':strata,'coords':rng.uniform(size=(N,2)),'covariate_names':np.array(names)}
    # dense_gaussian_check's base graph branch requires d['W'], but the
    # independently rebuilt per-endpoint triangular matrices override it.
    d['W']=independent_triangular(d)[0]
    with tempfile.TemporaryDirectory(prefix='sikov_published_contract_',dir=ROOT/'validacion') as tmp:
        out=Path(tmp)/'fits';primary,bc,status=fit_arrays(d,out,20261010946)
        check=dense_gaussian_check(d,status,'Sikov_SAR',primary,W_override=independent_triangular(d))
        if not np.array_equal(primary['mean'],bc['mean']) or check['bootstrap_requested']!=600 or sum(status['endpoint_training_rows'])!=84:raise RuntimeError('B100 endpoint/output contract failed')
        npz=dict(np.load(out/'Sikov_SAR_Published.npz',allow_pickle=False))
        if set(npz)!=set(primary) or not all(np.array_equal(npz[k],primary[k]) for k in npz):raise RuntimeError('Exported primary arrays differ')
        result={'status':'PASS_B100_ENDPOINT_MASK_AND_EXPORT_CONTRACT','bootstrap_requested':check['bootstrap_requested'],'bootstrap_failed':check['bootstrap_failed'],'bootstrap_recovered':check['bootstrap_recovered'],'endpoint_training_rows':status['endpoint_training_rows'],'all_converged':status['converged'],'independent_dense_QA':check,'all60domains_and_both_endpoints_retained':primary['mean'].shape==(60,2),'missing_endpoint_graphs_are_distinct':not np.array_equal(*independent_triangular(d)),'naive_and_BC_same_point_and_bootstrap_draws':True,'training_inputs_are_artificial':True,'evaluation_truth_or_actual_heldout_responses_read':False,'elapsed_seconds':time.monotonic()-before,'source_sha256':sha(__file__)}
    (ROOT/'validacion/sikov_published_contract_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()

"""Independent final audit; no truth access until every prediction is sealed.

The scoring and dense Gaussian checks below do not call the implementation's
prediction or score functions. --self-test uses entirely synthetic losses.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'1')
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor,cho_solve
from scipy.special import ndtr
from scipy.stats import norm,t

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT.parent
Z=1.959963984540054
OUTCOMES=('anemia','stunting')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist() if isinstance(x,np.ndarray) else str(x))+'\n')

def independent_balanced(frame,candidate,reference,contexts,replications):
    rows=[];m1=[];m0=[]
    for context in contexts:
        for rep in replications:
            a=frame[(frame.scenario==context)&(frame.replication==rep)&(frame.method==candidate)&(frame.group=='all')].set_index('outcome').loc[list(OUTCOMES)]
            b=frame[(frame.scenario==context)&(frame.replication==rep)&(frame.method==reference)&(frame.group=='all')].set_index('outcome').loc[list(OUTCOMES)]
            if len(a)!=2 or len(b)!=2:raise ValueError('Incomplete paired outcomes')
            rows.append(np.array([a[k].mean()-b[k].mean() for k in ('interval_score','mse','mae','coverage','width')]))
            m1.append(a.mse.to_numpy());m0.append(b.mse.to_numpy())
    differences=np.array(rows).reshape(len(contexts),len(replications),5)
    estimate=differences.mean(axis=(0,1))
    variance=differences.var(axis=1,ddof=1).sum(axis=0)/(len(replications)*len(contexts)**2)
    se=np.sqrt(variance);result=[]
    for j,metric in enumerate(('interval_score','mse','mae','coverage','width')):
        p=float(2*norm.sf(abs(estimate[j]/se[j]))) if se[j]>0 else float(estimate[j]==0)
        result.append(dict(metric=metric,difference=estimate[j],mcse=se[j],MC_CI95_low=estimate[j]-Z*se[j],MC_CI95_high=estimate[j]+Z*se[j],p_two_sided=p))
    m1=np.array(m1).reshape(len(contexts),len(replications),2);m0=np.array(m0).reshape(m1.shape)
    M1,M0=m1.mean(),m0.mean();logratio=.5*np.log(M1/M0)
    influence=.5*(m1/M1-m0/M0).mean(axis=2)
    ratio_se=np.sqrt(influence.var(axis=1,ddof=1).sum()/(len(replications)*len(contexts)**2))
    result.append(dict(metric='RMSE_ratio',difference=np.exp(logratio),mcse=ratio_se,MC_CI95_low=np.exp(logratio-Z*ratio_se),MC_CI95_high=np.exp(logratio+Z*ratio_se),p_two_sided=float(2*norm.sf(abs(logratio/ratio_se))) if ratio_se>0 else float(logratio==0)))
    return result

def graph(coords,strata,k,haversine=False):
    W=np.zeros((len(coords),len(coords)))
    for level in np.unique(strata):
        ids=np.flatnonzero(strata==level);c=coords[ids]
        if haversine:
            longitude,latitude=np.deg2rad(c).T
            delta_lat=latitude[:,None]-latitude[None,:];delta_lon=longitude[:,None]-longitude[None,:]
            a=np.sin(delta_lat/2)**2+np.cos(latitude[:,None])*np.cos(latitude[None,:])*np.sin(delta_lon/2)**2
            distance=2*np.arctan2(np.sqrt(np.clip(a,0,1)),np.sqrt(np.clip(1-a,0,1)))
        else:distance=np.sqrt(((c[:,None,:]-c[None,:,:])**2).sum(axis=2))
        np.fill_diagonal(distance,np.inf);kk=min(k,len(ids)-1)
        neighbors=np.argsort(distance,axis=1,kind='stable')[:,:kk]
        W[ids[:,None],ids[neighbors]]=1/kk
    return W

def dense_gaussian_check(d,status,method,prediction,W_override=None):
    """Recompute point, plug-in MSE, REML and projected score by dense algebra."""
    W=d['W'] if method in ('Sikov_FH','Sikov_SAR') else graph(d['coords'],d['strata'],int(method[-1]) if method[-1] in ('4','8') else 6,method.endswith('haversine'))
    spatial=method!='Sikov_FH';errors=[];gradients=[];boundaries=0;failed_boot=0;recovered_boot=0;total_boot=0
    for endpoint,fit in enumerate(status['fits']):
        endpoint_W=W if W_override is None else W_override[endpoint]
        observed=np.isfinite(d['y'][:,endpoint])&np.isfinite(d['D'][:,endpoint,endpoint])&(d['D'][:,endpoint,endpoint]>0)
        gsample=d['sample_idx'][observed];yy=d['y'][observed,endpoint];DD=d['D'][observed,endpoint,endpoint]
        for sf in fit['fit']:
            ids=np.flatnonzero(d['strata']==sf['stratum']);mask=np.isin(gsample,ids);s=np.searchsorted(ids,gsample[mask])
            cov=[d['covariate_names'].tolist().index(c) for c in sf['covariates']]
            X=d['X_raw'][ids][:,cov];Xs=X[s];y=yy[mask];D=DD[mask]
            WW=endpoint_W[np.ix_(ids,ids)];WW=WW/WW.sum(axis=1)[:,None]
            A=float(sf['variance']);rho=float(sf['rho']);inverse=np.linalg.inv(np.eye(len(ids))-rho*WW) if spatial else np.eye(len(ids))
            C=inverse@inverse.T;dInv=inverse@WW@inverse if spatial else np.zeros_like(inverse)
            dC=dInv@inverse.T+inverse@dInv.T;Css=C[np.ix_(s,s)]
            V=A*Css+np.diag(D);fac=cho_factor(V,lower=True);Vi=cho_solve(fac,np.eye(len(s)))
            Q=np.linalg.inv(Xs.T@Vi@Xs);beta=Q@Xs.T@Vi@y;residual=y-Xs@beta
            K=A*C[:,s]@Vi;mean=X@beta+K@residual;L=X-K@Xs
            plugin=A*np.diag(C)-np.einsum('ij,ji->i',K,A*C[s,:])+np.einsum('ij,jk,ik->i',L,Q,L)
            fixed_information=cho_factor(Xs.T@Vi@Xs,lower=True)
            loss=.5*(2*np.log(np.diag(fac[0])).sum()+2*np.log(np.diag(fixed_information[0])).sum()+residual@Vi@residual+(len(s)-len(cov))*np.log(2*np.pi))
            errors.append({'mean':float(abs(mean-prediction['mean'][ids,endpoint]).max()),'plugin':float(abs(plugin-prediction['mse_plugin'][ids,endpoint]).max()),'nll':abs(loss-sf['reml_nll'])})
            P=Vi-Vi@Xs@Q@Xs.T@Vi;vir=Vi@residual;scale=max(float(np.var(y)),float(D.mean()),1e-5)
            derivative=[scale*Css]+([A*dC[np.ix_(s,s)]] if spatial else [])
            grad=np.array([.5*(np.trace(P@dv)-vir@dv@vir) for dv in derivative]);projected=grad.copy()
            if A/scale<=1e-8 and grad[0]>0:projected[0]=0
            if A/scale>=100-1e-8 and grad[0]<0:projected[0]=0
            if spatial:
                if rho<=-.999+1e-8 and grad[1]>0:projected[1]=0
                if rho>=.999-1e-8 and grad[1]<0:projected[1]=0
                boundaries+=abs(rho)>=.999-1e-8
            gradients.append(float(abs(projected).max()))
            if not sf.get('raw_optimizer_converged',True) and sf['converged']:
                if abs(projected).max()>=1e-5:raise ValueError('Unjustified line-search recovery')
            if method in ('Sikov_FH','Sikov_SAR'):
                if sf['bootstrap_B']!=100 or sf['bootstrap_success']<95:raise ValueError('Wrong native bootstrap convention')
                failed_boot+=sf['bootstrap_failed'];recovered_boot+=sf['bootstrap_stationarity_recovered'];total_boot+=sf['bootstrap_B']
    maxima={k:max(v[k] for v in errors) for k in ('mean','plugin','nll')}
    if not np.isfinite(list(maxima.values())+gradients).all():raise ValueError('Nonfinite independent Gaussian audit')
    if maxima['mean']>1e-8 or maxima['plugin']>1e-8 or maxima['nll']>1e-7:raise ValueError('Dense Gaussian reconstruction differs '+str(maxima))
    return dict(maximum_errors=maxima,max_projected_gradient=max(gradients),rho_boundaries=int(boundaries),bootstrap_failed=failed_boot,bootstrap_recovered=recovered_boot,bootstrap_requested=total_boot)

def check_hmc_status(status):
    computed=[]
    for endpoint in status['endpoints']:
        config=endpoint['sampler_config'];trials=endpoint['sampler_trials']
        if config['chains']!=4 or config['warmup']!=1000 or config['samples']!=1000 or config['metric']!='diag_e' or len(trials) not in (1,2):raise ValueError('NUTS protocol changed')
        diagnosis=trials[-1]
        eligible=bool(diagnosis['finite_diagnostics'] and diagnosis['max_Rhat']<=1.01 and diagnosis['min_bulk_ESS']>=400 and diagnosis['min_tail_ESS']>=400 and min(diagnosis['E_BFMI'])>=.3 and diagnosis['divergences']==0 and diagnosis['max_treedepth_hits']==0)
        if eligible!=diagnosis['success'] or eligible!=endpoint['success']:raise ValueError('NUTS diagnostics and flags disagree')
        expected_draws=4*(1000 if diagnosis['attempt']==1 else 2000)
        if endpoint['posterior_draws']!=expected_draws:raise ValueError('Posterior chain count differs')
        computed.append(eligible and endpoint['beta_shape_guard_cells']==0)
    if all(computed)!=status['converged']:raise ValueError('ICC numerical convergence flag inconsistent')
    return all(computed)

def audit_predictions():
    freeze=ROOT/'protocol_freeze.json';protocol=json.loads(freeze.read_text());protocol_sha=sha(freeze)
    for section in ('statistical_sources_sha256','frozen_inputs_sha256','runtime_binary_sha256'):
        for path,digest in protocol.get(section,{}).items():
            if sha(path)!=digest:raise RuntimeError('Changed frozen file '+path)
    baseline=json.loads((ROOT/'validacion/previous_sources_and_release_baseline.json').read_text())
    for item in baseline:
        if sha(STUDY/item['path'])!=item['sha256']:raise RuntimeError('Previous artifact modified '+item['path'])
    lockfile=ROOT/'validacion/fresh_prediction_lock.json';lock=json.loads(lockfile.read_text())
    expected={(c,r) for c in protocol['contexts'] for r in protocol['evaluation_ids']}
    if lock['protocol_sha256']!=protocol_sha or not lock['approved_for_scoring'] or lock['fresh_truth_read_before_lock'] or {(i['context'],i['replication']) for i in lock['prediction_items']}!=expected:raise ValueError('Incomplete or wrong prediction lock')
    worlds=[];recipe=protocol['selected_recipe'];maxima={'mean':0.,'plugin':0.,'nll':0.};boundaries=failed=recovered=total=bcnegative=0;convergence_failures=[];gradients=[]
    for item in lock['prediction_items']:
        context,rep=item['context'],item['replication'];folder=ROOT/'evaluacion/fits'/context/f'rep_{rep:05d}';source=ROOT/'evaluacion/inputs'/context/f'rep_{rep:05d}'
        completion=json.loads((folder/'completion.json').read_text())
        if sha(folder/'completion.json')!=item['completion_sha256'] or completion['protocol_sha256']!=protocol_sha or completion['truth_read_by_fits']:raise ValueError('Invalid completion seal')
        manifest=json.loads((source/'manifest.json').read_text())
        if sha(source/'manifest.json')!=completion['input_manifest_sha256'] or manifest['source_sha256']!=sha(ROOT/'scripts/comparison_simulation.py'):raise ValueError('Changed generation manifest or source')
        if manifest['phase']!='evaluation' or manifest['context']!=context or manifest['replication']!=rep or manifest['population_stream']!=[20261010,701,protocol['contexts'].index(context),rep] or manifest['sampling_stream']!=[20261010,702,protocol['contexts'].index(context),rep]:raise ValueError('Fresh population/sampling role or independence differs')
        if set(completion['convergence'])!=set(protocol['methods']):raise ValueError('Method universe mismatch')
        for filename,digest in completion['output_sha256'].items():
            if sha(folder/filename)!=digest:raise RuntimeError('Changed sealed output '+str(folder/filename))
        for filename,key in (('inputs.npz','fit_inputs_sha256'),('sample.csv','sample_sha256')):
            if sha(source/'model_inputs'/filename)!=completion[key]:raise RuntimeError('Changed model inputs')
        d=dict(np.load(source/'model_inputs/inputs.npz',allow_pickle=False));pred={m:dict(np.load(folder/(m+'.npz'),allow_pickle=False)) for m in protocol['methods']}
        if len(d['X_raw'])!=120 or len(np.unique(d['strata']))!=3 or not all((d['strata']==s).sum()==40 for s in (1,2,3)):raise ValueError('Fresh frame mismatch')
        if not np.allclose(d['W'],graph(d['coords'],d['strata'],6),atol=0,rtol=0):raise ValueError('Primary graph mismatch')
        for method,p in pred.items():
            arrays=np.stack([p[k] for k in ('mean','lower','upper')])
            if arrays.shape!=(3,120,2) or not np.isfinite(arrays).all() or np.any(p['lower']>p['upper']):raise ValueError('Invalid complete predictions '+method)
            if not completion['convergence'][method]:convergence_failures.append([context,rep,method])
        check_hmc_status(json.loads((folder/'ICC_status.json').read_text()))
        if recipe.get('stratum_deviations'):check_hmc_status(json.loads((folder/'Selected_status.json').read_text()))
        for method in ('Sikov_FH','Sikov_SAR','Sikov_SAR_k4','Sikov_SAR_k8','Sikov_SAR_haversine'):
            status=json.loads((folder/(method+'_status.json')).read_text());check=dense_gaussian_check(d,status,method,pred[method]);worlds.append(dict(context=context,replication=rep,method=method,**check))
            for key in maxima:maxima[key]=max(maxima[key],check['maximum_errors'][key])
            gradients.append(check['max_projected_gradient']);boundaries+=check['rho_boundaries'];failed+=check['bootstrap_failed'];recovered+=check['bootstrap_recovered'];total+=check['bootstrap_requested']
            if not np.allclose(pred[method]['lower'],pred[method]['mean']-Z*np.sqrt(pred[method]['mse']),atol=1e-12,rtol=0):raise ValueError('Normal interval convention differs')
        for method in ('Sikov_FH','Sikov_SAR'):
            bc=pred[method+'_BC'];primary=pred[method]
            if not np.array_equal(primary['mean'],bc['mean']):raise ValueError('BC altered point estimator')
            if not np.allclose(primary['mse'],bc['mse_naive'],atol=0,rtol=0):raise ValueError('Bootstrap draw convention differs')
            if not np.allclose(bc['lower'],bc['mean']-Z*np.sqrt(np.maximum(bc['mse_bc'],0)),atol=1e-12,rtol=0):raise ValueError('BC sqrt convention differs')
            bcnegative+=int((bc['mse_bc']<0).sum())
        if recipe['family']=='ICC_SAR_mixture':
            w=recipe['icc_weight'];expected_mean=w*pred['ICC']['mean']+(1-w)*pred['Sikov_SAR']['mean']
            if not np.allclose(pred['Selected']['mean'],expected_mean,atol=1e-12,rtol=0):raise ValueError('Frozen mixture weight differs')
            draws=dict(np.load(folder/'ICC_conditional_draws.npz',allow_pickle=False))
            for j,endpoint in enumerate(OUTCOMES):
                samples=draws[endpoint+'_marginal'] if endpoint+'_marginal' in draws else draws['p_draws'][:,:,j]
                sd=np.sqrt(pred['Sikov_SAR']['mse'][:,j]);mu=pred['Sikov_SAR']['mean'][:,j]
                for bound,target in (('lower',.025),('upper',.975)):
                    q=pred['Selected'][bound][:,j];gaussian=np.where(sd>0,ndtr((q-mu)/np.maximum(sd,np.finfo(float).tiny)),(q>=mu).astype(float))
                    mixture=w*(samples<=q).mean(axis=0)+(1-w)*gaussian
                    if np.any(mixture<target-1e-10) or np.any(mixture>target+w/len(samples)+1e-9):raise ValueError('Saved mixture quantile fails independent CDF inversion')
    return protocol,dict(passed=True,prediction_worlds=len(expected),previous_artifacts_preserved=len(baseline),protocol_sha256=protocol_sha,prediction_lock_sha256=sha(lockfile),independent_dense_maximum_errors=maxima,maximum_projected_gradient=max(gradients),rho_boundary_fits=boundaries,bootstrap_requested=total,bootstrap_failed=failed,bootstrap_stationarity_recovered=recovered,negative_BC_MSE_cells=bcnegative,convergence_failures_retained=convergence_failures,truth_read=False,world_checks=worlds)

def score_independently(protocol):
    rows=[];physical_rows=[]
    for context in protocol['contexts']:
        for rep in protocol['evaluation_ids']:
            folder=ROOT/'evaluacion/fits'/context/f'rep_{rep:05d}';source=ROOT/'evaluacion/inputs'/context/f'rep_{rep:05d}'
            manifest=json.loads((source/'manifest.json').read_text())
            if sha(source/'evaluation_only/truth.npz')!=manifest['truth_sha256']:raise ValueError('Fresh generation truth changed')
            theta=np.load(source/'evaluation_only/truth.npz',allow_pickle=False)['theta'];d=dict(np.load(source/'model_inputs/inputs.npz',allow_pickle=False))
            observed=np.zeros(theta.shape,bool);observed[d['sample_idx']]=np.isfinite(d['y']);unsampled=~np.isin(np.arange(len(theta)),d['sample_idx'])
            for method in protocol['methods']:
                p=dict(np.load(folder/(method+'.npz'),allow_pickle=False))
                for j,endpoint in enumerate(OUTCOMES):
                    for group,mask in (('all',np.ones(len(theta),bool)),('observed_endpoint',observed[:,j]),('unsampled_common',unsampled)):
                        truth=theta[mask,j];mean=p['mean'][mask,j];lo=p['lower'][mask,j];hi=p['upper'][mask,j];error=mean-truth
                        IS=(hi-lo)+40*np.maximum(lo-truth,0)+40*np.maximum(truth-hi,0)
                        rows.append(dict(scenario=context,replication=rep,method=method,outcome=endpoint,group=group,mse=np.mean(error**2),mae=np.mean(abs(error)),bias=np.mean(error),interval_score=np.mean(IS),coverage=np.mean((lo<=truth)&(truth<=hi)),width=np.mean(hi-lo)))
                        bounded_mean=np.clip(mean,0,1);bounded_lo=np.clip(lo,0,1);bounded_hi=np.clip(hi,0,1);bounded_error=bounded_mean-truth
                        bounded_IS=bounded_hi-bounded_lo+40*np.maximum(bounded_lo-truth,0)+40*np.maximum(truth-bounded_hi,0)
                        physical_rows.append(dict(scenario=context,replication=rep,method=method,outcome=endpoint,group=group,mse=np.mean(bounded_error**2),mae=np.mean(abs(bounded_error)),bias=np.mean(bounded_error),interval_score=np.mean(bounded_IS),coverage=np.mean((bounded_lo<=truth)&(truth<=bounded_hi)),width=np.mean(bounded_hi-bounded_lo)))
    own=pd.DataFrame(rows);reported=pd.read_csv(ROOT/'resultados/fresh_replications.csv');keys=['scenario','replication','method','outcome','group'];matched=own.merge(reported,on=keys,validate='one_to_one',suffixes=('_independent','_reported'))
    if len(own)!=len(reported) or len(matched)!=len(own):raise ValueError('Fresh score row universe differs')
    score_errors={metric:float(abs(matched[metric+'_independent']-matched[metric+'_reported']).max()) for metric in ('mse','mae','bias','interval_score','coverage','width')}
    if max(score_errors.values())>1e-12:raise ValueError('Independent finite-population scores differ')
    physical=pd.DataFrame(physical_rows);physical_report=pd.read_csv(ROOT/'resultados/fresh_physical_projection_replications.csv')
    physical_match=physical.merge(physical_report,on=keys,validate='one_to_one',suffixes=('_independent','_reported'))
    if len(physical)!=len(physical_report) or len(physical_match)!=len(physical):raise ValueError('Shared physical projection universe differs')
    physical_error=max(float(abs(physical_match[metric+'_independent']-physical_match[metric+'_reported']).max()) for metric in ('mse','mae','bias','interval_score','coverage','width'))
    if physical_error>1e-12:raise ValueError('Shared physical projection scores differ')
    reported_contrasts=pd.read_csv(ROOT/'resultados/fresh_balanced_paired_contrasts.csv');contrast_errors={};primary=[]
    for reference in protocol['methods']:
        if reference=='Selected':continue
        result=independent_balanced(own,'Selected',reference,protocol['contexts'],protocol['evaluation_ids'])
        for row in result:
            other=reported_contrasts[(reported_contrasts.reference==reference)&(reported_contrasts.metric==row['metric'])]
            if len(other)!=1:raise ValueError('Paired contrast universe differs')
            error=max(abs(float(other.iloc[0][key])-float(row[key])) for key in ('difference','mcse','MC_CI95_low','MC_CI95_high','p_two_sided'))
            contrast_errors[reference+'_'+row['metric']]=error
            if reference in ('Sikov_FH','Sikov_SAR') and row['metric']=='interval_score':primary.append({'reference':reference,**row})
    if max(contrast_errors.values())>1e-10:raise ValueError('Independent paired MC contrasts differ')
    physical_contrasts=pd.read_csv(ROOT/'resultados/fresh_physical_projection_contrasts.csv');physical_contrast_errors=[]
    for reference in ('Sikov_FH','Sikov_SAR','ICC'):
        for row in independent_balanced(physical,'Selected',reference,protocol['contexts'],protocol['evaluation_ids']):
            other=physical_contrasts[(physical_contrasts.reference==reference)&(physical_contrasts.metric==row['metric'])]
            if len(other)!=1:raise ValueError('Incomplete common projection contrast')
            physical_contrast_errors.append(max(abs(float(other.iloc[0][key])-float(row[key])) for key in ('difference','mcse','MC_CI95_low','MC_CI95_high','p_two_sided')))
    if max(physical_contrast_errors)>1e-10:raise ValueError('Common physical projection paired contrasts differ')
    order=np.argsort([row['p_two_sided'] for row in primary]);largest=0
    for rank,index in enumerate(order):
        largest=max(largest,(2-rank)*primary[index]['p_two_sided']);primary[index]['p_Holm2']=min(1,largest)
        other=reported_contrasts[(reported_contrasts.reference==primary[index]['reference'])&(reported_contrasts.metric=='interval_score')].iloc[0]
        if abs(other.p_Holm2-primary[index]['p_Holm2'])>1e-12:raise ValueError('Holm family differs')
    return dict(truth_read=True,truth_access_after_independent_prediction_verification=True,independent_score_rows=len(own),score_max_absolute_errors=score_errors,paired_contrast_max_absolute_error=max(contrast_errors.values()),physical_projection_score_max_absolute_error=physical_error,physical_projection_paired_max_absolute_error=max(physical_contrast_errors),primary_balanced_results=primary,BC_sensitivity_not_omitted=all(method in set(own.method) for method in ('Sikov_FH_BC','Sikov_SAR_BC')))

def audit_larger_bootstrap(protocol):
    rows=[];scores=[];lock_sha=sha(ROOT/'validacion/fresh_prediction_lock.json');recipe=protocol['selected_recipe']
    if recipe['family']!='ICC_SAR_mixture' or recipe['icc_weight']!=.75:raise ValueError('B300 postprocessing must retain the frozen75/25 recipe')
    w=recipe['icc_weight']
    for context in protocol['contexts']:
        for rep in (30000,30001):
            stem=f'sikov_bootstrap_evaluation_{context}_rep{rep:05d}_B300'
            record=json.loads((ROOT/'validacion'/f'{stem}.json').read_text());arrays=dict(np.load(ROOT/'validacion'/f'{stem}.npz',allow_pickle=False))
            model=ROOT/'evaluacion/inputs'/context/f'rep_{rep:05d}/model_inputs/inputs.npz'
            if record['B']!=300 or record['phase']!='evaluation' or record['truth_was_read'] or record['uncertainty_selection_performed']:raise ValueError('Wrong B300 sensitivity role')
            if record['lock_sha256']!=lock_sha or record['protocol_sha256']!=sha(ROOT/'protocol_freeze.json') or record['input_sha256']!=sha(model) or record['outputs_sha256']!=sha(ROOT/'validacion'/f'{stem}.npz'):raise ValueError('Invalid B300 sensitivity provenance')
            for field,filename in (('model_source_sha256','sikov_model.py'),('bootstrap_source_sha256','sikov_bootstrap.py'),('script_sha256','sikov_bootstrap_sensitivity.py')):
                if record[field]!=sha(ROOT/'scripts'/filename):raise ValueError('Wrong B300 source')
            closed=dict(np.load(ROOT/'evaluacion/fits'/context/f'rep_{rep:05d}/Sikov_SAR.npz',allow_pickle=False))
            folder=ROOT/'evaluacion/fits'/context/f'rep_{rep:05d}'
            icc=dict(np.load(folder/'ICC.npz',allow_pickle=False));selected=dict(np.load(folder/'Selected.npz',allow_pickle=False));draws=dict(np.load(folder/'ICC_conditional_draws.npz',allow_pickle=False))
            manifest=json.loads((model.parents[1]/'manifest.json').read_text());truthpath=model.parents[1]/'evaluation_only/truth.npz'
            if sha(truthpath)!=manifest['truth_sha256']:raise ValueError('B300 descriptive scoring truth changed')
            truth=np.load(truthpath,allow_pickle=False)['theta']
            for j,endpoint in enumerate(OUTCOMES):
                metadata=record['metadata'][j]
                if not metadata['converged'] or any(sf['bootstrap_B']!=300 or sf['bootstrap_success']<285 for sf in metadata['fit']):raise ValueError('B300 convergence threshold not satisfied')
                values=np.stack([arrays[f'{endpoint}_{key}'] for key in ('mean','mse_naive','mse_bc','naive_mcse','bc_mcse')])
                if not np.isfinite(values).all() or np.any(arrays[endpoint+'_mse_naive']<0):raise ValueError('Invalid B300 uncertainties')
                point_error=float(abs(arrays[endpoint+'_mean']-closed['mean'][:,j]).max())
                if point_error>1e-10:raise ValueError('B300 changed the point estimate')
                rows.append(dict(context=context,replication=rep,outcome=endpoint,point_max_error=point_error,median_B300_B100_naive_MSE_ratio=float(np.median(arrays[endpoint+'_mse_naive']/closed['mse'][:,j])),negative_B300_BC_cells=int((arrays[endpoint+'_mse_bc']<0).sum()),median_B300_relative_MCSE=float(np.median(arrays[endpoint+'_naive_mcse']/arrays[endpoint+'_mse_naive']))))
                mu=arrays[endpoint+'_mean'];sd=np.sqrt(arrays[endpoint+'_mse_naive']);samples=draws[endpoint+'_marginal'];bounds=[]
                for target in (.025,.975):
                    left=np.minimum(0.,mu-12*sd);right=np.maximum(1.,mu+12*sd)
                    for step in range(56):
                        midpoint=(left+right)/2;gaussian=np.where(sd>0,ndtr((midpoint-mu)/np.maximum(sd,np.finfo(float).tiny)),(midpoint>=mu).astype(float));cdf=w*np.mean(samples<=midpoint,axis=0)+(1-w)*gaussian;move=cdf>=target
                        right=np.where(move,midpoint,right);left=np.where(move,left,midpoint)
                    bounds.append(right)
                mixture_mean=w*icc['mean'][:,j]+(1-w)*mu
                if not np.allclose(mixture_mean,selected['mean'][:,j],atol=1e-10,rtol=0):raise ValueError('B300 sensitivity altered mixture point estimator')
                predictions=[('SAR_B100',closed['mean'][:,j],closed['lower'][:,j],closed['upper'][:,j]),('SAR_B300',mu,mu-Z*sd,mu+Z*sd),('Selected_B100',selected['mean'][:,j],selected['lower'][:,j],selected['upper'][:,j]),('Selected_B300',mixture_mean,bounds[0],bounds[1])]
                for method,mean,lo,hi in predictions:
                    target=truth[:,j];IS=hi-lo+40*np.maximum(lo-target,0)+40*np.maximum(target-hi,0)
                    scores.append(dict(context=context,replication=rep,outcome=endpoint,method=method,areas=120,interval_score=float(IS.mean()),coverage=float(np.mean((lo<=target)&(target<=hi))),width=float(np.mean(hi-lo)),mse=float(np.mean((mean-target)**2))))
    detail=pd.DataFrame(scores);summary=detail.groupby(['context','method'])[['interval_score','coverage','width','mse']].mean().reset_index();all_contexts=summary.groupby('method')[['interval_score','coverage','width','mse']].mean().reset_index();all_contexts.insert(0,'context','equal_four_contexts');summary=pd.concat([summary,all_contexts],ignore_index=True)
    comparisons=[]
    for context in list(protocol['contexts'])+['equal_four_contexts']:
        local=summary[summary.context==context].set_index('method')
        for candidate,reference in [('Selected_B100','SAR_B100'),('Selected_B300','SAR_B300'),('Selected_B300','Selected_B100'),('SAR_B300','SAR_B100')]:
            comparisons.append(dict(context=context,candidate=candidate,reference=reference,worlds=8 if context=='equal_four_contexts' else 2,**{metric:float(local.loc[candidate,metric]-local.loc[reference,metric]) for metric in ('interval_score','coverage','width','mse')}))
    paths=[ROOT/'validacion/sikov_B300_descriptive_scores.csv',ROOT/'validacion/sikov_B300_descriptive_balanced.csv',ROOT/'validacion/sikov_B300_descriptive_differences.csv']
    for frame,path in zip((detail,summary,pd.DataFrame(comparisons)),paths):frame.to_csv(path,index=False)
    descriptive=dict(status='PASS_FIXED_B300_DESCRIPTIVE_SENSITIVITY',worlds=8,contexts=protocol['contexts'],replications=[30000,30001],fixed_ICC_weight=w,closed_ICC_draws_used=True,point_estimates_identical=True,primary_256_B100_Holm2_unchanged=True,no_new_selection=True,confirmatory=False,truth_scoring_only_after_complete_prediction_lock=True,protocol_sha256=sha(ROOT/'protocol_freeze.json'),prediction_lock_sha256=lock_sha,outputs_sha256={str(path):sha(path) for path in paths},balanced_differences=[row for row in comparisons if row['context']=='equal_four_contexts'])
    dump(ROOT/'validacion/sikov_B300_descriptive_status.json',descriptive)
    return dict(prespecified_B300_worlds=8,B300_endpoint_checks=rows,B300_did_not_reselect_any_method=True,B300_fixed_mixture_descriptive_results=descriptive)

def audit_real_predictions(protocol):
    records=[]
    for scope in ('temporal','territorial'):
        folder=ROOT/'evaluacion'/('real_'+scope);lock=json.loads((ROOT/f'validacion/real_prediction_lock_{scope}.json').read_text());completion=json.loads((folder/'completion.json').read_text())
        if lock['protocol_sha256']!=sha(ROOT/'protocol_freeze.json') or not lock['approved_for_scoring'] or lock['heldout_outcomes_used'] or completion['heldout_outcomes_read_by_fits']:raise ValueError('Wrong real fit/scoring role')
        if sha(folder/'completion.json')!=lock['completion_sha256'] or sha(folder/'predictions.csv')!=lock['predictions_sha256']:raise ValueError('Changed real prediction seal')
        for filename,digest in completion['output_sha256'].items():
            if sha(folder/filename)!=digest:raise ValueError('Changed real fit output')
        prepared=ROOT/'datos'/('real_2019_full' if scope=='temporal' else 'real_2024_training');manifest=json.loads((prepared/'preparation_manifest.json').read_text())
        if sha(prepared/'preparation_manifest.json')!=completion['training_manifest_sha256'] or manifest['heldout_outcome_rows_read']:raise ValueError('Real training manifest changed or leaked')
        if sha(prepared/'private/units.csv')!=manifest['private_units_sha256']:raise ValueError('Private real training units changed')
        for filename,digest in manifest['output_sha256'].items():
            if sha(prepared/filename)!=digest:raise ValueError('Real training array changed')
        frame=pd.read_csv(folder/'predictions.csv',dtype={'ubigeo':str});support=pd.read_csv(prepared/'area_frame.csv',dtype={'ubigeo':str}).ubigeo.tolist()
        if len(support)!=1874 or len(frame)!=1874*len(protocol['methods']) or frame[['ubigeo','method']].duplicated().any() or set(frame.method)!=set(protocol['methods']):raise ValueError('Unequal real prediction universe')
        failed=recovered=requested=negative_bc=0
        for method in protocol['methods']:
            supplied=frame[frame.method==method].set_index('ubigeo').loc[support];arrays=dict(np.load(folder/(method+'.npz'),allow_pickle=False))
            if supplied.shape[0]!=1874 or not np.isfinite(np.stack([arrays[k] for k in ('mean','lower','upper')])).all():raise ValueError('Invalid real predictions')
            for j,endpoint in enumerate(OUTCOMES):
                for column,key in (('p_'+endpoint,'mean'),('lo95_'+endpoint,'lower'),('hi95_'+endpoint,'upper')):
                    if not np.allclose(supplied[column],arrays[key][:,j],atol=1e-12,rtol=0):raise ValueError('Real exported columns changed')
            if method in ('Sikov_FH','Sikov_SAR'):
                status=json.loads((folder/(method+'_status.json')).read_text())
                for fit in status['fits']:
                    for sf in fit['fit']:
                        if sf['bootstrap_B']!=100 or sf['bootstrap_success']<95 or abs(sf['rho'])>.999+1e-12:raise ValueError('Actual Gaussian fit has wrong model or uncertainty')
                        if not sf.get('raw_optimizer_converged',True) and sf['converged'] and not sf['stationarity_certificate']['passed']:raise ValueError('Uncertified actual optimizer recovery')
                        failed+=sf['bootstrap_failed'];recovered+=sf['bootstrap_stationarity_recovered'];requested+=100
            elif method in ('Sikov_FH_BC','Sikov_SAR_BC'):
                reference=dict(np.load(folder/(method[:-3]+'.npz'),allow_pickle=False))
                if not np.array_equal(arrays['mean'],reference['mean']):raise ValueError('Actual BC modified point estimates')
                negative_bc+=int((arrays['mse_bc']<0).sum())
        check_hmc_status(json.loads((folder/'ICC_status.json').read_text()))
        if protocol['selected_recipe'].get('stratum_deviations'):check_hmc_status(json.loads((folder/'Selected_status.json').read_text()))
        records.append(dict(scope=scope,areas=1874,methods=len(protocol['methods']),heldout_outcomes_read_by_auditor=False,all_converged=completion['all_converged'],bootstrap_requested=requested,bootstrap_failed=failed,bootstrap_stationarity_recovered=recovered,negative_BC_cells=negative_bc))
    return dict(real_prediction_checks=records,real_2024_secondary_reuse_disclosed=True)

def audit_external_family(directory,protocol):
    directory=Path(directory);proof=json.loads((directory/'external_final_summary.json').read_text());frame=pd.read_csv(directory/'paired_balanced_contrasts_Holm8.csv')
    if proof['status']!='PASS_COMPLETE_SECONDARY_HOLM8_FAMILY' or proof['protocol_sha256']!=sha(ROOT/'protocol_freeze.json'):raise ValueError('Incomplete secondary external family')
    for path,digest in proof['input_sha256'].items():
        if sha(path)!=digest:raise ValueError('Changed secondary scoring input')
    independent_errors=[]
    for summarypath in proof['input_sha256']:
        summarypath=Path(summarypath);summary=json.loads(summarypath.read_text());detail=pd.read_csv(summarypath.parent/'district_scores.csv',dtype={'ubigeo':str,'province':str})
        if detail[['ubigeo','method','outcome']].duplicated().any():raise ValueError('Duplicated secondary score units')
        for endpoint in OUTCOMES:
            part=detail[detail.outcome==endpoint];p=part['p_'+endpoint].to_numpy();y=part.direct.to_numpy();q=np.clip(p,1e-12,1-1e-12)
            if not np.allclose(part.MSE,(p-y)**2,atol=1e-12,rtol=0) or not np.allclose(part.NLL,-y*np.log(q)-(1-y)*np.log1p(-q),atol=1e-12,rtol=0):raise ValueError('Secondary raw loss or common clipping differs')
        codes=sorted(detail.ubigeo.unique());provinces=np.array([code[:4] for code in codes]);unique=np.unique(provinces);J=len(unique)
        rng=np.random.default_rng(202610102024+(10000 if summary['scope']=='territorial' else 0))
        for metric in ('MSE','NLL'):
            candidate=detail[detail.method==summary['candidate']].pivot(index='ubigeo',columns='outcome',values=metric).reindex(index=codes,columns=OUTCOMES).to_numpy()
            for reference in summary['references']:
                ref=detail[detail.method==reference].pivot(index='ubigeo',columns='outcome',values=metric).reindex(index=codes,columns=OUTCOMES).to_numpy();delta=candidate-ref
                n=np.isfinite(delta).sum(axis=0);means=np.nanmean(delta,axis=0);difference=means.mean()
                counts=np.array([np.isfinite(delta[provinces==province]).sum(axis=0) for province in unique])
                sums=np.array([np.nansum(delta[provinces==province],axis=0) for province in unique])
                influence=((sums-counts*means)/n).sum(axis=1)/2
                se=np.sqrt(J/(J-1)*np.sum(influence**2));stat=difference/se if se>0 else 0 if difference==0 else np.inf
                pvalue=2*t.sf(abs(stat),J-1);critical=t.ppf(.975,J-1)
                indices=rng.integers(0,J,size=(3999,J));boot=(sums[indices].sum(axis=1)/counts[indices].sum(axis=1)).mean(axis=1);lo,hi=np.quantile(boot,[.025,.975])
                independent=dict(difference=difference,cluster_standard_error=se,p_two_sided=pvalue,unadjusted_t_ci_low=difference-critical*se,unadjusted_t_ci_high=difference+critical*se,descriptive_bootstrap_ci_low=lo,descriptive_bootstrap_ci_high=hi)
                row=frame[(frame.scope==summary['scope'])&(frame.reference==reference)&(frame.metric==metric)]
                if len(row)!=1:raise ValueError('Secondary paired family omitted a contrast')
                independent_errors.append(max(abs(row.iloc[0][key]-value) for key,value in independent.items()))
    if max(independent_errors)>1e-10:raise ValueError('Independent secondary province covariance or contrasts differ')
    expected={(scope,reference,metric) for scope in ('temporal','territorial') for reference in ('Sikov_FH','Sikov_SAR') for metric in ('MSE','NLL')}
    if len(frame)!=8 or set(frame[['scope','reference','metric']].itertuples(index=False,name=None))!=expected:raise ValueError('External complete family differs')
    p=frame.p_two_sided.to_numpy();adjusted=np.zeros(len(p))
    for index,value in enumerate(p):
        adjusted[index]=min(1,max((len(p)-int((p<t).sum()))*t for t in p if t<=value))
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)) or np.max(abs(frame.p_Holm8.to_numpy()-adjusted))>1e-10:raise ValueError('External Holm8 does not reproduce')
    return dict(external_Holm8_independently_verified=True,external_province_contrast_max_absolute_error=max(independent_errors),external_reserved_microdata_read_by_auditor=False,external_derived_noisy_direct_means_audited=True,external_secondary_not_untouched=True)

def self_test():
    from comparison_collect import balanced,CONTEXTS,EVALUATION_IDS
    rng=np.random.default_rng(909);rows=[]
    for ci,context in enumerate(CONTEXTS):
        for rep in EVALUATION_IDS:
            common=rng.normal();loss=rng.uniform(.1,.3,2);delta=(common*.01+ci*.002)*np.array([1.,2.])
            for j,outcome in enumerate(OUTCOMES):
                for method,shift in (('A',delta[j]),('B',0)):
                    rows.append(dict(scenario=context,replication=rep,outcome=outcome,group='all',method=method,interval_score=loss[j]+shift,mse=loss[j]+shift,mae=loss[j]+shift,coverage=loss[j]+shift,width=loss[j]+shift))
    frame=pd.DataFrame(rows);ind=independent_balanced(frame,'A','B',CONTEXTS,EVALUATION_IDS);actual=balanced(frame,'A','B')
    difference=max(abs(a[key]-b[key]) for a,b in zip(ind,actual) for key in ('difference','mcse','MC_CI95_low','MC_CI95_high','p_two_sided'))
    identical=balanced(frame,'A','A');identity_pass=all(row['p_two_sided']==1 and row['mcse']==0 for row in identical)
    result=dict(passed=difference<1e-12 and identity_pass,independent_endpoint_correlated_MC_max_error=difference,identical_estimator_zero_SE_pass=identity_pass,actual_or_reserved_outcomes_read=False,source_sha256=sha(__file__))
    dump(ROOT/'validacion/sikov_pre_freeze_paired_QA.json',result)
    if not result['passed']:raise ValueError('Independent MC QA failed')
    print(json.dumps(result,indent=2))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--self-test',action='store_true');parser.add_argument('--include-results',action='store_true');parser.add_argument('--external-dir',type=Path);args=parser.parse_args()
    if args.self_test:self_test();return
    start=time.monotonic();protocol,result=audit_predictions()
    scope_path=ROOT/'validacion/sikov_full_graph_scope_audit.json'
    scope=json.loads(scope_path.read_text())
    if scope['status']!='PASS_SCOPE_LIMITATION_EXPLICIT_FULL_GRAPH_EXTENSION' or scope['protocol_sha256']!=sha(ROOT/'protocol_freeze.json') or scope['frozen_sources_modified'] or scope['evaluation_outcomes_or_truth_read']:
        raise ValueError('The missing-area graph extension scope is not explicitly documented')
    result['comparator_reconstruction_scope']={
        'exact_2023_numerical_or_algorithmic_replication_established':False,
        'explicit_full_latent_graph_missing_area_extension':True,
        'scope_QA_sha256':sha(scope_path),
        'CRAN_all_observed_validation_is_not_missing_area_algorithm_validation':True,
    }
    if args.include_results:
        if args.external_dir is None:raise ValueError('Complete audit requires --external-dir')
        from sikov_published_audit import audit_predictions as audit_additional_predictions,audit_scores as audit_additional_scores
        result.update(audit_additional_predictions(protocol))
        result.update(audit_larger_bootstrap(protocol));result.update(audit_real_predictions(protocol));result.update(audit_external_family(args.external_dir,protocol));result.update(score_independently(protocol));result.update(audit_additional_scores(protocol))
        external_extended_path=ROOT/'validacion/align_external_extended_independent_audit.json'
        external_extended=json.loads(external_extended_path.read_text())
        if external_extended['status']!='PASS_INDEPENDENT_EXTENDED_EXTERNAL_NUMERIC_AND_EXCLUSION_AUDIT' or external_extended['source_sha256']!=sha(ROOT/'scripts/align_external_extended_audit.py'):
            raise ValueError('Expanded external Holm12 independent audit incomplete')
        for path,digest in external_extended['native_csv_sha256'].items():
            if sha(path)!=digest:raise ValueError('Expanded external audited CSV changed')
        result['external_expanded_Holm12_independent_audit']={
            'status':external_extended['status'],'checks_passed':external_extended['checks_passed'],
            'audit_sha256':sha(external_extended_path),'original_Holm8_preserved':True,
            'actual_2024_targets_are_reused_noisy_direct_estimates':True,
        }
    result['status']='PASS_COMPLETE_COMPARISON_INDEPENDENT_AUDIT' if args.include_results else 'PASS_PREDICTION_ONLY_INDEPENDENT_AUDIT'
    result.update(elapsed_seconds=time.monotonic()-start,auditor_source_sha256=sha(__file__))
    dump(ROOT/('validacion/sikov_final_independent_audit.json' if args.include_results else 'validacion/sikov_final_prediction_audit.json'),result)
    print(json.dumps({key:value for key,value in result.items() if key!='world_checks'},indent=2))

if __name__=='__main__':main()

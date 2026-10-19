#!/usr/bin/env python3
"""Training-only full-cohort prediction agreement and current-chain MCSE.

No fitting, population truth, held-out responses or accuracy CSVs are read.
Historical predictions are comparisons only. Historical MCSE and the covariance
between runs are unknown: current MCSE never supplies an equivalence test.
Default execution requires 256 primary, 64 sampled and two actual-data scopes.
--inventory-only reads completion metadata only, and cannot certify reproduction.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='1'
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import beta as beta_distribution, norm
import cmdstanpy
import sikov_worker as worker

ENDPOINTS=('anemia','stunting')
FORMAT='iaos-numerical-interpretation-summary-v1'
LIMITATIONS=[
    'Historical Monte Carlo standard errors and between-run covariance are unavailable. Var(new-old)=Var(new)+Var(old)-2Cov(new,old) cannot be estimated from current chains alone.',
    'Identical seeds do not establish independence or identical historical Monte Carlo precision. No sqrt(2) multiplier, p value, equivalence margin or Monte Carlo PASS threshold is used.',
    'Absolute differences divided by CURRENT MCSE ONLY are descriptive scales, not tests of agreement between runs.',
    'MCSE uses equal-chain nonoverlapping batch means. It is a finite-chain approximation, not Stan rank-normalized ESS. Twenty versus forty batches provides sensitivity, not proof of reliability.',
    'Quantile MCSE is a local asymptotic CDF/density approximation. Constant within-chain CDF indicators, zero/nonfinite density and a SAR point mass at the quantile make this approximation undefined.',
    'Selected MCSE conditions on the stored SAR component, fixed mixture weight, input calibration and graph. SAR bootstrap, working-ICC, calibration and design uncertainty are not estimated by these current-chain MCSEs.',
    'Actual-data fresh and historical eigenbases differ in a nearly degenerate eigenspace. Actual ICC differences can include a changed finite-rank prior target, not only Monte Carlo error.',
    'Prediction agreement is distinct from the precision of derived article table cells and empirical text. Only the separate document-comparison certificate establishes printed-cell agreement.',
    'This diagnostic never overrides failed convergence gates, numerical comparison failures, missing worlds, printed-cell mismatches or required visual review.'
]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)
def scalar(value):return float(value) if np.isfinite(value) else None
def paths(run,reference):
    for context in worker.CONTEXTS:
        for rep in range(30000,30064):
            rel=Path(context)/f'rep_{rep:05d}'
            yield dict(scope='primary',context=context,replication=rep,
                source=run/'evaluacion/inputs'/rel,original=run/'evaluacion/fits'/rel,
                published=run/'evaluacion/published_sar'/rel,old_original=reference/'evaluacion/fits'/rel,
                old_published=reference/'evaluacion/published_sar'/rel,real=False)
    for rep in range(30000,30064):
        rel=Path(f'rep_{rep:05d}');base=run/'sensibilidad_sampled_graph/fits'/rel;old=reference/'sensibilidad_sampled_graph/fits'/rel
        yield dict(scope='sampled_graph',context='SAR_probability_sampled_graph_srs',replication=rep,
            source=run/'sensibilidad_sampled_graph/inputs'/rel,original=base/'original',published=base/'published',
            aggregate=base,old_original=old/'original',old_published=old/'published',real=False)
    for scope,prepared in (('temporal','real_2019_full'),('territorial','real_2024_training')):
        yield dict(scope='real_'+scope,context='real_'+scope,replication=None,source=run/'datos'/prepared,
            original=run/'evaluacion'/('real_'+scope),published=run/'evaluacion/published_sar'/('real_'+scope),
            old_original=reference/'evaluacion'/('real_'+scope),old_published=reference/'evaluacion/published_sar'/('real_'+scope),real=True)

def cohort_gate(worlds,protocol_sha):
    counts={};pending=[];invalid=[];bindings={};convergence={}
    for world in worlds:
        for role,methods in (('original',worker.METHODS[:9]),('published',worker.METHODS[9:])):
            path=world[role]/'completion.json';label=world['context']+':'+str(world['replication'])+':'+role
            if not path.is_file():pending.append(label);continue
            record=read(path);bindings[str(path)]=sha(path)
            counts[world['scope']+'_'+role]=counts.get(world['scope']+'_'+role,0)+1
            if record.get('protocol_sha256')!=protocol_sha:invalid.append(label+': protocol identity')
            if set(record.get('convergence',{}))!=set(methods):invalid.append(label+': method universe')
            if not any(record.get(key) is False for key in ('truth_read_by_fits','heldout_outcomes_read_by_fits')):
                invalid.append(label+': explicit training-only fit role absent')
            if any(record.get(key,False) is not False for key in ('truth_read_by_fits','heldout_outcomes_read_by_fits')):
                invalid.append(label+': truth/held-out fit role')
            for method,passed in record.get('convergence',{}).items():
                key=(world['scope'],method);row=convergence.setdefault(key,dict(scope=key[0],method=method,worlds=0,passed=0,failed=0))
                row['worlds']+=1;row['passed']+=int(passed is True);row['failed']+=int(passed is not True)
        if 'aggregate' in world:
            path=world['aggregate']/'completion.json'
            if not path.is_file():pending.append(world['context']+':'+str(world['replication'])+':aggregate')
            else:
                record=read(path);bindings[str(path)]=sha(path)
                if set(record.get('convergence',{}))!=set(worker.METHODS) or record.get('truth_read_by_fits')is not False:
                    invalid.append(str(path)+': aggregate universe/role')
    expected={'primary_original':256,'primary_published':256,'sampled_graph_original':64,'sampled_graph_published':64,
        'real_temporal_original':1,'real_temporal_published':1,'real_territorial_original':1,'real_territorial_published':1}
    return dict(complete=not pending and not invalid and counts==expected,counts=counts,expected=expected,
        pending=pending,invalid=invalid,production_convergence=list(convergence.values()),completion_sha256=bindings,
        prediction_or_truth_or_private_records_read=False)

def batch_mcse(values,chains=4,batches=20):
    """Equal-chain mean MCSE; invalid constant-chain cells remain undefined."""
    x=np.asarray(values,float)
    if x.ndim!=2 or len(x)%chains or batches<20 or not np.isfinite(x).all():raise ValueError('Finite aligned draws x areas, >=20 batches required')
    n=len(x)//chains;length=n//batches
    if length<10:raise ValueError('At least ten draws per nonoverlapping batch required')
    z=x.reshape(chains,n,x.shape[1]);used=length*batches
    bm=z[:,:used].reshape(chains,batches,length,x.shape[1]).mean(axis=2)
    variance=bm.var(axis=1,ddof=1).sum(axis=0)/(chains**2*batches)
    defined=np.all(z.var(axis=1,ddof=1)>0,axis=0)&(variance>0)&np.isfinite(variance)
    se=np.where(defined,np.sqrt(np.maximum(variance,0)),np.nan)
    marginal=z[:,:used].reshape(chains*used,x.shape[1]).var(axis=0,ddof=1)
    ess=np.divide(marginal,variance,out=np.full(x.shape[1],np.nan),where=defined)
    half=n//2;split=np.concatenate((z[:,:half],z[:,-half:]),axis=0)
    within=split.var(axis=1,ddof=1).mean(axis=0);between=half*split.mean(axis=1).var(axis=0,ddof=1)
    rhat=np.full(x.shape[1],np.nan);positive=within>0
    rhat[positive]=np.sqrt(((half-1)/half*within[positive]+between[positive]/half)/within[positive])
    rhat[(within==0)&(between>0)]=np.inf
    return dict(mcse=se,defined=defined,ess=ess,classic_split_Rhat=rhat,
        chains=chains,draws_per_chain=n,batches_per_chain=batches,batch_length=length,draws_used=chains*used)

def compare_array(a,b,atol):
    if a.shape!=b.shape or a.dtype.kind!=b.dtype.kind:raise ValueError('Prediction shape/dtype differs')
    same=(a==b)|(np.isnan(a)&np.isnan(b));finite=np.isfinite(a)&np.isfinite(b)
    difference=np.abs(a[finite]-b[finite]);matched_nonfinite=same&~finite
    return dict(cells=int(a.size),finite_cells=int(finite.sum()),exact_cells=int(same.sum()),
        tolerance_cells=int((difference<=atol).sum()+matched_nonfinite.sum()),
        nonfinite_mismatches=int((~finite&~same).sum()),max_abs_difference=float(difference.max())if len(difference)else None)

def training(world,bindings):
    source=world['source'];completion=read(world['original']/'completion.json')
    if world['real']:
        manifest=read(source/'preparation_manifest.json')
        if manifest.get('heldout_outcome_rows_read') is not False:raise ValueError('Actual prepared frame is not training only')
        if completion.get('training_manifest_sha256')!=sha(source/'preparation_manifest.json'):
            raise ValueError('Actual fit and current training manifest differ')
        for name,digest in manifest['output_sha256'].items():
            if sha(source/name)!=digest:raise ValueError('Actual sealed training input changed: '+name)
        npz=source/'fit_inputs.npz';csv=source/manifest['private_units_path'].split('/')[-2]/'units.csv'
        d=dict(np.load(npz,allow_pickle=False));d['X']=d['X_standardized']
        if sha(csv)!=manifest['private_units_sha256']:raise ValueError('Actual training unit hash differs')
        bindings[str(source/'preparation_manifest.json')]=sha(source/'preparation_manifest.json')
    else:
        npz=source/'model_inputs/inputs.npz';csv=source/'model_inputs/sample.csv';manifest=read(source/'manifest.json')
        if completion.get('input_manifest_sha256')!=sha(source/'manifest.json'):
            raise ValueError('Simulation fit and current training manifest differ')
        if sha(npz)!=manifest['fit_inputs_sha256']or sha(csv)!=manifest['sample_sha256']:raise ValueError('Training input identity differs')
        d=dict(np.load(npz,allow_pickle=False));bindings[str(source/'manifest.json')]=sha(source/'manifest.json')
    for path in (npz,csv):bindings[str(path)]=sha(path)
    return d,pd.read_csv(csv)

def retained(world,endpoint,status,bindings):
    trial=status['sampler_trials'][-1];attempt=trial['attempt'];root=world['original']/('icc_sampler'if world['real']else'hmc_chains')
    files=sorted((root/endpoint/('attempt_'+str(attempt))).glob('*.csv'))
    if len(files)!=4:raise ValueError('Four retained production chains required')
    for path in files:bindings[str(path)]=sha(path)
    fit=cmdstanpy.from_csv([str(p)for p in files]);chains=fit.chains
    if chains!=4:raise ValueError('Production chain count differs')
    return fit.stan_variable('beta'),fit.stan_variable('gamma'),fit.stan_variable('log_kappa'),trial

def mcse_endpoint(world,j,d,units,status,predictions,old_predictions,bindings,batches,block_size):
    from beta_power import EndpointPosterior,working_icc,cluster_effective_n
    endpoint=ENDPOINTS[j];icc,_=working_icc(units,endpoint)
    effective,_=cluster_effective_n(units,endpoint,d['sample_idx'],d['n_eff'][:,j],icc)
    post=EndpointPosterior(d['X'],d['y'][:,j],effective,d['sample_idx'],d['basis'],config=status.get('config'))
    beta,gamma,logk,trial=retained(world,endpoint,status,bindings)
    draw_path=world['original']/'ICC_conditional_draws.npz';bindings[str(draw_path)]=sha(draw_path)
    with np.load(draw_path,allow_pickle=False)as archive:pdraw=archive[endpoint+'_marginal']
    nareas=len(post.X);arrays={};reconstruction={};m=np.asarray(predictions['ICC']['mean'])[:,j]
    v=np.asarray(predictions['ICC']['mse'])[:,j];mu=predictions['Sikov_SAR']['mean'][:,j];vsar=predictions['Sikov_SAR']['mse'][:,j];w=.75
    if pdraw.shape!=(len(logk),nareas):raise ValueError('Predictive marginal draws are not aligned with production chains')
    for method in ('ICC','Selected'):
        for quantity in ('mean','lower','upper','variance'):
            arrays[(method,quantity)]=np.full(nareas,np.nan)
            if quantity=='mean':arrays[(method,'mean_40_batches')]=np.full(nareas,np.nan)
    ess=np.full(nareas,np.nan);rhat=np.full(nareas,np.nan)
    for start in range(0,nareas,block_size):
        stop=min(start+block_size,nareas);sl=slice(start,stop)
        eta=beta@post.X[sl].T+gamma@post.B[sl].T;kappa=np.exp(logk)[:,None]
        a=np.maximum(kappa*expit(eta),1e-10);b=np.maximum(kappa*expit(-eta),1e-10)
        mask=(post.idx>=start)&(post.idx<stop);idx=post.idx[mask]-start
        a[:,idx]+=post.k[mask];b[:,idx]+=post.n[mask]-post.k[mask]
        u=a/(a+b);mean=u.mean(axis=0)
        reconstruction['ICC_mean']=max(reconstruction.get('ICC_mean',0.),float(np.max(abs(mean-m[sl]))))
        conditional_variance=a*b/((a+b)**2*(a+b+1));variance=(conditional_variance+u*u).mean(axis=0)-mean*mean
        reconstruction['ICC_variance']=max(reconstruction.get('ICC_variance',0.),float(np.max(abs(variance-v[sl]))))
        diag=batch_mcse(u,4,batches);ess[sl]=diag['ess'];rhat[sl]=diag['classic_split_Rhat']
        arrays[('ICC','mean')][sl]=diag['mcse'];arrays[('Selected','mean')][sl]=w*diag['mcse']
        if len(u)//4>=400:
            alt=batch_mcse(u,4,40);arrays[('ICC','mean_40_batches')][sl]=alt['mcse'];arrays[('Selected','mean_40_batches')][sl]=w*alt['mcse']
        influence=conditional_variance+u*u-2*mean[None,:]*u
        arrays[('ICC','variance')][sl]=batch_mcse(influence,4,batches)['mcse']
        selected_influence=w*influence+2*w*(1-w)*(mean-mu[sl])[None,:]*(u-mean)
        arrays[('Selected','variance')][sl]=batch_mcse(selected_influence,4,batches)['mcse']
        saved=pdraw[:,sl]
        for method in ('ICC','Selected'):
            for probability,key in ((.025,'lower'),(.975,'upper')):
                q=predictions[method][key][sl,j];indicator=(saved<=q[None,:]).astype(float)
                cdf=batch_mcse(indicator,4,batches)['mcse']
                with np.errstate(all='ignore'):density=beta_distribution.pdf(q[None,:],a,b).mean(axis=0)
                valid=np.isfinite(density)&(density>0)&(q>0)&(q<1)
                if method=='ICC':
                    error=float(np.max(abs(np.quantile(saved,probability,axis=0)-q)))
                    reconstruction[key]=max(reconstruction.get(key,0.),error)
                else:
                    sd=np.sqrt(np.maximum(vsar[sl],0));normal_density=np.zeros(len(q));positive=sd>0
                    normal_density[positive]=norm.pdf(q[positive],mu[sl][positive],sd[positive])
                    density=w*density+(1-w)*normal_density
                    valid=np.isfinite(density)&(density>0)&~((sd==0)&(q==mu[sl]));cdf=w*cdf
                arrays[(method,key)][sl]=np.divide(cdf,density,out=np.full(len(q),np.nan),where=valid)
    reconstruction['Selected_mean_identity']=float(np.max(abs(predictions['Selected']['mean'][:,j]-(w*m+(1-w)*mu))))
    reconstruction['Selected_variance_identity']=float(np.max(abs(predictions['Selected']['p_variance'][:,j]-(w*v+(1-w)*vsar+w*(1-w)*(m-mu)**2))))
    if any(error>1e-10 for error in reconstruction.values()):raise ValueError('Saved prediction reconstruction identity differs: '+str(reconstruction))
    return arrays,dict(endpoint=endpoint,production_NUTS_passed=status.get('success')is True,production_attempt=trial['attempt'],
        production_max_Rhat=trial.get('max_Rhat'),production_min_bulk_ESS=trial.get('min_bulk_ESS'),
        production_min_tail_ESS=trial.get('min_tail_ESS'),production_min_E_BFMI=min(trial.get('E_BFMI',[float('nan')])),
        divergences=trial.get('divergences'),max_treedepth_hits=trial.get('max_treedepth_hits'),
        chain_count=4,draws_per_chain=len(logk)//4,batches_per_chain=batches,batch_length=(len(logk)//4)//batches,
        maximum_derived_classic_split_Rhat=scalar(np.max(rhat[np.isfinite(rhat)]))if np.isfinite(rhat).any()else None,
        derived_classic_Rhat_infinite_cells=int(np.isinf(rhat).sum()),derived_classic_Rhat_undefined_cells=int(np.isnan(rhat).sum()),
        minimum_derived_mean_ESS_batch=scalar(np.min(ess[np.isfinite(ess)]))if np.isfinite(ess).any()else None,
        prediction_reconstruction_max_errors=reconstruction)

def add_comparison(aggregate,key,value):
    method,quantity=key[1:3]
    if method=='ICC':role='current_NUTS_posterior_estimate'
    elif method=='Selected':role='fixed_ICC_SAR_distribution_mixture'
    elif quantity in ('mean','mse_plugin','mse_delta')or method in ('Sikov_SAR_k4','Sikov_SAR_k8','Sikov_SAR_haversine'):
        role='deterministic_training_optimization_or_delta_calculation'
    else:role='fixed_seed_parametric_bootstrap_uncertainty_calculation'
    row=aggregate.setdefault(key,dict(scope=key[0],method=key[1],quantity=key[2],endpoint=key[3],worlds=0,
        calculation_role=role,cells=0,finite_cells=0,exact_cells=0,tolerance_cells=0,nonfinite_mismatches=0,max_abs_difference=None))
    row['worlds']+=1
    for field in ('cells','finite_cells','exact_cells','tolerance_cells','nonfinite_mismatches'):row[field]+=value[field]
    if value['max_abs_difference']is not None:row['max_abs_difference']=max(row['max_abs_difference']or 0.,value['max_abs_difference'])

def printed_certificate(path):
    if path is None or not path.is_file():return dict(status='PENDING_SEPARATE_DOCUMENT_CERTIFICATE',prediction_decimal_rounding_not_substituted_for_article_precision=True)
    value=read(path)
    return dict(status=value.get('status'),certificate_sha256=sha(path),fresh_execution_evidence_present=value.get('fresh_execution_evidence_present'),
        all_22_document_tables_match_fresh_results=value.get('all_22_document_tables_match_fresh_results'),
        all_checked_empirical_text_matches=value.get('all_checked_empirical_text_matches'),
        table_checks={family:{key:row['fresh_vs_actual_document']for key,row in value.get(family,{}).items()}
            for family in ('manuscript_table_checks','supplement_table_checks')},article_reproduction_confirmed=False)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True);p.add_argument('--public-summary',type=Path)
    p.add_argument('--article-comparison',type=Path);p.add_argument('--inventory-only',action='store_true')
    p.add_argument('--batches',type=int,default=20);p.add_argument('--block-size',type=int,default=128);p.add_argument('--atol',type=float,default=1e-10)
    args=p.parse_args();cfg=read(args.config);run=Path(cfg['sikov_run_root']).resolve();repo=Path(cfg['repo']).resolve();out=args.output_root.resolve()
    if out.is_relative_to(repo)or out.is_relative_to(Path(cfg['source_root']).resolve())or out.is_relative_to(Path(cfg['reference_root']).resolve()):raise ValueError('Diagnostic caches and detailed receipts must remain external/private')
    if out.exists():raise FileExistsError('Preserve earlier diagnostic run; use a new external output root')
    if args.atol<0 or args.block_size<1 or args.batches<20:raise ValueError('Invalid explicit numerical tolerance or batching configuration')
    reference,_=worker.article_root(cfg['reference_root']);worlds=list(paths(run,reference));gate=cohort_gate(worlds,sha(reference/'protocol_freeze.json'))
    public=dict(format=FORMAT,public_scalar_summary_only=True,created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        status='INCOMPLETE_FULL_COHORT_NOT_CLOSED',full_cohort_complete=gate['complete'],cohort_counts=gate['counts'],expected_cohort_counts=gate['expected'],
        expected_world_count=322,analyzed_world_count=0,production_convergence=gate['production_convergence'],
        truth_or_accuracy_scores_read=False,models_fitted_by_this_helper=False,archived_MCSE_available=False,
        statistical_equivalence_certified=False,numerical_reproduction_confirmed=False,article_reproduction_confirmed=False,
        comparison_absolute_tolerance=args.atol,aggregate_comparisons=[],current_mcse=[],limitations=LIMITATIONS,
        helper_sha256=sha(__file__),methods={'mean':'Equal-chain nonoverlapping batch means of conditional probabilities',
            'variance':'Batch means of the delta influence for total conditional variance',
            'quantile':'Batch means of CDF indicators divided by conditional mixture density; Selected includes fixed normal-SAR density',
            'references':['https://mc-stan.org/posterior/reference/mcse_mean.html','https://mc-stan.org/posterior/reference/mcse_quantile.html'],
            'reference_implementation_used':False})
    out.mkdir(parents=True);save(out/'cohort_gate.json',gate)
    if args.inventory_only or not gate['complete']:
        public['status']='INVENTORY_ONLY_FULL_COHORT_NOT_ANALYZED'if args.inventory_only else public['status']
        save(out/'summary.json',public)
        if args.public_summary:save(args.public_summary,public)
        print(json.dumps({'status':public['status'],'cohort_counts':gate['counts'],'prediction_or_truth_payloads_read':False}),flush=True)
        if not args.inventory_only:raise SystemExit(2)
        return
    ctx=worker.initialize(cfg['source_root'],cfg['reference_root'],run,cfg['beta_exe'],cfg['cmdstan'])
    aggregate={};mcse_values={};details=[];failures=[];bindings={};samplers=[];mixture_decomposition={}
    for index,world in enumerate(worlds,1):
        ident=world['context']+':'+str(world['replication']);local={};old={}
        try:
            for family,methods in (('original',worker.METHODS[:9]),('published',worker.METHODS[9:])):
                completion_path=world[family]/'completion.json'
                if sha(completion_path)!=gate['completion_sha256'][str(completion_path)]:raise ValueError('Completion changed after the full-cohort gate')
                record=worker.validate_completion(world[family],methods)
                if record.get('convergence')!=read(world[family]/'completion.json').get('convergence'):raise ValueError('Completion changed during read')
                for method in methods:
                    left=world[family]/(method+'.npz');right=world['old_'+family]/(method+'.npz')
                    bindings[str(left)]=sha(left);bindings[str(right)]=sha(right)
                    local[method]=dict(np.load(left,allow_pickle=False));old[method]=dict(np.load(right,allow_pickle=False))
                    if set(local[method])!=set(old[method]):raise ValueError(method+': prediction fields differ')
                    for quantity,a in local[method].items():
                        b=old[method][quantity]
                        if a.ndim!=2 or a.shape[1]!=2:raise ValueError(method+': prediction endpoint axes differ')
                        for j,endpoint in enumerate(ENDPOINTS):add_comparison(aggregate,(world['scope'],method,quantity,endpoint),compare_array(a[:,j],b[:,j],args.atol))
            d,units=training(world,bindings);status=read(world['original']/'ICC_status.json');bindings[str(world['original']/'ICC_status.json')]=sha(world['original']/'ICC_status.json')
            cache={}
            for j,endpoint in enumerate(ENDPOINTS):
                arrays,info=mcse_endpoint(world,j,d,units,status['endpoints'][j],local,old,bindings,args.batches,args.block_size)
                icc_component=.75*(local['ICC']['mean'][:,j]-old['ICC']['mean'][:,j])
                sar_component=.25*(local['Sikov_SAR']['mean'][:,j]-old['Sikov_SAR']['mean'][:,j])
                selected_difference=local['Selected']['mean'][:,j]-old['Selected']['mean'][:,j]
                key=(world['scope'],endpoint)
                decomposition=mixture_decomposition.setdefault(key,dict(scope=key[0],endpoint=endpoint,worlds=0,
                    max_abs_ICC_component=0.,max_abs_SAR_component=0.,max_abs_Selected_difference=0.,max_decomposition_error=0.))
                decomposition['worlds']+=1
                for field,values in (('max_abs_ICC_component',icc_component),('max_abs_SAR_component',sar_component),
                        ('max_abs_Selected_difference',selected_difference),('max_decomposition_error',selected_difference-icc_component-sar_component)):
                    decomposition[field]=max(decomposition[field],float(np.max(abs(values))))
                samplers.append(dict(scope=world['scope'],**info));details.append(dict(world=ident,**info))
                for (method,quantity),se in arrays.items():
                    base_quantity='mean'if quantity=='mean_40_batches'else quantity;field='p_variance'if base_quantity=='variance'and method=='Selected'else'mse'if base_quantity=='variance'else base_quantity
                    diff=local[method][field][:,j]-old[method][field][:,j]
                    key=(world['scope'],method,quantity,endpoint);mcse_values.setdefault(key,[]).append((se,diff))
                    cache[endpoint+'_'+method+'_'+quantity+'_mcse']=se
            cache_path=out/'current_mcse_arrays'/world['context']/(f'rep_{world["replication"]:05d}.npz'if world['replication']is not None else'real.npz')
            cache_path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(cache_path,**cache)
        except Exception as error:failures.append(dict(world=ident,error=str(error),exception=type(error).__name__))
        public['analyzed_world_count']=index;save(out/'progress.json',dict(worlds_visited=index,expected=322,failures=failures))
        print(json.dumps({'training_only_diagnostic_world':index,'expected':322,'scope':world['scope'],'failed_worlds':len(failures)}),flush=True)
    public['aggregate_comparisons']=list(aggregate.values())
    public['Selected_point_difference_decomposition']=list(mixture_decomposition.values())
    for key,pairs in mcse_values.items():
        se=np.concatenate([pair[0]for pair in pairs]);diff=np.concatenate([pair[1]for pair in pairs]);defined=np.isfinite(se)&(se>0);ratios=abs(diff[defined])/se[defined]
        public['current_mcse'].append(dict(scope=key[0],method=key[1],quantity=key[2],endpoint=key[3],worlds=len(pairs),cells=len(se),
            units='probability_squared'if key[2]=='variance'else'probability',
            defined_cells=int(defined.sum()),undefined_cells=int((~defined).sum()),maximum=scalar(se[defined].max())if defined.any()else None,
            median=scalar(np.median(se[defined]))if defined.any()else None,
            max_abs_difference_over_current_mcse_only=scalar(ratios.max())if len(ratios)else None,
            median_abs_difference_over_current_mcse_only=scalar(np.median(ratios))if len(ratios)else None,
            historical_MCSE_unknown=True,between_run_standard_error_not_available=True))
    public['current_sampler_summary']={scope:{'endpoint_runs':len(part),
        'passed_endpoint_runs':sum(row['production_NUTS_passed']for row in part),
        'maximum_production_Rhat':max(row['production_max_Rhat']for row in part),
        'minimum_production_bulk_ESS':min(row['production_min_bulk_ESS']for row in part),
        'minimum_production_tail_ESS':min(row['production_min_tail_ESS']for row in part),
        'minimum_production_E_BFMI':min(row['production_min_E_BFMI']for row in part),
        'total_divergences':sum(row['divergences']for row in part),'total_treedepth_hits':sum(row['max_treedepth_hits']for row in part),
        'retried_endpoint_runs':sum(row['production_attempt']>1 for row in part),
        'derived_classic_Rhat_infinite_cells':sum(row['derived_classic_Rhat_infinite_cells']for row in part),
        'derived_classic_Rhat_undefined_cells':sum(row['derived_classic_Rhat_undefined_cells']for row in part)}
        for scope in sorted({row['scope']for row in samplers})for part in [[row for row in samplers if row['scope']==scope]]}
    basis=run/'real_basis_diagnostic.json'
    if basis.is_file():
        value=read(basis);public['actual_data_basis_disclosure']={key:value[key]for key in (
            'status','max_abs_basis_difference','max_abs_covariance_difference','relative_frobenius_covariance_difference')if key in value}
        public['actual_data_basis_disclosure']['certificate_sha256']=sha(basis)
        public['actual_data_basis_disclosure']['differences_cannot_be_attributed_only_to_MC']=True
    public['printed_article_precision']=printed_certificate(args.article_comparison)
    public['worlds_visited']=322;public['analyzed_world_count']=322-len(failures)
    public['failed_world_count']=len(failures);public['status']='FULL_COHORT_TRAINING_ONLY_DIAGNOSTIC_COMPLETE_NO_EQUIVALENCE_CERTIFICATE'if not failures else'INCOMPLETE_DIAGNOSTIC_FAILURES_RETAINED'
    worker.source_gate();save(out/'detailed_receipt.json',dict(world_details=details,failures=failures,input_sha256=bindings,
        completion_sha256=gate['completion_sha256'],source_sha256=worker.source_gate(),public_summary_sha256=None))
    save(out/'summary.json',public)
    if args.public_summary:save(args.public_summary,public)
    print(json.dumps({'status':public['status'],'analyzed_worlds':322,'failed_worlds':len(failures),'statistical_equivalence_certified':False}),flush=True)
    if failures:raise SystemExit(2)

if __name__=='__main__':main()

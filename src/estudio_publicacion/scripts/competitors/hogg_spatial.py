"""Separate full-ICAR BYM2 S2 sensitivity following Hogg supplementD.2.

All positive graph eigenmodes are retained. The Gaussian IID component and
Gaussian modular measurement noise are integrated exactly. This is not the
Australian full case model: no S1 individual residual or benchmarking added.
"""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd
from scipy.linalg import eigh,cho_factor,cho_solve
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import hogg_tsln as h


def graph_factor(area_codes,edges):
    """Exact component-wise scaled Q+ factor, retaining all M-C CAR modes."""
    codes=np.asarray(area_codes).astype(str)
    edge=pd.read_csv(edges,dtype=str)if isinstance(edges,(str,Path))else edges.copy().astype(str)
    if list(edge.columns[:2])!=['area1','area2']:raise ValueError('Need area1,area2 graph columns')
    lookup={code:i for i,code in enumerate(codes)}
    if len(lookup)!=len(codes):raise ValueError('Duplicated graph area')
    pairs=np.array([[lookup[a],lookup[b]]for a,b in edge[['area1','area2']].to_numpy()],int)
    if (pairs[:,0]==pairs[:,1]).any():raise ValueError('Self edge not allowed')
    pairs=np.unique(np.sort(pairs,axis=1),axis=0)
    M=len(codes)
    A=coo_matrix((np.ones(2*len(pairs)),(np.r_[pairs[:,0],pairs[:,1]],
        np.r_[pairs[:,1],pairs[:,0]])),shape=(M,M)).tocsr()
    C,component=connected_components(A,directed=False)
    sizes=np.bincount(component)
    if(sizes<2).any():raise ValueError('Isolated nodes require a declared separate prior; not silently linked')
    factor=np.zeros((M,M-C));scale=np.zeros(M);offset=0;checks=[];order=[]
    for c,n in enumerate(sizes):
        ix=np.flatnonzero(component==c);order.extend(ix.tolist())
        adj=A[ix][:,ix].toarray();Q=np.diag(adj.sum(axis=1))-adj
        vals,vecs=eigh(Q,check_finite=False)
        tolerance=max(vals[-1]*1e-12,1e-14)
        if abs(vals[0])>tolerance or vals[1]<=tolerance:
            raise FloatingPointError('Numerical CAR rank does not equal component size minus one')
        raw=vecs[:,1:]/np.sqrt(vals[1:])
        variances=np.sum(raw**2,axis=1)
        kappa=float(np.exp(np.log(variances).mean()))
        block=raw/np.sqrt(kappa)
        factor[np.ix_(ix,np.arange(offset,offset+n-1))]=block
        scale[ix]=kappa;offset+=n-1
        checks.append({'component':int(c),'nodes':int(n),'rank':int(n-1),
            'kappa_geometric_mean_unscaled_diagonal_Qplus':kappa,
            'scaled_geometric_mean_marginal_variance':float(np.exp(np.log(np.sum(block**2,axis=1)).mean())),
            'sum_to_zero_max_column_residual':float(abs(block.sum(axis=0)).max()),
            'min_retained_eigenvalue':float(vals[1]),'max_eigenvalue':float(vals[-1])})
    metadata={'M':M,'components':int(C),'rank':M-int(C),'undirected_binary_edges':len(pairs),
        'all_positive_modes_retained':True,'low_rank_approximation':False,
        'component_checks':checks,
        'scaling':'Eachconnectedcomponent Q+ computed fromallnonzeroeigenpairs; kappa=geommean diagonal(Q+); no diagonaljitter.',
        'constraints':'Exactcomponentwise sumzero; isolatesfailratherthaninventedneighbours'}
    reference={'C':int(C),'component_size':sizes,'node_order':np.array(order)+1,
               'icar_scale':scale,'E':len(pairs),'node1':pairs[:,0]+1,'node2':pairs[:,1]+1}
    return {'R':M-int(C),'scaled_icar_factor':factor},metadata,reference


def compile_spatial(reference=False):
    name='hogg_stage2_bym2_full_reference'if reference else'hogg_stage2_bym2'
    source=h.HERE/(name+'.stan');stamp=h.HERE/(name+'_compiled_source.json')
    before=hashlib.sha256(source.read_bytes()).hexdigest()
    old=json.loads(stamp.read_text())if stamp.exists()else{}
    model=h._backend().CmdStanModel(stan_file=str(source),force_compile=old.get('stan_sha256')!=before)
    after=hashlib.sha256(source.read_bytes()).hexdigest()
    if before!=after:raise RuntimeError('Spatial source changed during compilation')
    h.write_json(stamp,{'stan_sha256':after,'executable_sha256':hashlib.sha256(Path(model.exe_file).read_bytes()).hexdigest()})
    return model


def recover_stage2_from_saved(run_dir,units,areas,*,outcome,weight,area_col='area',
                             area_features=('X1','X2'),unit_features=(),population_N=None):
    """Reconstruct the identical S1 modular summary from retained S1 traces."""
    run=Path(run_dir);status=json.loads((run/'status.json').read_text())
    if not status['success']:raise ValueError('Source S1/S2 run failed finaldiagnostics')
    if status['input_sha256']!={'units':h._input_hash(units),'areas':h._input_hash(areas)}:
        raise ValueError('Source input identity differs; reconstruct the same frames or file inputs')
    prepared=h.prepare_hogg_data(units,areas,outcome=outcome,weight=weight,area_col=area_col,
        area_features=area_features,unit_features=unit_features,population_N=population_N)
    P=prepared['s1_data']['P'];m=len(prepared['n'])
    columns=['intercept']+[f'beta.{j+1}'for j in range(P)]+[f'effect_area.{j+1}'for j in range(m)]
    traces=[pd.read_csv(p,comment='#',usecols=columns)[columns].to_numpy(float)
            for p in sorted((run/'stage1').glob('*.csv'))]
    if len(traces)!=4:raise ValueError('Four retained source chaintraces required')
    draw=np.concatenate(traces)
    summary=h.aggregate_stage1(prepared,draw[:,0],draw[:,1:P+1],draw[:,P+1:])
    d2,meta=h.prepare_stage2(prepared,summary,profile='author_code',seed=status['config']['seed']+21)
    original=pd.read_csv(run/'stage1_area_summary.csv',dtype={'area':str})
    if not np.array_equal(original.area,prepared['sampled_codes']):raise ValueError('Source areas differ')
    diffs={
        'theta_mean':float(abs(original.theta_s1_mean_all_draws-summary['theta_draws'].mean(axis=0)).max()),
        'theta_sd':float(abs(original.theta_s1_posterior_sd-summary['theta_draws'].std(axis=0,ddof=1)).max()),
        'stable_sampling_sd':float(abs(original.sampling_sd_s1_used_stable-d2['sampling_sd_s1']).max())}
    if max(diffs.values())>1e-10:raise ValueError('Recovered S1 modular summaries differ from source')
    same_subset=np.array_equal(meta['stage2_subset_indices'],status['model_metadata']['stage2_subset_indices'])
    if not same_subset:raise ValueError('Recovered S1 draw subset differs from the source')
    return prepared['all_codes'],d2,{'source_status_sha256':hashlib.sha256((run/'status.json').read_bytes()).hexdigest(),
        'recovery_summary_maxdiff':diffs,'no_S1_refit':True,
        'same_subset_indices':same_subset,'source_inputs_hash_exactly_match':True}


def fixed_hyper_gaussian_check(factor,*,seed=20261005):
    """DensefullBYM2posterior vs exact integrateIID-first construction."""
    rng=np.random.default_rng(seed);M=len(factor);idx=np.arange(M)[::2]
    C=factor@factor.T;sigma=.7;rho=.5
    Ksp=sigma**2*rho*C;A=sigma**2*(1-rho);K=Ksp+A*np.eye(M)
    D=np.linspace(.08,.6,len(idx));residual=rng.normal(size=len(idx))
    observed=K[np.ix_(idx,idx)]+np.diag(D);cf=cho_factor(observed)
    exactmean=K[:,idx]@cho_solve(cf,residual)
    exactcov=K-K[:,idx]@cho_solve(cf,K[idx,:])
    spmean=Ksp[:,idx]@cho_solve(cf,residual)
    spcov=Ksp-Ksp[:,idx]@cho_solve(cf,Ksp[idx,:])
    weights=np.zeros(M);weights[idx]=A/(A+D)
    equivalentmean=(1-weights)*spmean
    equivalentmean[idx]+=weights[idx]*residual
    extra=np.full(M,A);extra[idx]=A*D/(A+D)
    equivalentcov=(1-weights[:,None])*spcov*(1-weights[None,:])+np.diag(extra)
    # Orthonormal component mapping used by the latent reference: Q restricted
    # inversion yields the same proper constrained CAR covariance.
    return {'rho':rho,'sigma':sigma,'sampled_areas':len(idx),
        'posterior_mean_maxerror':float(abs(exactmean-equivalentmean).max()),
        'posterior_covariance_maxerror':float(abs(exactcov-equivalentcov).max()),
        'marginal_covariance_equality':'Ksp+IID+measurement_noise identical before orafter integratingIID',
        'success':bool(np.allclose(exactmean,equivalentmean,atol=1e-12,rtol=1e-12)
                       and np.allclose(exactcov,equivalentcov,atol=1e-12,rtol=1e-12))}


def fit_spatial_stage2(area_codes,modular_data,edges,*,output_dir,config=None,model=None):
    cfg=h.MCMCConfig(parallel_chains=1,phase='spatial_sensitivity')if config is None else config
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    graph,graph_metadata,_=graph_factor(area_codes,edges)
    data={**modular_data,**graph}
    started=time.monotonic();model=compile_spatial()if model is None else model
    fit=h._sample(model,data,cfg,out/'stage2')
    summary=fit.summary();summary.to_csv(out/'stage2_parameter_summary.csv')
    # Only actual sampled parameters; every latent white CAR coordinate is
    # included. Deterministic effects are retained in the fullsummary CSV.
    names=summary.index.to_series().astype(str)
    sampled=names.str.startswith(('intercept','beta[','sigma_area','rho','spatial_z[','omega[','sigma_gvf'))
    diagnostics=h.diagnose_fit(fit,cfg,summary=summary.loc[sampled.values])
    draws=fit.stan_variable('prevalence')
    if not np.isfinite(draws).all()or((draws<=0)|(draws>=1)).any():raise FloatingPointError('Invalidspatialprevalence')
    lo,hi=np.quantile(draws,[.025,.975],axis=0)
    predictions=pd.DataFrame({'area':np.asarray(area_codes).astype(str),'mean':draws.mean(axis=0),
        'lo95_ET':lo,'hi95_ET':hi})
    predictions.to_csv(out/'area_predictions.csv',index=False)
    np.savez_compressed(out/'posterior_area_draws.npz',area=np.asarray(area_codes).astype(str),prevalence=draws)
    status={'method':'Hogg_S2_BYM2_author_component_sensitivity','completed':True,
        'success':diagnostics['success'],'stage2':diagnostics,'config':h.asdict(cfg),
        'seconds_total':time.monotonic()-started,'graph':graph_metadata,
        'priors':'rhoBeta(3.05,1.65),sigmaHalfNormal2; sameauthorcodeS2QRbeta/intercept/GVFpriors',
        'integration':'ExactGaussianIID andmeasurement integration; allnonzeroICARmodesretained',
        'scope':'D.2 spatialcomponent extension, not fullAustralian casereplication; noindividualS1residual orbenchmarking',
        'uncertainty':'AuthorGaussianmodularS1summary, not fulljointunit-levelS1/S2posterior; noautomaticdesigncoverageclaim',
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path(__file__),h.HERE/'hogg_stage2_bym2.stan']}}
    h.write_json(out/'status.json',status)
    return {'predictions':predictions,'draws':draws,'status':status,'fit':fit}


def load_modular_summary(path):
    """Read saved S1 Gaussian summary only; no outcomes or evaluation truth."""
    with np.load(path,allow_pickle=False)as saved:
        codes=saved['area'].astype(str)
        data={name:saved[name].copy()for name in saved.files if name!='area'}
    for name in ['M','m','P','stable_n','unstable_n']:
        data[name]=int(data[name])
    for name in ['sampled_area','stable','unstable']:
        data[name]=data[name].astype(int)
    if data['M']!=len(codes):raise ValueError('Modular summary area count differs')
    return codes,data


def fit_spatial_pair(modular_sources,edges,*,output_dir,config=None,model=None,
                     retry_on_diagnostics=True,retain_sampler_csv=False):
    """S2-only paired endpoint wrapper; source S1 is reused exactly, never refit.

    Preserves all final marginal draws even when the fixed diagnostic retry
    yields unequal counts. Endpoint fits do not identify a joint outcome law.
    """
    cfg=h.MCMCConfig(parallel_chains=1,phase='spatial_sensitivity')if config is None else config
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();model=compile_spatial()if model is None else model
    results={};attempts={};source_hashes={}
    for j,endpoint in enumerate(['anemia','stunting']):
        source=Path(modular_sources[endpoint]);codes,data=load_modular_summary(source)
        source_hashes[endpoint]=hashlib.sha256(source.read_bytes()).hexdigest()
        endpoint_config=h.MCMCConfig(**{**h.asdict(cfg),'seed':cfg.seed+100*j})
        result=fit_spatial_stage2(codes,data,edges,output_dir=out/endpoint/'attempt_0',
                                 config=endpoint_config,model=model)
        attempts[endpoint]=[result['status']]
        if retry_on_diagnostics and not result['status']['success']:
            retry=h.MCMCConfig(**{**h.asdict(endpoint_config),'warmup':1000,
                                 'samples':2000,'adapt_delta':.99})
            result=fit_spatial_stage2(codes,data,edges,output_dir=out/endpoint/'attempt_1',
                                     config=retry,model=model)
            attempts[endpoint].append(result['status'])
        results[endpoint]=result
    a=results['anemia']['predictions'];s=results['stunting']['predictions']
    if not np.array_equal(a.area,s.area):raise ValueError('Spatial endpoint area order differs')
    pred=pd.DataFrame({'area':a.area})
    for endpoint,result in results.items():
        for source,target in [('mean','p_mean_'),('lo95_ET','lo95_'),('hi95_ET','hi95_')]:
            pred[target+endpoint]=result['predictions'][source]
    pred.to_csv(out/'predictions.csv',index=False)
    np.savez_compressed(out/'posterior_draws.npz',area=a.area.to_numpy(),
        anemia=results['anemia']['draws'],stunting=results['stunting']['draws'])
    status={'method':'Hogg_S2_BYM2_author_component_sensitivity','completed':True,
        'success':all(r['status']['success']for r in results.values()),
        'seconds_total':time.monotonic()-started,'all_attempts':attempts,
        'outcomes':{k:v['status']for k,v in results.items()},
        'modular_source_sha256':source_hashes,'S1_refit':False,
        'retry_rule':'One numerical-diagnostic retry at 1000 warmup/2000 retained/adapt_delta .99 with the same seed, graph, priors and four chains; every attempt retained.',
        'outcome_dependence':'Separate marginal endpoint fits; no identified cross-outcome dependence or paired-rule change posterior.',
        'primary_protocol':'Additional sensitivity only, outside the frozen three primary Holm contrasts.',
        'retain_sampler_csv':retain_sampler_csv,
        'full_marginal_draw_counts':{k:len(v['draws'])for k,v in results.items()}}
    if not retain_sampler_csv:
        removed={}
        for endpoint in results:
            for attempt in range(len(attempts[endpoint])):
                for path in(out/endpoint/f'attempt_{attempt}'/'stage2').glob('*.csv'):
                    removed[str(path.relative_to(out))]=hashlib.sha256(path.read_bytes()).hexdigest()
                    path.unlink()
        status['removed_redundant_sampler_csv_sha256']=removed
    h.write_json(out/'status.json',status)
    return {'predictions':pred,'results':results,'status':status}

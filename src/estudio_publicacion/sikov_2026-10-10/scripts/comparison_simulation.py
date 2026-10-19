"""Fresh finite populations for a matched Sikov/ICC comparison.

Auxiliaries are observed census variables; generating outcomes are synthetic.
Fitting never reads evaluation_only. All four contexts and streams are fixed
before development scores. Native probability SAR is an explicit favourable
context for the published Gaussian comparator.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'1')
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
import pandas as pd
from scipy.special import expit, logit, ndtr
from scipy.linalg import solve
from sikov_model import knn_weights, poverty_strata

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT.parent
sys.path.insert(0,str(STUDY/'scripts'))
from models import spatial_basis
CONTEXTS=('SAR_probability_srs','SAR_logit_srs','SAR_logit_psu_sparse','IID_logit_psu_sparse')
DEVELOPMENT_IDS=list(range(8))
EVALUATION_IDS=list(range(30000,30064))
NAMES=['intercept','Altitud','PisoTierra','Cemento','Agua','Elect','Desague','Internet','SIS','EsSalud','NoSeguro','Refrig','Castellano','Analfabet','Rural']
BETAS={
 'anemia':{1:{'intercept':.472,'Altitud':.017,'Internet':-.221,'Castellano':-.176},
           2:{'intercept':.752,'Altitud':.019,'Cemento':-.148,'Agua':-.089,'Internet':-.369,'SIS':-.189,'Castellano':-.232,'Rural':-.071},
           3:{'intercept':.772,'Cemento':-.221,'SIS':-.287,'Castellano':-.221}},
 'stunting':{1:{'intercept':.176,'Altitud':.012,'Refrig':-.388,'Internet':.269,'Analfabet':1.498},
             2:{'intercept':.511,'Agua':.118,'Elect':-.146,'Refrig':-.202,'NoSeguro':-.260,'Analfabet':.771},
             3:{'intercept':.647,'Altitud':.018,'Elect':-.322,'Refrig':-.354,'SIS':-.345,'Analfabet':1.492}}}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else str(x))+'\n')

def prepare_template(aligned_npz, area_csv):
    """Select geography only, forty real districts per poverty stratum."""
    out=ROOT/'datos/simulation_template'
    if out.exists():raise FileExistsError(out)
    d=dict(np.load(aligned_npz,allow_pickle=False))
    X=d['X_raw']; names=d['covariate_names'].astype(str).tolist()
    strata=d['strata']; coords=d['coords']
    assert names==NAMES
    frame=pd.read_csv(area_csv,dtype={'ubigeo':str})
    rng=np.random.default_rng(20261010120)
    chosen=np.sort(np.concatenate([rng.choice(np.flatnonzero(strata==g),40,replace=False) for g in (1,2,3)]))
    X=X[chosen]; strata=strata[chosen]; coords=coords[chosen]
    W=knn_weights(coords,k=6,strata=strata)
    center=X[:,1:].mean(0);scale=X[:,1:].std(0);scale=np.where(scale>0,scale,1.)
    standardized=np.c_[np.ones(len(X)),(X[:,1:]-center)/scale]
    # Shared public geography, coordinate-only spectral approximation for ICC.
    basis=spatial_basis(coords[:,::-1],rank=18,neighbors=6)
    out.mkdir()
    np.savez_compressed(out/'auxiliary.npz',X_raw=X,X=standardized,strata=strata,coords=coords,W=W,basis=basis,covariate_names=np.asarray(names),selected_full_indices=chosen)
    frame.iloc[chosen].to_csv(out/'area_ids.csv',index=False)
    save(out/'manifest.json',{'areas':120,'stratum_sizes':[40,40,40],'seed':20261010120,'selection_uses_outcomes':False,'aligned_inputs_sha256':sha(aligned_npz),'area_ids_sha256':sha(area_csv),'auxiliary_sha256':sha(out/'auxiliary.npz'),'source_sha256':sha(__file__),'source_table_3_5_sign':'Internet +0.269 retained exactly as printed, not silently corrected'})
    return out

def generate(context,rep,phase):
    if context not in CONTEXTS:raise ValueError(context)
    if rep not in (DEVELOPMENT_IDS if phase=='development' else EVALUATION_IDS):raise ValueError('unplanned stream')
    root=ROOT/('desarrollo/inputs' if phase=='development' else 'evaluacion/inputs')/context/f'rep_{rep:05d}'
    if root.exists():raise FileExistsError(root)
    a=dict(np.load(ROOT/'datos/simulation_template/auxiliary.npz',allow_pickle=False))
    X=a['X_raw']; strata=a['strata']; W=a['W']; m=len(X)
    rng=np.random.default_rng(np.random.SeedSequence([20261010,701,CONTEXTS.index(context),rep]))
    srng=np.random.default_rng(np.random.SeedSequence([20261010,702,CONTEXTS.index(context),rep]))
    base=np.zeros((m,2))
    for t,name in enumerate(('anemia','stunting')):
        for g in (1,2,3):
            mask=strata==g
            for key,b in BETAS[name][g].items():base[mask,t]+=X[mask,NAMES.index(key)]*b
    base=np.clip(base,.03,.75)
    innov=rng.normal(size=(m,2));innov[:,1]=.4*innov[:,0]+np.sqrt(.84)*innov[:,1]
    rho=0. if context.startswith('IID') else .6
    C=solve(np.eye(m)-rho*W,np.eye(m))
    C/=np.sqrt(np.mean(np.sum(C*C,axis=1)))
    u=C@innov
    if context=='SAR_probability_srs':p=np.clip(base+u*np.array([.07,.05]),.005,.90)
    else:p=expit(logit(base)+u*np.array([.45,.40]))
    sparse=context.endswith('psu_sparse');J=2 if sparse else 6
    M=60;L=24
    # Sampling supports and missingness are fixed without inspecting outcomes.
    sampled=np.sort(np.concatenate([srng.choice(np.flatnonzero(strata==g),32,replace=False) for g in (1,2,3)]))
    ys=[];variances=[];kish=[];rows=[];truth=[];boundary=[]
    for i in range(m):
        psu=rng.normal(size=(M,2));psu[:,1]=.5*psu[:,0]+np.sqrt(.75)*psu[:,1]
        if not sparse:psu[:]=0
        else:psu*=np.array([.9,.8])
        pp=expit(logit(p[i])[None,:]+psu)
        H=rng.binomial(1,.5,size=(M,L))
        z=rng.normal(size=(M,L,2));z[:,:,1]=.4*z[:,:,0]+np.sqrt(.84)*z[:,:,1]
        yy=(ndtr(z)<pp[:,None,:]).astype(float)
        truth.append(yy.mean(axis=(0,1)))
        if i not in sampled:continue
        selected=srng.choice(M,J,replace=False)
        num=np.zeros((J,2));den=np.zeros((J,2));sumsq=np.zeros(2)
        for j,k in enumerate(selected):
            q=np.where(H[k]==1,.40,.80)
            children=np.flatnonzero(srng.uniform(size=L)<q)
            values=yy[k,children]
            obs=srng.uniform(size=values.shape)<np.array([.85,.90])
            ww=np.where(obs,(M/J)/(q[children,None]*np.array([.85,.90])),0.)
            num[j]=(ww*values).sum(0);den[j]=ww.sum(0);sumsq+=(ww*ww).sum(0)
            for child,value,seen,w in zip(children,values,obs,ww):
                rows.append({'area':i,'psu':f'{i}:{int(k)}','stratum':i,'child_id':f'{i}:{int(k)}:{int(child)}','y_anemia':float(value[0]) if seen[0] else np.nan,'y_stunting':float(value[1]) if seen[1] else np.nan,'weight_anemia':float(w[0]) if seen[0] else np.nan,'weight_stunting':float(w[1]) if seen[1] else np.nan})
        totals=den.sum(0);y=num.sum(0)/totals
        residual=num-den*y
        D=J/(J-1)*(residual.T@residual)/(totals[:,None]*totals[None,:])
        n=totals*totals/sumsq
        ys.append(y);variances.append(D);kish.append(n);boundary.append((y==0)|(y==1))
    y=np.asarray(ys);D=np.asarray(variances);D_raw=D.copy();ne=np.asarray(kish)
    # Same disclosed GVF stabilization is supplied to every FH reconstruction;
    # ICC gets the identical direct y plus sampled weights/PSUs, not oracle D.
    rawdiag=np.diagonal(D,axis1=1,axis2=2).copy();padj=(ne*y+.5)/(ne+1)
    ref=padj*(1-padj)/np.maximum(ne,1)
    ratio=rawdiag/np.maximum(ref,1e-12)
    for t in range(2):
        valid=(rawdiag[:,t]>1e-15)&(y[:,t]>0)&(y[:,t]<1)
        deff=np.median(ratio[valid,t]) if valid.any() else 1.
        ref[:,t]*=max(deff,.05)
        mix=(J-1)/(J-1+4)
        DD=mix*rawdiag[:,t]+(1-mix)*ref[:,t]
        DD=np.where((y[:,t]==0)|(y[:,t]==1),ref[:,t],DD)
        D[:,t,t]=np.maximum(DD,1e-10)
    # Fits are endpoint-independent. Retain raw cross-covariance in D_raw;
    # diagonal working D avoids an indefinite mixed raw/shrunken joint matrix.
    D[:,0,1]=D[:,1,0]=0.
    fit=root/'model_inputs';ev=root/'evaluation_only';fit.mkdir(parents=True);ev.mkdir()
    np.savez_compressed(fit/'inputs.npz',**{k:a[k] for k in ('X_raw','X','strata','coords','W','basis','covariate_names')},y=y,D=D,D_raw=D_raw,n_eff=ne,sample_idx=sampled)
    pd.DataFrame(rows).to_csv(fit/'sample.csv',index=False)
    np.savez_compressed(ev/'truth.npz',theta=np.asarray(truth),sample_idx=sampled)
    save(root/'manifest.json',{'context':context,'replication':rep,'phase':phase,'population_stream':[20261010,701,CONTEXTS.index(context),rep],'sampling_stream':[20261010,702,CONTEXTS.index(context),rep],'areas':m,'population_PSUs_per_area':M,'children_per_PSU':L,'sampled_areas':len(sampled),'sampled_PSUs_per_area':J,'boundary_endpoints':int(np.sum(boundary)),'same_shared_X_y_D_n_psu_for_methods':True,'truth_excluded_from_model_inputs':True,'SAR_probability_clipping':'base risk [0.03,.75], probability SAR [.005,.90]','design_variance':'ultimate-cluster with-replacement Taylor; finite correction omitted, GVF shrink J-1/(J-1+4); boundary GVF, all only training sample','source_sha256':sha(__file__),'fit_inputs_sha256':sha(fit/'inputs.npz'),'sample_sha256':sha(fit/'sample.csv'),'truth_sha256':sha(ev/'truth.npz')})
    return root

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--template',type=Path);p.add_argument('--area-csv',type=Path);p.add_argument('--phase',choices=['development','evaluation']);p.add_argument('--context',choices=CONTEXTS);p.add_argument('--rep',type=int);args=p.parse_args()
    if args.template:print(prepare_template(args.template,args.area_csv))
    else:print(generate(args.context,args.rep,args.phase))

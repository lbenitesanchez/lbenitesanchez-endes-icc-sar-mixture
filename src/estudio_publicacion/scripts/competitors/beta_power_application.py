"""Apply a fixed beta-power candidate to ENDES without retuning test outcomes.

Private individual files stay private. Output consists of district summaries,
working-ICC diagnostics, marginal binary prediction scores and aggregate fold
comparisons. These scores assess prediction of noisy design-weighted outcomes;
ENDES does not reveal true district prevalence or latent interval coverage.
"""
import os,sys,json,importlib.util,time,argparse
from pathlib import Path
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ.setdefault(key,'1')
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from models import spatial_basis
from beta_power import CONFIG
from beta_power_hmc import fit

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('beta_application_area_data',ROOT/'scripts/02_application.py')
app=importlib.util.module_from_spec(spec);spec.loader.exec_module(app)


def prepare_units(year,rule,allareas):
    units=pd.read_csv(ROOT/f'datos/private_competitors/children_{year}.csv',dtype={'ubigeo':str})
    lookup=dict(zip(allareas.ubigeo,np.arange(len(allareas))))
    units['area']=units.ubigeo.map(lookup)
    units=units.loc[units.area.notna()].copy();units['area']=units.area.astype(int)
    units['psu']=units['psu_key']
    units['y_anemia']=units['y_anemia_old' if rule=='old' else 'y_anemia_who2024']
    for name in ['anemia','stunting']:
        units[f'weight_{name}']=np.where(np.isfinite(units[f'y_{name}']),units.weight,np.nan)
    return units


def score(mean,y):
    p=np.clip(mean,1e-12,1-1e-12)
    return -(y*np.log(p)+(1-y)*np.log1p(-p))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--year',type=int,required=True)
    parser.add_argument('--rule',choices=['who2024','old'],default='who2024');parser.add_argument('--tag',required=True)
    parser.add_argument('--q',type=int,default=40);parser.add_argument('--spatial-sd',type=float,default=.5)
    parser.add_argument('--no-cluster',action='store_true');parser.add_argument('--skip-test',action='store_true')
    args=parser.parse_args();out=ROOT/f'resultados/resultados_comparadores/{args.tag}/aplicacion_{args.year}_{args.rule}'
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);allareas,X,y,D,ne,idx,df,unsupported,manifest=app.dataset(args.year,args.rule)
    basis=None if args.q==0 else spatial_basis(allareas[['latitude','longitude']].to_numpy(float),rank=args.q)
    units=prepare_units(args.year,args.rule,allareas)
    checks=[]
    for j,name in enumerate(['anemia','stunting']):
        u=units.loc[np.isfinite(units[f'y_{name}']) & np.isfinite(units[f'weight_{name}'])]
        w=u[f'weight_{name}'];tot=w.groupby(u.area).sum()
        yy=(w*u[f'y_{name}']).groupby(u.area).sum()/tot
        nn=tot**2/(w*w).groupby(u.area).sum()
        delta=float(np.nanmax(abs(yy.reindex(idx).to_numpy()-y[:,j])))
        ndelta=float(np.nanmax(abs(nn.reindex(idx).to_numpy()-ne[:,j])))
        if delta>1e-10 or ndelta>1e-8:raise AssertionError((name,delta,ndelta))
        checks.append(dict(outcome=name,max_direct_difference=delta,max_Kish_n_difference=ndelta))
    config=dict(spatial_sd=args.spatial_sd,cluster_adjustment=not args.no_cluster)
    tick=time.perf_counter();result=fit(X,y,ne,idx,units,basis,seed=20261005+args.year,config=config)
    output=allareas[['ubigeo','prov_code','dept_code']].copy();output['sampled']=np.isin(np.arange(len(allareas)),idx)
    for j,name in enumerate(['anemia','stunting']):
        for key,label in [('p_mean','p_mean'),('lower95','lo95'),('upper95','hi95')]:output[f'{label}_{name}']=result[key][:,j]
        output[f'working_n_{name}']=np.nan;output.loc[idx,f'working_n_{name}']=result['effective_n'][:,j]
    output.to_csv(out/'district_predictions.csv',index=False)
    (out/'fit_status.json').write_text(json.dumps(dict(endpoints=result['status'],seconds=time.perf_counter()-tick),indent=2))
    (out/'data_checks.json').write_text(json.dumps(checks,indent=2))
    if not args.skip_test:
        oldfolder=ROOT/f'resultados/aplicacion_{args.year}_{args.rule}'
        folds=pd.read_csv(oldfolder/'validation_folds.csv',dtype={'ubigeo':str})
        f=folds.set_index('ubigeo').loc[allareas.iloc[idx].ubigeo]
        locked=f.locked_test.to_numpy(bool);tr=np.flatnonzero(~locked);test=np.flatnonzero(locked)
        training_units=units.loc[units.area.isin(idx[tr])]
        fitted=fit(X,y[tr],ne[tr],idx[tr],training_units,basis,seed=20261005+args.year+10000,config=config)
        detail=allareas.iloc[idx[test]][['ubigeo','prov_code']].reset_index(drop=True)
        for j,name in enumerate(['anemia','stunting']):
            detail[f'direct_{name}']=y[test,j];detail[f'pred_{name}']=fitted['p_mean'][idx[test],j]
            detail[f'binary_NLL_{name}']=score(fitted['p_mean'][idx[test],j],y[test,j])
            detail[f'squared_error_{name}']=(fitted['p_mean'][idx[test],j]-y[test,j])**2
        detail.to_csv(out/'province_test_predictions.csv',index=False)
        (out/'test_fit_status.json').write_text(json.dumps(fitted['status'],indent=2))
    (out/'manifest.json').write_text(json.dumps(dict(year=args.year,rule=args.rule,q=args.q,config=result['status'][0]['config'],
        data=manifest,full_endpoints_converged=all(x['success'] for x in result['status']),
        analysis_scope='fixed candidate; reused province folds are exploratory; weighted marginal binary scoring',
        uncertainty='conditional working-ICC, NUTS hyperparameter draws; no latent ENDES coverage claim'),indent=2))
    print(json.dumps(dict(year=args.year,rule=args.rule,q=args.q,status=[x['success'] for x in result['status']],seconds=time.perf_counter()-tick)),flush=True)


if __name__=='__main__':main()

"""External Hogg smoothing diagnostics; no sampler or estimator changes."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit,logit
import hogg_tsln as h


def weighted_slope(x,y,variance):
    valid=np.isfinite(x)&np.isfinite(y)&np.isfinite(variance)&(variance>0)
    if valid.sum()<3:
        return {'slope':None,'intercept':None,'usable_areas':int(valid.sum()),
                'status':'insufficient_finite_positive_variance_areas'}
    x,y,v=x[valid],y[valid],variance[valid]
    w=(1/v);w=w/w.sum()
    mx,my=np.sum(w*x),np.sum(w*y)
    den=np.sum(w*(x-mx)**2)
    if den<=1e-15:
        return {'slope':None,'intercept':None,'usable_areas':int(valid.sum()),
                'status':'no_direct_logit_variation'}
    slope=np.sum(w*(x-mx)*(y-my))/den
    return {'slope':float(slope),'intercept':float(my-slope*mx),
            'usable_areas':int(valid.sum()),'status':'computed'}


def smoothing_diagnostics(prepared,s1):
    direct=prepared['direct']; vd=s1['direct_variance']
    stable=(direct>0)&(direct<1)&np.isfinite(vd)&(vd>0)
    dx=np.full(len(direct),np.nan);dx[stable]=logit(direct[stable])
    theta_median=np.median(s1['theta_draws'],axis=0)
    logit_variance=np.full(len(direct),np.nan)
    logit_variance[stable]=vd[stable]/(direct[stable]*(1-direct[stable]))**2
    paper=weighted_slope(dx,theta_median,vd)
    author=weighted_slope(dx,theta_median,logit_variance)
    ratio=np.nanmedian(s1['psi_draws'][:,stable]/vd[stable]) if stable.any() else None
    return {'ALC_paper_weights_inverse_direct_proportion_variance':paper,
            'ALC_author_step4_weights_inverse_direct_logit_variance':author,
            'median_stage1_sampling_variance_to_direct_variance_ratio_stable':ratio,
            'stage1_summary':'posterior median of logit(S1 proportion)',
            'direct_summary':'ordinary logit, restricted to direct0<p<1 and finite positive variance; no boundary perturbation',
            'author_exploratory_range':[.55,.75],
            'range_interpretation':'Empirical exploratory suggestion from Hogg Supporting InformationE; no theoretical threshold, no coverage guarantee, no outcome-based model tuning.',
            'source_text':'Hogg2024 section2.2.3; arXivv3 text lines549/552',
            'source_code':'Author simulation_study/r_src/step4.R uses logit variance weights, which differ from the proportion-variance weights in the text.'}


def age_sex_features(units):
    """Six fixed age bands and sex; defined without inspecting outcomes."""
    u=units.copy();age=u.age_months.to_numpy(float)
    if not np.isfinite(age).all() or ((age<6)|(age>=60)).any():
        raise ValueError('Age/sex sensitivity requires ages6–59 months')
    bands=[(6,12),(12,18),(18,24),(24,36),(36,48),(48,60)]
    features=[]
    for low,upper in bands[1:]:
        name=f'age_{low}_{upper-1}'
        u[name]=((age>=low)&(age<upper)).astype(int)
        features.append(name)
    if not u.sex.isin([1,2]).all():raise ValueError('Sex coding requires1male/2female')
    u['sex_male']=(u.sex==1).astype(int);features.append('sex_male')
    return u,features,{'age_bands_months':[[low,upper-1] for low,upper in bands],
        'age_reference':'6–11months','sex_reference':'female','sex_covariate':'male',
        'budget':'higher_information_survey_only_covariates_stage1; stage2 keeps six district auxiliaries',
        'selection':'prespecified by parent before sensitivity fitting; no outcome screening'}


def annotate_pair(result,units,areas,*,outcomes=('y_anemia','y_stunting'),
                  weights=('weight_anemia','weight_stunting'),area_col='area',
                  area_features=None,unit_features=(),population_N=None):
    diag={}
    for j,label in enumerate(['anemia','stunting']):
        prepared=h.prepare_hogg_data(units,areas,outcome=outcomes[j],weight=weights[j],
            area_col=area_col,area_features=area_features,unit_features=unit_features,
            population_N=population_N)
        diag[label]=smoothing_diagnostics(prepared,result['results'][label].stage1)
    target=result['results']['anemia'].output_dir.parents[1]
    h.write_json(target/'smoothing_diagnostics_ALC.json',diag)
    return diag


def saved_area_only_ALC(run_dir,units,areas,*,outcome,weight,area_col='area',
                        area_features=None,population_N=None):
    """Add ALC to completed baseline fits without rerunning NUTS."""
    run_dir=Path(run_dir)
    prepared=h.prepare_hogg_data(units,areas,outcome=outcome,weight=weight,
        area_col=area_col,area_features=area_features,population_N=population_N)
    groups=prepared['s1_data'];m=len(prepared['n'])
    if groups['G']!=m or not np.array_equal(groups['area'],np.arange(m)+1):
        raise ValueError('Saved fast ALC path is explicitly area-covariate-only')
    # CmdStan CSV flattens vector indices with dots, unlike summary row labels.
    names=[f'eta.{i+1}' for i in range(m)]
    traces=[pd.read_csv(x,comment='#',usecols=names).to_numpy(float)
            for x in sorted((run_dir/'stage1').glob('*.csv'))]
    if len(traces)!=4:raise ValueError('Require four retained chain CSVs')
    theta=np.concatenate(traces,axis=0)
    summaries=pd.read_csv(run_dir/'stage1_area_summary.csv',dtype={'area':str})
    if not np.array_equal(summaries.area.to_numpy(),prepared['sampled_codes']):
        raise ValueError('Saved stage1 domain ordering mismatch')
    vd=summaries.direct_variance_Hogg.to_numpy(float)
    psi=np.broadcast_to(2*vd,(len(theta),m))
    diag=smoothing_diagnostics(prepared,{'theta_draws':theta,
        'direct_variance':vd,'psi_draws':psi})
    diag['saved_trace_precision']='ALC uses saved CmdStan eta trace, six significant figures; exact constant-within-area logit identity. No sampler refit.'
    h.write_json(run_dir/'smoothing_diagnostics_ALC.json',diag)
    return diag

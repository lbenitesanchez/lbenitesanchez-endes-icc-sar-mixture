"""Common locked-area marginal binary forecast score, with no Gaussian D score.

District survey ratio pDirect is a noisy heldout assessment, never known truth.
This scores posterior-mean Bernoulli forecasts, not interval calibration or a
joint binary outcome distribution. All methods use the same heldout districts.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.special import xlogy
import hogg_tsln as h


def score_year(year,files,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    heldout=h.ROOT/f'resultados/aplicacion_{year}_who2024/locked_test_area_scores.csv'
    legacy=pd.read_csv(heldout,dtype={'ubigeo':str,'province':str})
    target=legacy[['ubigeo','province','direct_anemia','direct_stunting']].drop_duplicates()
    if target.ubigeo.duplicated().any():raise ValueError('Heldout direct is inconsistent across methods')
    methods={}
    for method,frame in legacy.groupby('method'):
        methods[method]=frame[['ubigeo','pred_anemia','pred_stunting']]
    for label,path in files.items():
        frame=pd.read_csv(path,dtype={'ubigeo':str,'area':str})
        if 'ubigeo'not in frame:frame=frame.rename(columns={'area':'ubigeo'})
        frame=frame.rename(columns={'p_mean_anemia':'pred_anemia','p_mean_stunting':'pred_stunting'})
        methods[label]=frame[['ubigeo','pred_anemia','pred_stunting']]
    rows=[];summaries=[]
    for label,frame in methods.items():
        if frame.ubigeo.duplicated().any():raise ValueError(f'Duplicate prediction area for{label}')
        merged=target.merge(frame,on='ubigeo',how='left',validate='one_to_one')
        finite=np.isfinite(merged[['pred_anemia','pred_stunting']].to_numpy()).all(axis=1)
        if not finite.all():
            summaries.append({'year':year,'method':label,'requested_areas':len(target),
                'areas_with_finite_predictions':int(finite.sum()),'complete_comparison':False})
            continue  # Never shrink the primary comparison to an easier subset.
        for outcome in ['anemia','stunting']:
            direct=merged['direct_'+outcome].to_numpy(float)
            q=merged['pred_'+outcome].to_numpy(float)
            if ((q<0)|(q>1)).any():raise ValueError('Predicted probability outside[0,1]')
            merged['negative_log_score_'+outcome]=-(xlogy(direct,q)+xlogy(1-direct,1-q))
            merged['squared_error_'+outcome]=(q-direct)**2
            merged['absolute_error_'+outcome]=abs(q-direct)
        merged['negative_log_score_balanced']=merged[['negative_log_score_anemia','negative_log_score_stunting']].mean(axis=1)
        merged['year']=year;merged['method']=label
        rows.append(merged)
        summaries.append({'year':year,'method':label,'requested_areas':len(target),
            'areas_with_finite_predictions':len(target),'complete_comparison':True,
            'negative_log_score_balanced':float(merged.negative_log_score_balanced.mean()),
            **{f'negative_log_score_{out}':float(merged['negative_log_score_'+out].mean())for out in ['anemia','stunting']},
            **{f'RMSE_against_noisy_direct_{out}':float(np.sqrt(merged['squared_error_'+out].mean()))for out in ['anemia','stunting']},
            **{f'MAE_against_noisy_direct_{out}':float(merged['absolute_error_'+out].mean())for out in ['anemia','stunting']}})
    table=pd.DataFrame(summaries);table.to_csv(output/'locked_marginal_binary_score.csv',index=False)
    if rows:pd.concat(rows).to_csv(output/'locked_area_forecast_scores.csv',index=False)
    metadata={'year':year,'heldout_definition_sha256':hashlib.sha256(heldout.read_bytes()).hexdigest(),
        'prediction_files_sha256':{label:hashlib.sha256(Path(path).read_bytes()).hexdigest()for label,path in files.items()},
        'heldout_districts':len(target),'heldout_provinces':target.province.nunique(),
        'score':'-[pDirect*log(pPosteriorMean)+(1-pDirect)*log(1-pPosteriorMean)] for eachendpoint, equalendpoint/equaldistrict mean',
        'no_prediction_clipping':True,'boundary_forecasts':'Exact0/1forecasts receive infinite score for conflicting directmass, using xlogy0*log0=0 convention.',
        'interpretation':'Empirical noisy surveyweighted marginal Bernoulli forecast score; no known districttruth, no intervalcalibration, no jointoutcomedependence score.',
        'common_target':'Every method must predict all previouslylocked areas; incomplete methods markedunavailable without selective dropping.',
        'reused_holdout':'These provinces were previously used for original diagnostic comparisons; this is a common reanalysis, not a newly pristine independent test.',
        'intervals':'Not evaluated ascoverage against survey directs.'}
    h.write_json(output/'score_definition.json',metadata)
    return table


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--year',type=int,required=True)
    parser.add_argument('--predictions',nargs='*',default=[],help='LABEL=/absolute/predictions.csv')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    files=dict(item.split('=',1)for item in args.predictions)
    print(score_year(args.year,files,args.output).to_string(index=False))


if __name__=='__main__':main()

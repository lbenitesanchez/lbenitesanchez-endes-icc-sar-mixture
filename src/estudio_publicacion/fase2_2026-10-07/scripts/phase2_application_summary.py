"""Compare completed2025 map summaries; no in-sample performance inference."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    paths=[('BetaPowerExisting','application2025_baseline','predictions.csv','status.json'),
           ('BetaDesignUncertainty','application2025_designcut8','district_predictions.csv','completion.json')]
    summaries=[];files=[]
    for method,folder,predfile,statusfile in paths:
        target=ROOT/'resultados'/folder
        status=json.loads((target/statusfile).read_text())
        if not status['success']:raise RuntimeError('Numerical application gate failed')
        d=pd.read_csv(target/predfile,dtype={'ubigeo':str})
        if len(d)!=1874 or d.ubigeo.nunique()!=1874:raise RuntimeError('Historical support incomplete')
        files.append({'method':method,'path':str((target/predfile).relative_to(ROOT)),'sha256':sha(target/predfile),
                      'status_sha256':sha(target/statusfile)})
        for outcome in ('anemia','stunting'):
            mean=d['p_mean_'+outcome].to_numpy(float)
            width=(d['upper95_'+outcome]-d['lower95_'+outcome]).to_numpy(float)
            var=d['p_variance_'+outcome].to_numpy(float)
            between=d['between_state_variance_'+outcome].to_numpy(float)if method=='BetaDesignUncertainty'else np.zeros(len(d))
            within=d['within_state_variance_'+outcome].to_numpy(float)if method=='BetaDesignUncertainty'else var
            if not np.allclose(var,within+between,atol=1e-12,rtol=1e-10):raise RuntimeError('Variance decomposition failed')
            if not np.isfinite(mean).all()or (width<0).any():raise RuntimeError('Invalid map values')
            summaries.append({'method':method,'outcome':outcome,'districts':len(d),'mean_prevalence':mean.mean(),
                'mean_width':width.mean(),'median_width':np.median(width),'mean_between_share':np.mean(between/var),
                'max_between_share':np.max(between/var),'scope':'equal district summaries, not national prevalence or heldout accuracy'})
    out=ROOT/'resultados/application2025_summary.csv'
    pd.DataFrame(summaries).to_csv(out,index=False)
    proof={'status':'PASS_COMPLETED_SECONDARY2025_SUMMARY','files':files,'summary_sha256':sha(out),
           'source_sha256':sha(__file__),'predictive_accuracy_claim':False,'national_prevalence_claim':False}
    (ROOT/'validacion/application2025_summary_validation.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(proof),flush=True)
if __name__=='__main__':main()

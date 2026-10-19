import csv,json,os,subprocess,time
from pathlib import Path
import numpy as np
from sikov_model import fit_sikov,knn_weights
from sikov_bootstrap import fit_sikov_bootstrap

ROOT=Path(__file__).resolve().parents[1]
DIR=ROOT/'validacion/sikov_bootstrap_reference';INPUT=DIR/'inputs';OUTPUT=DIR/'outputs'
INPUT.mkdir(parents=True,exist_ok=True);OUTPUT.mkdir(parents=True,exist_ok=True)
start=time.monotonic();rng=np.random.default_rng(741690);n=100;B=12
coords=rng.normal(size=(n,2));x=rng.uniform(0,1,n);X=np.c_[np.ones(n),x];W=knn_weights(coords,4);D=rng.uniform(.0003,.0006,n)
u=np.linalg.solve(np.eye(n)-.60*W,rng.normal(0,np.sqrt(.004),n));y=.40+.06*x+u+rng.normal(0,np.sqrt(D))
initial=fit_sikov(X,y,D,np.arange(n),W);par=initial['fit'][0]
U=rng.normal(size=(B,n));E=rng.normal(size=(B,n))
f=fit_sikov_bootstrap(X,y,D,np.arange(n),W,B=B,innovations={1:{'u':U,'e':E}})
np.savetxt(INPUT/'data.csv',np.c_[y,x,D],delimiter=',',header='y,x,D',comments='')
np.savetxt(INPUT/'W.csv',W,delimiter=',');np.savetxt(INPUT/'u.csv',U,delimiter=',');np.savetxt(INPUT/'e.csv',E,delimiter=',')
np.savetxt(INPUT/'params.csv',np.array([[par['variance'],par['rho'],*par['beta']]]),delimiter=',',header='A,rho,beta0,beta1',comments='')
env=os.environ.copy();env['SIKOV_SAE_LIBRARY']=str(ROOT/'fuentes/sae_library')
subprocess.run(['Rscript',str(ROOT/'scripts/sikov_bootstrap_reference.R'),str(INPUT),str(OUTPUT)],env=env,check=True)
with (OUTPUT/'mse.csv').open() as s:ref=list(csv.DictReader(s))
checks=[]
for nk,rk in [('mse_naive','naive'),('mse_bc','bc'),('mse_plugin','plugin')]:
 error=float(np.max(np.abs(f[nk]-np.array([float(r[rk]) for r in ref]))))
 checks.append({'check':nk,'max_abs_error':error,'passed':error<2e-7})
checks.append({'check':'unchanged_point_estimates','max_abs_error':float(abs(f['mean']-initial['mean']).max()),'passed':np.allclose(f['mean'],initial['mean'],atol=1e-12)})
fh_initial=fit_sikov(X,y,D,np.arange(n),W,spatial=False);fhpar=fh_initial['fit'][0]
fh=fit_sikov_bootstrap(X,y,D,np.arange(n),W,spatial=False,B=B,innovations={1:{'u':U,'e':E}})
np.savetxt(INPUT/'params.csv',np.array([[fhpar['variance'],0.,*fhpar['beta']]]),delimiter=',',header='A,rho,beta0,beta1',comments='')
subprocess.run(['Rscript',str(ROOT/'scripts/sikov_bootstrap_reference.R'),str(INPUT),str(OUTPUT/'FH'),'FH'],env=env,check=True)
with (OUTPUT/'FH/mse.csv').open() as s:ref=list(csv.DictReader(s))
for nk,rk in [('mse_naive','naive'),('mse_bc','bc'),('mse_plugin','plugin')]:
 error=float(np.max(np.abs(fh[nk]-np.array([float(r[rk]) for r in ref]))))
 checks.append({'check':'FH_'+nk,'max_abs_error':error,'passed':error<2e-7})
result={'passed':all(c['passed'] for c in checks),'checks':checks,'B':B,'areas':n,'identical_innovations':True,'independent_ref':'CRAN sae 1.3 eblupSFH predictors and pbmseSFH BC formula','elapsed_seconds':time.monotonic()-start,'note':'B12 is a numerical cross-language unit check, not scientific bootstrap B100 precision.'}
(ROOT/'validacion/sikov_bootstrap_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
if not result['passed']:raise RuntimeError('bootstrap reference mismatch')

"""Exact finite binomial diagnostic; no fitting or ENDES response tuning."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.pydeps'))
import numpy as np,pandas as pd
from scipy.stats import binom
from scipy.special import logit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
records=[]
for p in [.05,.10,.15,.30,.50]:
 for n in [1,2,5,9,10,20,50,100]:
  k=np.arange(n+1);prob=binom.pmf(k,n,p);adj=(k+.5)/(n+1)
  records.append({'n':n,'p':p,'mean_adjusted_p':float(prob@adj),'bias_adjusted_p':float(prob@adj-p),'mean_adjusted_logit':float(prob@logit(adj)),'bias_adjusted_logit':float(prob@logit(adj)-logit(p)),'analytic_proportion_bias':(.5-p)/(n+1)})
d=pd.DataFrame(records);valid=ROOT/'validacion';valid.mkdir(exist_ok=True)
d.to_csv(valid/'continuity_mean_bias_exact_binomial.csv',index=False)
err=float(np.max(abs(d.bias_adjusted_p-d.analytic_proportion_bias)))
if err>1e-12:raise AssertionError('Exact binomial mean identity failed')
(valid/'continuity_mean_bias_validation.json').write_text(json.dumps({'max_proportion_identity_residual':err,'passed':True,'meaning':'Exact ordinary-binomial enumeration, not a weighted-survey model or isolated explanation of area-model bias'},indent=2))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(7.2,3.3),layout='constrained')
for p,color in [(0.05,'#5B6F91'),(.10,'#AB613A'),(.15,'#42856E'),(.30,'#826596')]:
 a=d[d.p==p];ax.plot(a.n,a.bias_adjusted_logit,'o-',color=color,label=f'p = {p:.2f}',markersize=3.5)
ax.axhline(0,color='black',lw=.7);ax.set_xscale('log');ax.set_xticks([1,2,5,10,20,50,100],[1,2,5,10,20,50,100])
ax.set_xlabel('Fixed binomial sample size n');ax.set_ylabel('E[logit(adjusted proportion)] − logit(p)');ax.legend(frameon=False,ncol=4)
figdir=ROOT/'figuras';figdir.mkdir(exist_ok=True)
fig.savefig(figdir/'continuity_mean_bias.png',dpi=300);fig.savefig(figdir/'continuity_mean_bias.svg');plt.close(fig)
print('Exact binomial diagnostic passed, residual',err)

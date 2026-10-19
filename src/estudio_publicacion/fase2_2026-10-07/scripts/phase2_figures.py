"""Publication plots of completed results; no fitting or held-out tuning."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'figuras'
LABEL={'baseline':'ICC beta','point':'Variance point','cut8':'Variance CUT8',
       'BetaPowerExisting':'ICC beta','HoggAgeSexSpatial':'Hogg age/sex BYM2'}
COLOUR={'baseline':'#1b6c8e','point':'#777777','cut8':'#ac4e23'}
plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','axes.spines.top':False,
                     'axes.spines.right':False,'svg.fonttype':'none'})

def save(fig,name):
    OUT.mkdir(exist_ok=True)
    for ext in ('png','svg'):
        fig.savefig(OUT/f'{name}.{ext}',dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def simulations():
    src=ROOT/'resultados/design_simulation/final'
    if not json.loads((src/'completion.json').read_text())['all40_completed']:raise ValueError('All40 required')
    d=pd.read_csv(src/'summary.csv')
    fig,axes=plt.subplots(2,2,figsize=(10,6.3),layout='constrained')
    for row,outcome in enumerate(('anemia','stunting')):
        for col,(metric,scale,title)in enumerate((('coverage95',100,'95% interval coverage (%)'),
                                                ('interval_score95',1,'95% interval score'))):
            ax=axes[row,col]
            for k,method in enumerate(('baseline','point','cut8')):
                s=d[(d.outcome==outcome)&(d.method==method)].set_index('context').loc[['dense6','sparse2']]
                ax.errorbar(np.arange(2)+(k-1)*.18,s[metric]*scale,
                            yerr=s[metric+'_MCSE']*2*scale,color=COLOUR[method],marker='o',
                            markersize=5,linestyle='none',capsize=3,label=LABEL[method])
            ax.set_xticks([0,1],['6 sampled PSU','2 sampled PSU'])
            ax.set_xlim(-.45,1.45);ax.set_ylabel(title)
            ax.set_title(('Anaemia'if outcome=='anemia'else'Stunting')+' — 20 populations per context')
            ax.grid(axis='y',alpha=.18)
            if col==0:ax.axhline(95,color='#333333',lw=.9,ls='--')
    axes[0,1].legend(frameon=False,loc='upper left')
    save(fig,'design_diagnostics')

def external():
    src=ROOT/'resultados/externa_final'
    completion=json.loads((src/'completion.json').read_text())
    if completion['status']!='COMPLETE_RESERVED_EXTERNAL_VALIDATION':raise ValueError('External incomplete')
    d=pd.read_csv(src/'contrasts_Holm8.csv')
    if len(d)!=8:raise ValueError('All8 required')
    fig,axes=plt.subplots(1,2,figsize=(10,4.4),layout='constrained')
    for ax,scope in zip(axes,('temporal','territorial')):
        subset=d[d.scope==scope]
        names=[]
        for i,r in enumerate(subset.itertuples()):
            sig=bool(r.candidate_significantly_better or r.candidate_significantly_worse)
            colour='#ac4e23'if r.difference>0 else'#1b6c8e'
            ax.plot([r.unadjusted_t_ci_low,r.unadjusted_t_ci_high],[i,i],color=colour,lw=1.4)
            ax.scatter([r.difference],[i],s=42,facecolors=colour if sig else'white',edgecolors=colour,zorder=3)
            names.append(('Anaemia'if r.outcome=='anemia'else'Stunting')+'\nvs '+LABEL[r.reference])
        ax.set_yticks(np.arange(len(names)),names);ax.invert_yaxis()
        ax.axvline(0,color='#333333',lw=.9,ls='--');ax.grid(axis='x',alpha=.18)
        ax.set_title('2019 → 2024 transport'if scope=='temporal'else'2024 reserved provinces')
        ax.set_xlabel('CUT8 minus comparator: binary log score')
    save(fig,'external_contrasts')

def application_map(method):
    src=ROOT/('resultados/application2025_baseline' if method=='baseline' else 'resultados/application2025_designcut8')
    status='status.json' if method=='baseline' else 'completion.json'
    filename='predictions.csv' if method=='baseline' else 'district_predictions.csv'
    if not json.loads((src/status).read_text())['success']:raise ValueError('Application incomplete')
    d=pd.read_csv(src/filename,dtype={'ubigeo':str})
    a=pd.read_csv(ROOT/'datos/geografia/aux_verified_2017_2018.csv',dtype={'ubigeo':str})
    d=d.merge(a[['ubigeo','latitude','longitude']],on='ubigeo',validate='one_to_one')
    if len(d)!=1874:raise ValueError('Full historical support required')
    fig,axes=plt.subplots(2,2,figsize=(8.8,10),layout='constrained')
    for row,outcome in enumerate(('anemia','stunting')):
        for col,kind in enumerate(('mean','width')):
            ax=axes[row,col]
            value=100*(d['p_mean_'+outcome]if kind=='mean'else d['upper95_'+outcome]-d['lower95_'+outcome])
            sc=ax.scatter(d.longitude,d.latitude,c=value,s=4,cmap='viridis'if kind=='mean'else'magma',
                          linewidths=0,rasterized=True)
            ax.set_aspect(1/np.cos(np.deg2rad(-10)))
            ax.set_xlabel('Longitude');ax.set_ylabel('Latitude')
            ax.set_title(('Anaemia'if outcome=='anemia'else'Stunting')+' — '+('mean'if kind=='mean'else'95% interval width'))
            fig.colorbar(sc,ax=ax,shrink=.8,label='Percent'if kind=='mean'else'Percentage points')
    fig.suptitle(LABEL[method]+' — ENDES 2025 district application',fontsize=12)
    save(fig,'application2025_ICC_capitals' if method=='baseline' else 'application2025_capitals')

def application():
    for method in ('baseline','cut8'):application_map(method)

def main():
    p=argparse.ArgumentParser();p.add_argument('--only',choices=['simulation','external','application','all'],default='all')
    arg=p.parse_args()
    for key,function in [('simulation',simulations),('external',external),('application',application)]:
        if arg.only in (key,'all'):function()
    print(json.dumps({'status':'FIGURES_CREATED_VISUAL_INSPECTION_PENDING','scope':arg.only}))

if __name__=='__main__':main()

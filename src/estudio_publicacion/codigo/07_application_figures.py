"""Publication figures and aggregate decision tables for corrected applications."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

FIG=ROOT/'figuras';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300})
def read(path):return pd.read_csv(path,dtype={'ubigeo':str,'prov_code':str,'dept_code':str})
def save(fig,name):
    fig.savefig(FIG/(name+'.png'),dpi=300)
    fig.savefig(FIG/(name+'.svg'))
    plt.close(fig)

def support():
    fig,axes=plt.subplots(1,2,figsize=(10.4,3.7),layout='constrained')
    for year,col in ((2019,'#536B8C'),(2025,'#AE6036')):
        d=read(ROOT/f'datos/processed/areas_{year}_6_59.csv')
        x=np.sort(d.n_anemia_who2024.to_numpy())
        axes[0].step(x,np.arange(1,len(x)+1)/len(x),where='post',label=str(year),color=col)
    axes[0].set_xscale('log');axes[0].set_xlim(1,500)
    axes[0].set_xlabel('Measured children per district for haemoglobin')
    axes[0].set_ylabel('Cumulative fraction of observed districts')
    axes[0].yaxis.set_major_formatter(PercentFormatter(1));axes[0].legend(frameon=False)
    quality=json.loads((ROOT/'datos/processed/data_quality_checks.json').read_text())
    labels=['One PSU','Zero anaemia\nvariance','Zero stunting\nvariance','Singular\njoint matrix']
    for k,year in enumerate((2019,2025)):
        q=next(x for x in quality['domain_checks'] if x['year']==year and x['rule']=='who')
        vals=[q['one_psu_domain'],q['zero_variance_a'],q['zero_variance_s'],q['raw_singular_D']]
        axes[1].bar(np.arange(4)+(k-.5)*.34,np.array(vals)/q['areas'],width=.34,label=str(year),color=['#536B8C','#AE6036'][k])
    axes[1].set_xticks(range(4),labels);axes[1].set_ylim(0,.85)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_ylabel('Fraction of observed districts');axes[1].legend(frameon=False)
    save(fig,'district_support')

def application():
    now=ROOT/'resultados/aplicacion_2025_who2024'
    old=ROOT/'resultados/aplicacion_2025_old'
    for f in (now,old):
        if not (f/'completed.json').exists():raise RuntimeError(f'Application incomplete: {f}')
    n=read(now/'district_predictions_SELECTED.csv');o=read(old/'district_predictions_SELECTED.csv')
    if n.method.nunique()!=1 or o.method.nunique()!=1 or n.method.iloc[0]!=o.method.iloc[0]:
        raise ValueError('Definition comparison must retain the same model family')
    pair=n[['ubigeo','department','province','district','latitude','longitude','sampled_any','p_anemia','p_stunting','prob_both_area_marginal_thresholds_plugin']].merge(o[['ubigeo','p_anemia','p_stunting','prob_both_area_marginal_thresholds_plugin']],on='ubigeo',validate='one_to_one',suffixes=('_new','_old'))
    pair['point_priority_old']=(pair.p_anemia_old>.30)&(pair.p_stunting_old>.15)
    pair['point_priority_new']=(pair.p_anemia_new>.30)&(pair.p_stunting_new>.15)
    pair['priority_change']=np.select([pair.point_priority_old&~pair.point_priority_new,~pair.point_priority_old&pair.point_priority_new],['leaves','enters'],default='unchanged')
    pair['conditional_high_probability_new']=pair.prob_both_area_marginal_thresholds_plugin_new>=.8
    pair.to_csv(now/'definition_comparison_descriptive.csv',index=False)
    trans=pd.crosstab(pair.point_priority_old,pair.point_priority_new,dropna=False)
    trans.to_csv(now/'definition_priority_transition.csv')
    width_a=n.p_anemia_hi95_bootstrap_conditional_D-n.p_anemia_lo95_bootstrap_conditional_D
    width_s=n.p_stunting_hi95_bootstrap_conditional_D-n.p_stunting_lo95_bootstrap_conditional_D
    detail={
      'districts':len(pair),'method':n.method.iloc[0],
      'point_priority_old':int(pair.point_priority_old.sum()),'point_priority_new':int(pair.point_priority_new.sum()),
      'leaves_point_priority':int((pair.priority_change=='leaves').sum()),'enters_point_priority':int((pair.priority_change=='enters').sum()),
      'conditional_probability_ge80_new':int(pair.conditional_high_probability_new.sum()),
      'definition_change_mean_anemia':float((pair.p_anemia_new-pair.p_anemia_old).mean()),
      'median_bootstrap_width_anemia':float(width_a.median()),'median_bootstrap_width_stunting':float(width_s.median()),
      'caveat':'Point reclassification is descriptive. Exceedance probabilities are conditional working-model quantities, not calibrated programme eligibility or probability of individual co-occurrence.'}
    (now/'application_decision_summary.json').write_text(json.dumps(detail,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.0),layout='constrained')
    col=pair.priority_change.map({'unchanged':'#78858D','leaves':'#A05735','enters':'#337B5E'})
    for group,label,c in [('unchanged','No priority change','#78858D'),('leaves','Leaves point priority','#A05735'),('enters','Enters point priority','#337B5E')]:
        b=pair[pair.priority_change==group]
        axes[0].scatter(b.p_anemia_old*100,b.p_anemia_new*100,s=8 if group=='unchanged' else 17,alpha=.45 if group=='unchanged' else .8,label=label,color=c,rasterized=True)
    lim=max(pair.p_anemia_old.max(),pair.p_anemia_new.max())*100+3
    axes[0].plot([0,lim],[0,lim],color='black',lw=.75)
    axes[0].axhline(30,color='black',ls=':',lw=.8);axes[0].axvline(30,color='black',ls=':',lw=.8)
    axes[0].set_xlim(0,lim);axes[0].set_ylim(0,lim)
    axes[0].set_xlabel('Historical-rule estimated anaemia (%)')
    axes[0].set_ylabel('WHO 2024 estimated anaemia (%)')
    axes[0].legend(frameon=False,fontsize=7.5,loc='lower right')
    for series,label,c in ((width_a,'Anaemia','#536B8C'),(width_s,'Stunting','#AE6036')):
        x=np.sort(series.to_numpy()*100)
        axes[1].step(x,np.arange(1,len(x)+1)/len(x),label=label+' · conditional D',color=c)
    stress=read(now/'modelled_D_uncertainty_sensitivity.csv')
    for variable,label,c in [('anemia','Anaemia','#536B8C'),('stunting','Stunting','#AE6036')]:
        width=stress['p_'+variable+'_hi95_modelled_D_sensitivity']-stress['p_'+variable+'_lo95_modelled_D_sensitivity']
        x=np.sort(width.to_numpy()*100)
        axes[1].step(x,np.arange(1,len(x)+1)/len(x),label=label+' · modelled-D stress',color=c,ls='--')
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_xlabel('95% band width (percentage points)')
    axes[1].set_ylabel('Cumulative fraction of districts');axes[1].legend(frameon=False,fontsize=7.5)
    save(fig,'definition_and_uncertainty')
    # Centroid point map preserves the distinction from current district polygons.
    fig,axes=plt.subplots(1,2,figsize=(7.5,7.4),layout='constrained')
    for ax,variable,title in zip(axes,['p_anemia','p_stunting'],['Anaemia','Stunting']):
        im=ax.scatter(n.longitude,n.latitude,c=n[variable]*100,cmap='viridis',s=9,vmin=0,vmax=60 if variable=='p_anemia' else 35,rasterized=True)
        ax.set_xlabel('Longitude');ax.set_ylabel('Latitude');ax.set_aspect(1.02)
        ax.set_title(title);fig.colorbar(im,ax=ax,label='Estimated prevalence (%)',shrink=.6)
    save(fig,'district_centroid_estimates')

if __name__=='__main__':
    support()
    application()

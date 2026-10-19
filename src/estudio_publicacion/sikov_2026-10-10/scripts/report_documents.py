"""Scientific table-formatting helpers for the ENDES scoring adapters.

This code-only extraction keeps the original numerical operations and labels.
It excludes narrative interpretation templates, manuscript/document assembly,
Word/XML APIs, bibliography, title pages and cover letters. Extraction and
parity checks are recorded in provenance/code_only_formatter.json.
"""
from __future__ import annotations

CONTEXTS = ('SAR_probability_srs', 'SAR_logit_srs', 'SAR_logit_psu_sparse', 'IID_logit_psu_sparse')


ENDPOINTS = ('anemia', 'stunting')


METHODS = ('Sikov_FH', 'Sikov_SAR', 'Sikov_SAR_Published', 'ICC', 'Selected')


REFERENCES = ('Sikov_FH', 'Sikov_SAR', 'Sikov_SAR_Published')


LABELS = {'Sikov_FH':'Sikov FH','Sikov_SAR':'SAR full-N','Sikov_SAR_Published':'SAR published-scope',
          'ICC':'ICC beta','Selected':'Selected mixture',
          'Sikov_FH_BC':'FH BC','Sikov_SAR_BC':'SAR full-N BC','Sikov_SAR_Published_BC':'SAR published BC'}


def aggregate(data, outcome, method):
    d = data['summary']; p = d.loc[d.group.eq('all') & d.outcome.eq(outcome) & d.method.eq(method)]
    if len(p) != 4: raise ValueError('Equal four-context aggregate incomplete')
    return {'rmse': (p.rmse.pow(2).mean())**.5,
            **{key:p[key].mean() for key in ('coverage','width','interval_score')}}


def f(value, decimals=3):
    return f'{float(value):.{decimals}f}'


def sci(value):
    return f'{float(value):.3g}'


def context_rows(data, *, include_icc=False):
    values=[]
    for c in CONTEXTS:
        for e in ENDPOINTS:
            for m in (METHODS if include_icc else ('Sikov_SAR','Sikov_SAR_Published','Selected')):
                p=data['summary'].loc[(data['summary'].scenario==c)&(data['summary'].outcome==e)&
                    (data['summary'].method==m)&(data['summary'].group=='all')].iloc[0]
                values.append([str(CONTEXTS.index(c)+1), 'Anemia' if e=='anemia' else 'Talla',
                    LABELS[m].replace('Selected mixture','Mezcla').replace('SAR published-scope','SAR observado'),f(100*p.rmse,2),
                    f(100*p.coverage,1),f(100*p.width,2),f(p.interval_score,3)])
    return values


def aggregate_rows(data, spanish=False):
    values=[]
    for e in ENDPOINTS:
        for m in METHODS:
            row=aggregate(data,e,m)
            label=LABELS[m]
            if spanish: label=label.replace('Selected mixture','Mezcla seleccionada')
            values.append([label,('Anemia' if e=='anemia' else 'Talla') if spanish else e,
                          f(100*row['rmse'],2),f(100*row['coverage'],1),f(100*row['width'],2),f(row['interval_score'],3)])
    return values


def BC_rows(data, spanish=False):
    rows=[]
    for endpoint in ENDPOINTS:
        for method in ('Sikov_FH_BC','Sikov_SAR_BC','Sikov_SAR_Published_BC'):
            p=aggregate(data,endpoint,method)
            outcome = ('Anemia' if endpoint == 'anemia' else 'Talla') if spanish else endpoint
            rows.append([LABELS[method],outcome,
                         f(p['interval_score'],3),f(100*p['coverage'],1),f(100*p['width'],2)])
    return rows


def B300_rows(data):
    names={'SAR_B100':'SAR full-N B100','SAR_B300':'SAR full-N B300',
           'Selected_B100':'Mix B100','Selected_B300':'Mix B300'}
    part=data['B300_summary'];part=part[part.context.eq('equal_four_contexts')]
    if len(part)!=4 or set(part.method)!=set(names):raise RuntimeError('Incomplete B300 descriptive table')
    rows=[[names[r.method],f(r.interval_score,3),f(100*r.coverage,1),f(100*r.width,2)]
          for r in part.itertuples()]
    extra=data['published_B300_summary']
    extra=extra[extra.scenario.eq('context_balanced')&extra.outcome.eq('endpoint_balanced')&extra.group.eq('all')]
    indexed=extra.set_index(['method','metric'])
    labels={'Published_B100':'Observed SAR B100','Published_B100_BC':'Observed SAR B100 BC',
            'Published_B300':'Observed SAR B300','Published_B300_BC':'Observed SAR B300 BC'}
    if len(extra)!=35 or extra[['method','metric']].duplicated().any():raise RuntimeError('Complete Published B300 descriptive metrics required')
    for method,label in labels.items():
        rows.append([label,f(indexed.loc[method,'interval_score'].estimate,3),
                     f(100*indexed.loc[method,'coverage'].estimate,1),f(100*indexed.loc[method,'width'].estimate,2)])
    return rows


def poverty_rows(data,spanish=False):
    """All three strata, references and masks; never choose favourable cells."""
    part=data['poverty_differences']
    part=part.loc[part.scenario.eq('context_balanced')&part.outcome.eq('endpoint_balanced')
                  &part.metric.eq('interval_score')]
    groups=('all','observed_endpoint','unsampled_common')
    expected={(stratum,reference,group)for stratum in (1,2,3)for reference in REFERENCES for group in groups}
    keys=['poverty_stratum','reference','group']
    if len(part)!=27 or part.duplicated(keys).any()or set(map(tuple,part[keys].to_numpy()))!=expected:
        raise ValueError('Complete three-stratum/reference/group descriptive table required')
    labels={'Sikov_FH':'FH','Sikov_SAR':'SAR full-N',
            'Sikov_SAR_Published':'SAR observado'if spanish else'SAR observed'}
    indexed=part.set_index(keys);rows=[]
    for stratum in (1,2,3):
        for reference in REFERENCES:
            row=[str(stratum),labels[reference]]
            for group in groups:
                value=indexed.loc[stratum,reference,group]
                row.append(f(value.difference_Selected_minus_reference,4)+' ('+f(value.mcse,4)+')')
            rows.append(row)
    return rows


def primary_rows(data):
    return [[LABELS[p.reference],f(p.difference,4),
             '['+f(p.MC_CI95_low,4)+', '+f(p.MC_CI95_high,4)+']',sci(p.p_Holm3)]
            for p in data['primary'].itertuples()]


def sampled_graph_rows(data,spanish=False):
    part=data['sampled_graph_summary']
    part=part[part.group.eq('all')&part.outcome.eq('balanced_anemia_stunting')]
    if part.method.duplicated().any()or len(part)!=11 or not part.worlds.eq(64).all():
        raise RuntimeError('Complete separate 64-world all-method sensitivity required')
    labels={**LABELS,'Sikov_SAR_Published':'SAR observado'if spanish else'SAR observed',
            'Selected':'Mezcla 75/25'if spanish else'Selected mixture'}
    indexed=part.set_index('method')
    return [[labels[method],f(100*indexed.loc[method,'rmse'],2),
             f(100*indexed.loc[method,'coverage'],1),f(100*indexed.loc[method,'width'],2),
             f(indexed.loc[method,'interval_score'],4)]for method in METHODS]

"""Relabel three existing figures without fitting, scoring or changing values.

Original plotting statements are extracted read-only. Only visible strings and
file routing differ. An artist-data fingerprint is required to match before and
after relabelling. The availability CSV contains only district boolean masks
already displayed in the original application map, never response values.
"""
from __future__ import annotations
import ast
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT=Path(__file__).resolve().parents[3]
REV=ROOT/'revision_20261015'
OUT=REV/'results/terminology_20261018/figures'
CONTEXTS=('SAR_probability_srs','SAR_logit_srs','SAR_logit_psu_sparse','IID_logit_psu_sparse')
METHODS=('Sikov_FH','Sikov_SAR','Sikov_SAR_Published','ICC','Selected')
NAMES=('Probability SAR, SRS','Logit SAR, SRS','Logit SAR, sparse PSUs','IID logit, sparse PSUs')
COLORS=('#9e9e9e','#d7852d','#79599b','#527b9a','#184c3c')
Z=float(norm.ppf(.975))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p):return str(Path(p).relative_to(ROOT))
def json_bytes(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

def array_proof(v):
    a=np.ma.asarray(v)
    data=np.asarray(a.filled(np.nan),dtype=np.float64)
    return dict(shape=list(data.shape),sha256=hashlib.sha256(data.tobytes()).hexdigest())

def artist_fingerprint(fig):
    fig.canvas.draw()
    axes=[]
    for ax in fig.axes:
        d=dict(xlim=list(ax.get_xlim()),ylim=list(ax.get_ylim()),lines=[],collections=[],patches=[],images=[])
        for line in ax.lines:
            d['lines'].append(dict(x=array_proof(line.get_xdata()),y=array_proof(line.get_ydata())))
        for col in ax.collections:
            v=dict(type=type(col).__name__,offsets=array_proof(col.get_offsets()),
                   array=None if col.get_array() is None else array_proof(col.get_array()),
                   clim=list(col.get_clim()))
            if hasattr(col,'get_segments'):v['segments']=[array_proof(a) for a in col.get_segments()]
            if hasattr(col,'get_sizes'):v['sizes']=array_proof(col.get_sizes())
            if hasattr(col,'get_paths'):v['paths']=[array_proof(p.vertices) for p in col.get_paths()]
            d['collections'].append(v)
        for patch in ax.patches:
            d['patches'].append(dict(type=type(patch).__name__,vertices=array_proof(patch.get_path().vertices),
              bounds=[float(getattr(patch,k)()) for k in ('get_x','get_y','get_width','get_height')]
                     if all(hasattr(patch,k) for k in ('get_x','get_y','get_width','get_height'))else None))
        for image in ax.images:d['images'].append(dict(data=array_proof(image.get_array()),extent=list(image.get_extent())))
        axes.append(d)
    # NaN/Inf values cannot enter the serialized scalar limits; scientific
    # arrays are hashed bytewise and can faithfully retain masked locations.
    return dict(sha256=hashlib.sha256(json_bytes(axes)).hexdigest(),
                axes=len(axes),lines=sum(len(a['lines'])for a in axes),
                collections=sum(len(a['collections'])for a in axes),patches=sum(len(a['patches'])for a in axes))

def assignment(node,name):
    return isinstance(node,ast.Assign)and any(isinstance(n,ast.Name)and n.id==name for n in node.targets)

def plot_nodes(source,kind):
    tree=ast.parse(source.read_text())
    f=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==('figure'if kind=='incremental'else'main'))
    body=copy.deepcopy(f.body)
    if kind=='principal':
        a=next(i for i,n in enumerate(body)if assignment(n,'d'))
        b=next(i for i,n in enumerate(body)if assignment(n,'raw'))
        body=body[a:b]
    elif kind=='map':
        a=next(i for i,n in enumerate(body)if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)
               and ast.unparse(n.value.func)=='plt.rcParams.update')
        b=next(i for i,n in enumerate(body)if assignment(n,'result'))
        body=body[a:b]
    # Common, presentation-only export removal. We save the resulting figure
    # ourselves, with the original canvas, resolution and tight-bbox policy.
    body=[n for n in body if not assignment(n,'out')and not assignment(n,'png')and not assignment(n,'pdf')
          and not(isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)
                  and ast.unparse(n.value.func)in('fig.savefig','plt.close'))]
    return ast.Module(body=body,type_ignores=[])

class ReplaceText(ast.NodeTransformer):
    def __init__(self,replacements):self.replacements=replacements;self.changes=[]
    def visit_Constant(self,node):
        if isinstance(node.value,str)and node.value in self.replacements:
            self.changes.append([node.value,self.replacements[node.value]])
            return ast.copy_location(ast.Constant(self.replacements[node.value]),node)
        return node

def generate_one(kind,document,label,original,source,inputs,namespace,replacements,basename):
    before_ast=plot_nodes(source,kind)
    after_ast=copy.deepcopy(before_ast)
    transformer=ReplaceText(replacements);after_ast=transformer.visit(after_ast)
    # Normalize exactly the declared replacements, not arbitrary strings.
    reverse=ReplaceText({v:k for k,v in replacements.items()})
    normalized=reverse.visit(copy.deepcopy(after_ast))
    before_hash=hashlib.sha256(ast.dump(before_ast,include_attributes=False).encode()).hexdigest()
    after_hash=hashlib.sha256(ast.dump(normalized,include_attributes=False).encode()).hexdigest()
    assert before_hash==after_hash
    old_labels=('Sikov FH','Full-graph SAR','Published-scope SAR','ICC beta','Frozen 75/25 mixture')
    new_labels=(*old_labels[:-1],'Proposed mixture')
    proofs=[];new_fig=None
    for new,tree in [(False,before_ast),(True,after_ast)]:
        plt.rcdefaults()
        env=dict(np=np,pd=pd,plt=plt,Path=Path,ROOT=ROOT,out=OUT,CONTEXTS=CONTEXTS,NAMES=NAMES,
                 METHODS=METHODS,COLORS=COLORS,Z=Z,LABELS=new_labels if new else old_labels)
        env.update(namespace)
        exec(compile(ast.fix_missing_locations(tree),str(source),'exec'),env)
        fig=env['fig'];proofs.append(artist_fingerprint(fig))
        if new:new_fig=fig
        else:plt.close(fig)
    assert proofs[0]==proofs[1],(kind,proofs)
    OUT.mkdir(parents=True,exist_ok=True)
    png=OUT/(basename+'.png');svg=OUT/(basename+'.svg')
    kw=dict(bbox_inches='tight')if kind=='incremental'else{}
    new_fig.savefig(png,dpi=250 if kind=='incremental'else 300,**kw)
    new_fig.savefig(svg,**kw)
    canvas=list(new_fig.get_size_inches());plt.close(new_fig)
    actual_replacements=dict(transformer.changes)
    if kind=='principal':actual_replacements['Frozen 75/25 mixture']='Proposed mixture'
    return dict(document=document,label=label,original_path=rel(original),original_sha256=sha(original),
      revised_path=rel(png),revised_sha256=sha(png),revised_svg_path=rel(svg),revised_svg_sha256=sha(svg),
      generating_script_path=rel(__file__),generating_script_sha256=sha(__file__),
      source_script_path=rel(source),source_script_sha256=sha(source),
      input_files=[dict(path=rel(p),sha256=sha(p))for p in inputs],formatting_only=True,
      scientific_values_unchanged=True,label_replacements=actual_replacements,
      numeric_artist_fingerprint_before=proofs[0]['sha256'],numeric_artist_fingerprint_after=proofs[1]['sha256'],
      numeric_artist_counts=proofs[0],plotting_ast_sha256_before=before_hash,
      plotting_ast_sha256_after=after_hash,plotting_ast_normalization='Reverse only the explicitly declared visible text replacements; common removal of export calls and output paths.',
      canvas_inches=canvas,dpi=250 if kind=='incremental'else 300,
      visual_review_complete=False)

def main():
    source_root=ROOT/'src/estudio_publicacion/sikov_2026-10-10/scripts'
    score=ROOT/'expected/current_freeze/data/sikov_2026-10-10/resultados/published_sar_comparison/summary.csv'
    incremental=REV/'results/incremental_icc/incremental_summary.csv'
    districts=ROOT/'results_fresh/modern/resultados/aplicacion2019/matched_application2019_districts.csv'
    area_path=ROOT/'src/estudio_publicacion/sikov_2026-10-10/datos/real_2019_full/area_frame.csv'
    availability=REV/'results/terminology_20261018/district_availability_for_map.csv'
    areas=pd.read_csv(area_path,dtype={'ubigeo':str})
    data=pd.read_csv(districts,dtype={'ubigeo':str})
    mask=pd.read_csv(availability,dtype={'ubigeo':str}).set_index('ubigeo').loc[areas.ubigeo]
    ordered={m:g.set_index('ubigeo').loc[areas.ubigeo]for m,g in data.groupby('method')}
    for method,g in ordered.items():
        assert np.array_equal(g.latitude.to_numpy(),areas.latitude.to_numpy())
        assert np.array_equal(g.longitude.to_numpy(),areas.longitude.to_numpy())
    figures=[]
    figures.append(generate_one('principal','main','1',
      ROOT/'expected/current_freeze/data/sikov_2026-10-10/resultados/figuras/matched_rmse_coverage.png',
      source_root/'comparison_figures.py',[score],dict(source=score.parent),{},'main_figure1_proposed_mixture'))
    figures.append(generate_one('incremental','main','2',
      REV/'results/incremental_icc/Selected_vs_ICC_incremental.png',REV/'code/incremental_icc.py',[incremental],
      dict(summary=pd.read_csv(incremental),out=OUT),{'Selected − ICC':'Proposed mixture − ICC'},
      'main_figure2_proposed_mixture_vs_ICC'))
    figures.append(generate_one('map','supplement','S2',
      ROOT/'expected/current_freeze/data/sikov_2026-10-10/resultados/aplicacion2019/matched_application2019_pointmaps.png',
      source_root/'comparison_application_figures.py',[districts,area_path,availability],
      dict(ordered=ordered,ENDPOINTS=('anemia','stunting'),lon=areas.longitude.to_numpy(),lat=areas.latitude.to_numpy(),
           observed={e:mask[e+'_observed'].to_numpy(bool)for e in('anemia','stunting')},out=OUT),
      {': selected mixture':': Proposed mixture',': mixture minus sampled-graph SAR':': Proposed mixture minus sampled-graph SAR'},
      'supplement_figureS2_proposed_mixture_map'))
    receipt=dict(status='PASS_FORMATTING_ONLY_FIGURE_RELABEL_NUMERIC_ARTISTS_IDENTICAL',
      created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),scientific_fits_performed=False,
      raw_method_keys_changed=False,originals_modified=False,figures=figures,
      availability_mask_derivation=json.loads(availability.with_suffix('.provenance.json').read_text()))
    path=REV/'results/terminology_20261018/figure_relabel_receipt.json'
    path.write_text(json.dumps(receipt,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(dict(status=receipt['status'],figures=len(figures),receipt=rel(path))))

if __name__=='__main__':main()

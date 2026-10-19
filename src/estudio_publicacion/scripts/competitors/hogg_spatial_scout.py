"""Short full-rank S2 runtime scout; never a final posterior or acceptance gate."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import time
import numpy as np
import hogg_tsln as h
import hogg_spatial as sp


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source-npz',type=Path,required=True)
    p.add_argument('--edges',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--warmup',type=int,default=80)
    p.add_argument('--samples',type=int,default=80)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Keep every scout; choose a new output directory')
    args.output.mkdir(parents=True)
    codes,data=sp.load_modular_summary(args.source_npz)
    graph,meta,_=sp.graph_factor(codes,args.edges)
    if not meta['all_positive_modes_retained']:raise ValueError('Full-rank spatial scout required')
    model=sp.compile_spatial()
    cfg=h.MCMCConfig(chains=1,parallel_chains=1,warmup=args.warmup,samples=args.samples,
        seed=20261005+311,phase='runtime_scout_not_final')
    start=time.monotonic()
    fit=h._sample(model,{**data,**graph},cfg,args.output/'stage2')
    seconds=time.monotonic()-start
    mv=fit.method_variables()
    phases={}
    for csv in(args.output/'stage2').glob('*.csv'):
        for line in csv.read_text().splitlines():
            if not line.startswith('#'):continue
            for label in ['Warm-up','Sampling']:
                match=re.search(r'([0-9.eE+-]+) seconds \('+label+r'\)',line)
                if match:phases[label]=float(match.group(1))
    projection=None
    if 'Warm-up'in phases and 'Sampling'in phases:
        projection=4*(phases['Warm-up']*500/args.warmup+phases['Sampling']*1000/args.samples)
    report={'state':'Runtime/geometry scout only, not final posterior and not convergence acceptance.',
        'all_nonzero_CAR_modes_retained':True,'graph':meta,'config':h.asdict(cfg),
        'source_modular_sha256':hashlib.sha256(args.source_npz.read_bytes()).hexdigest(),
        'edges_sha256':hashlib.sha256(args.edges.read_bytes()).hexdigest(),
        'seconds_total_sampling_call':seconds,'one_chain_phase_seconds':phases,
        'projected_four_serial_chains_500warm_1000retained_seconds':projection,
        'projection_limits':'Linear phase extrapolation from one short chain; adaptation, tails, chain-specific geometry, CSV/summary overhead and a possible fixed diagnostic retry can materially change full-fit time.',
        'median_leapfrog_sampling':float(np.median(mv['n_leapfrog__'])),
        'p95_leapfrog_sampling':float(np.quantile(mv['n_leapfrog__'],.95)),
        'max_leapfrog_sampling':int(mv['n_leapfrog__'].max()),
        'observed_divergences_sampling':int(mv['divergent__'].sum()),
        'observed_max_depth_hits_sampling':int((mv['treedepth__']>=cfg.max_treedepth).sum()),
        'posterior_use':'These draws are excluded from all estimates, comparisons, coverage assessments and final posterior summaries. No scientific prior or graph is tuned from this run.'}
    h.write_json(args.output/'runtime_scout.json',report)
    print(json.dumps({'elapsed':seconds,'projection_seconds':projection,'median_leapfrog':report['median_leapfrog_sampling']}),flush=True)


if __name__=='__main__':main()

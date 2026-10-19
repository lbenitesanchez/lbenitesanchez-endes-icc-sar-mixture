"""Record an exact immutable set of three successful predictions before scoring."""
from pathlib import Path
import argparse
import datetime as dt
import json
import numpy as np
import pandas as pd
from validacion_score_external import ROOT, METHODS, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scope', choices=['temporal', 'territorial'], required=True)
    p.add_argument('--specifications', type=Path, required=True)
    args = p.parse_args()
    protocol = ROOT / 'protocol_freeze.json'
    proof = json.loads(protocol.read_text())
    if not proof.get('approved_for_development_fits', False):
        raise RuntimeError('Scientific protocol has not been frozen')
    if args.scope == 'temporal' and (ROOT / 'validacion/externa_first_reserved_open.json').exists():
        raise RuntimeError('Temporal predictions must be locked before reserved2024 response files are opened')
    output = ROOT / f'validacion/externa_prediction_lock_{args.scope}.json'
    if output.exists():
        raise FileExistsError('Existing prediction lock must be preserved')
    specs = json.loads(args.specifications.read_text())
    if sorted(s['method'] for s in specs) != sorted(METHODS):
        raise ValueError('Exact fixed three-method prediction family required')
    frame = None
    records = []
    for spec in specs:
        predpath, statuspath = ROOT / spec['path'], ROOT / spec['status_path']
        status = json.loads(statuspath.read_text())
        if not status.get('success', False):
            raise RuntimeError('Cannot lock unsuccessful model: ' + spec['method'])
        key, prefix = spec.get('area_key', 'ubigeo'), spec.get('mean_prefix', 'p_mean_')
        data = pd.read_csv(predpath, dtype={key: str})
        codes = data[key].str.zfill(6)
        if codes.duplicated().any() or not codes.str.fullmatch('[0-9]{6}').all():
            raise ValueError('Invalid or duplicate district predictions')
        if frame is None:
            frame = set(codes)
        elif set(codes) != frame:
            raise ValueError('Supplied prediction frames differ')
        for endpoint in ('anemia', 'stunting'):
            value = data[prefix + endpoint].to_numpy(float)
            if not np.isfinite(value).all() or ((value < 0) | (value > 1)).any():
                raise ValueError('Nonfinite or invalid endpoint prediction')
        records.append({**spec, 'sha256': sha(predpath), 'status_sha256': sha(statuspath), 'districts': len(data),
                        'area_key': key, 'mean_prefix': prefix})
    lock = {'scope': args.scope, 'locked_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
            'protocol_sha256': sha(protocol), 'approved_for_scoring': True,
            'reserved_outcomes_used_in_temporal_prediction': False,
            'reserved_territorial_test_outcomes_used_in_training': False,
            'predictions': records, 'source_sha256': sha(Path(__file__)),
            'specifications_sha256': sha(args.specifications)}
    output.write_text(json.dumps(lock, indent=2))
    print(json.dumps({'status': 'EXACT_PREDICTIONS_LOCKED', 'scope': args.scope,
                      'lock_sha256': sha(output), 'districts': len(frame)}), flush=True)


if __name__ == '__main__':
    main()

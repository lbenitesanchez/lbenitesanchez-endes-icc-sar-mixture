"""Training-only comparison of MAP initialization across basis/runtime inputs."""
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"
import argparse
import json
from pathlib import Path
import sys
import hashlib
import numpy as np
import scipy
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--endpoint", choices=("anemia", "stunting"), default="stunting")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    for path in (a.source_root / "scripts", a.source_root / "scripts/competitors"):
        sys.path.insert(0, str(path))
    from beta_power import CONFIG, EndpointPosterior, working_icc, cluster_effective_n
    if a.output.exists():
        raise FileExistsError("Preserve diagnostic output "+str(a.output))
    a.output.mkdir(parents=True)
    d = dict(np.load(a.inputs / "inputs.npz", allow_pickle=False)); units = pd.read_csv(a.inputs / "sample.csv")
    j = ("anemia", "stunting").index(a.endpoint)
    config = {**CONFIG, "implementation": "collapsed beta-power hyperposterior NUTS v1"}
    icc, info = working_icc(units, a.endpoint)
    n, _ = cluster_effective_n(units, a.endpoint, d["sample_idx"], d["n_eff"][:,j], icc)
    post = EndpointPosterior(d["X"], d["y"][:,j], n, d["sample_idx"], d["basis"], config).optimize()
    np.savez_compressed(a.output / "MAP_arrays.npz", mode=post.mode, covariance=post.cov,
                        inv_metric_diagonal=np.diag(post.cov), effective_n=n, basis=post.B)
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    report = {"role": "MAP diagnostic only; no NUTS or truth or score", "endpoint":a.endpoint,
        "inputs_npz_sha256":sha(a.inputs/"inputs.npz"),"sample_csv_sha256":sha(a.inputs/"sample.csv"),
        "source_beta_power_sha256":sha(a.source_root/"scripts/competitors/beta_power.py"),
        "source_stable_sha256":sha(a.source_root/"scripts/competitors/beta_power_stable.py"),
        "python":sys.version, "executable":sys.executable, "numpy":np.__version__, "scipy":scipy.__version__,
        "pandas":pd.__version__, "numerical_initialization":post.status,"working_ICC":info,
        "input_truth_values_read":False, "fitted_predictions_used_for_selection":False,
        "MAP_arrays_sha256":sha(a.output/"MAP_arrays.npz")}
    (a.output/"MAP_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"output":str(a.output),"log_kappa_mode":post.status["log_kappa_mode"],"gradient_max":post.status["max_gradient"]}))


if __name__ == "__main__":
    main()

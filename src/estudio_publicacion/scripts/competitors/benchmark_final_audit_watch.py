#!/usr/bin/env python3
"""Wait for both final evaluations before any arithmetic/readiness inspection."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from benchmark_inputs import ROOT, OUT, sha256


def completed(path, field):
    try:
        return json.loads(path.read_text()).get(field) is True
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def main():
    state = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                 state='waiting_for_both_evaluations', no_truth_or_performance_read_before_both=True,
                 source_sha256=sha256(Path(__file__)))
    marker = OUT / 'validation/final_audit_observer.json'
    while True:
        main_done = completed(OUT / 'final_evaluation/final_completion.json', 'evaluation_completed')
        spatial_done = completed(OUT / 'spatial_evaluation/completion.json', 'evaluation_completed')
        state.update(checked_utc=datetime.now(timezone.utc).isoformat(), main_completed=main_done, spatial_completed=spatial_done)
        marker.write_text(json.dumps(state, indent=2))
        if main_done and spatial_done:
            break
        time.sleep(30)
    state.update(state='checking_readiness_after_both_closed')
    marker.write_text(json.dumps(state, indent=2))
    script = Path(__file__).with_name('benchmark_validate_final_results.py')
    with (OUT / 'validation/final_readiness_after_both.log').open('w') as handle:
        readiness = subprocess.run([sys.executable, str(script)], stdout=handle, stderr=subprocess.STDOUT)
    while not completed(OUT / 'boundary_audit/completion.json', 'completed'):
        state.update(state='waiting_for_descriptive_boundary_accounting', readiness_returncode=readiness.returncode,
                     checked_utc=datetime.now(timezone.utc).isoformat())
        marker.write_text(json.dumps(state, indent=2)); time.sleep(30)
    script = Path(__file__).with_name('benchmark_independent_final_audit.py')
    state.update(state='independent_arithmetic_audit', independent_source_sha256=sha256(script))
    marker.write_text(json.dumps(state, indent=2))
    with (OUT / 'validation/final_independent_arithmetic.log').open('w') as handle:
        arithmetic = subprocess.run([sys.executable, str(script)], stdout=handle, stderr=subprocess.STDOUT)
    state.update(state='audit_processes_finished', readiness_returncode=readiness.returncode,
                 independent_arithmetic_returncode=arithmetic.returncode,
                 finished_utc=datetime.now(timezone.utc).isoformat())
    marker.write_text(json.dumps(state, indent=2))
    print(json.dumps(state, indent=2))


if __name__ == '__main__':
    main()

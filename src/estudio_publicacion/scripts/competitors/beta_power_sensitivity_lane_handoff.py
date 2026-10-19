"""Release one completed Gao operational lane to the declared sensitivities.

This observer changes only worker scheduling. It never reads predictions,
evaluation targets, sampling configurations, or statistical results.
"""
import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "resultados/resultados_comparadores"


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    os.replace(temporary, path)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    destination = BASE / "application_sensitivity_v2/application_sensitivity_budget.json"
    provenance = BASE / "application_sensitivity_v2/lane_handoff.json"
    while True:
        budget = read_json(destination)
        if budget and int(budget.get("active_workers", 0)) >= 1:
            print("Sensitivity lane is already enabled; no scheduling change.", flush=True)
            return
        for batch in ("gao_remaining550_C", "gao_remaining550_D"):
            continuation_path = BASE / f"dispatch_logs/continuation_{batch}.json"
            execution_path = BASE / f"dispatch_logs/{batch}/execution.json"
            continuation, execution = read_json(continuation_path), read_json(execution_path)
            if not continuation or not execution:
                continue
            accounted = (int(execution.get("completed_prediction_artifacts", 0))
                         + int(execution.get("explicit_failure_artifacts", 0)))
            ready = (
                continuation.get("state") == "finished"
                and continuation.get("returncode") == 0
                and execution.get("completed_execution") is True
                and execution.get("requested_fits") == 550
                and execution.get("worker_exit_codes") == [0]
                and execution.get("missing_artifacts") == 0
                and accounted == 550
            )
            if not ready:
                continue
            record = {
                "completed_utc": utc_now(),
                "released_batch": batch,
                "continuation_path": str(continuation_path.relative_to(ROOT)),
                "execution_path": str(execution_path.relative_to(ROOT)),
                "requested_and_accounted_fits": 550,
                "transferred_active_workers": 1,
                "maximum_concurrent_ENDES_tasks": int((read_json(ROOT/"resultados/competidores_hogg/runtime_capacity_limit.json") or {}).get("maximum_concurrent_ENDES_fitting_tasks",8)),
                "reads_predictions_or_evaluation_targets": False,
                "statistical_configuration_changed": False,
            }
            atomic_json(provenance, record)
            atomic_json(destination, {
                "active_workers": 1,
                "updated_utc": record["completed_utc"],
                "reason": "One fully completed Gao continuation lane is reassigned to the seven declared application sensitivities.",
                "lane_handoff": str(provenance.relative_to(ROOT)),
            })
            print(json.dumps(record), flush=True)
            return
        if not args.wait:
            print("No completed Gao continuation lane is available.", flush=True)
            return
        time.sleep(2)


if __name__ == "__main__":
    main()

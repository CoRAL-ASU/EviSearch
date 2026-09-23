"""A web-app extraction whose papers all ran is done even when a paper fails its check; a benchmark run is not."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_benchmark", ROOT / "experiment-scripts" / "run_benchmark.py")
run_benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_benchmark)

RECORDS = [{"doc_id": "ran-clean", "status": "ok", "check": {"result": "PASS"}},
           {"doc_id": "ran-check-failed", "status": "ok", "check": {"result": "FAIL"}},
           {"doc_id": "did-not-run", "status": "error", "check": None}]


def test_a_failed_check_fails_a_benchmark_run():
    failed, check_failed = run_benchmark.outcome(RECORDS, check_warnings=False)
    assert failed == ["did-not-run", "ran-check-failed"] and check_failed == ["ran-check-failed"]


def test_with_check_warnings_only_papers_that_did_not_run_fail():
    failed, check_failed = run_benchmark.outcome(RECORDS, check_warnings=True)
    assert failed == ["did-not-run"] and check_failed == ["ran-check-failed"]

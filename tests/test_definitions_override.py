"""Extraction can run on a generated schema's definitions; scoring always uses the hand-written ones."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = (
    "from src.config import config as c; "
    "from src.evaluation import claude_scoring as cs; "
    "print(c.DEFINITIONS_CSV_PATH); print(c.DEFINITIONS_EVAL_CATEGORY_PATH); print(c.HUMAN_DEFINITIONS_CSV_PATH)"
)


def _paths(env_value):
    env = {k: v for k, v in os.environ.items() if k != "EVISEARCH_DEFINITIONS_CSV"}
    if env_value:
        env["EVISEARCH_DEFINITIONS_CSV"] = env_value
    out = subprocess.run([sys.executable, "-c", PROBE], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    return [Path(line) for line in out.stdout.strip().splitlines()]


def test_extraction_definitions_follow_the_override_but_scoring_stays_on_the_hand_written_ones(tmp_path):
    extraction, scoring, human = _paths("")
    assert extraction == scoring == human and human.name == "Definitions_with_eval_category.csv"

    schema_csv = tmp_path / "schema_v1.csv"
    extraction, scoring, human = _paths(str(schema_csv))
    assert extraction == schema_csv
    assert scoring == human and scoring.name == "Definitions_with_eval_category.csv"

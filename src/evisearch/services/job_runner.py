"""Runs one job's command and records how it ended: `python -m src.evisearch.services.job_runner <job_id>`.

Started by jobs.run_process in its own session, so it outlives web-server restarts; the job file is its only link back.
SIGTERM (Cancel) stops the command's whole process group and records the job as cancelled.
"""
from __future__ import annotations

import signal
import subprocess
import sys
import time

from src.evisearch.services import jobs


def main(job_id: str) -> int:
    job = jobs.get(job_id)
    if not job or not job.get("cmd"):
        print(f"[job_runner] unknown job {job_id}", flush=True)
        return 2
    proc = subprocess.Popen(job["cmd"], cwd=jobs.PROJECT_ROOT)

    def stop(signum, frame):
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        jobs.update(job_id, status="cancelled", ended=time.time(), exit_code=proc.returncode)
        sys.exit(143)

    signal.signal(signal.SIGTERM, stop)
    code = proc.wait()
    current = jobs._read(job_id) or {}
    if current.get("status") != "cancelled":
        jobs.update(job_id, status="done" if code == 0 else "error", exit_code=code, ended=time.time(),
                    error=None if code == 0 else f"exit code {code} (see the log)")
    return code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))

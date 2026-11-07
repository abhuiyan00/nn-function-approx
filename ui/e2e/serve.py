"""Server for the browser tests: real engine and real results, but saved models go to a
temporary folder and jobs run a stand-in, so a test never overwrites anything in the repo."""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import uvicorn  # noqa: E402

from funcprox_ui.app import create_app  # noqa: E402
from funcprox_ui.jobs import JobQueue  # noqa: E402

FAKE_JOB = "import time\nprint('stand-in for the experiment σ', flush=True)\ntime.sleep(0.3)\nprint('finished', flush=True)"


def main(port: int) -> None:
    with tempfile.TemporaryDirectory() as models:
        jobs = JobQueue(command=lambda exp, profile, workers: [sys.executable, "-c", FAKE_JOB])
        uvicorn.run(create_app(models_dir=Path(models), jobs=jobs), host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main(int(sys.argv[1]))

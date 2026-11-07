"""The command line, as a user runs it."""
import os
import subprocess
import sys
from pathlib import Path

from funcprox import cli

SRC = str(Path(__file__).resolve().parents[1] / "src")


def test_targets_can_be_piped_whatever_the_locale():
    # a pipe uses the locale encoding (cp1252 on Windows), which has no π
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONUTF8", "PYTHONIOENCODING")}
    env["PYTHONPATH"] = SRC
    done = subprocess.run([sys.executable, "-m", "funcprox", "targets"], env=env, capture_output=True)
    assert done.returncode == 0, done.stderr.decode(errors="replace")[-300:]
    assert "sin(2πx)" in done.stdout.decode("utf-8")


def test_ui_without_its_extra_explains_how_to_install_it(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "uvicorn", None)  # makes `import uvicorn` raise ModuleNotFoundError
    assert cli.main(["ui", "--no-browser"]) == 1
    assert 'pip install -e ".[ui]"' in capsys.readouterr().err

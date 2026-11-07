"""The shared experiment machinery must run on every platform the team uses."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import common  # noqa: E402
import run_all  # noqa: E402

TINY = dict(target="f4", hidden=(4,), epochs=2, patience=0, n_train=30, n_val=10)


def test_run_many_uses_worker_processes_on_this_platform():
    rows = common.run_many([dict(TINY, seed=1), dict(TINY, seed=2)], workers=2)
    assert [r["seed"] for r in rows] == [1, 2]


def test_outputs_are_utf8_whatever_the_locale(workdir, monkeypatch):
    monkeypatch.setattr(common, "RES_DIR", workdir)
    common.write_summary("x", "σ × ω₀", "π", "quick")
    common.save_rows("x", [dict(model="MLP 2×H · LM σ")])
    common.save_json("x", dict(name="σ"), "blob")
    assert "σ × ω₀" in (workdir / "x" / "summary.md").read_bytes().decode("utf-8")
    assert "MLP 2×H · LM σ" in (workdir / "x" / "runs.csv").read_bytes().decode("utf-8")


def test_the_log_does_not_show_where_the_repository_lives(workdir, monkeypatch, capsys):
    (workdir / "docs").mkdir()
    monkeypatch.setattr(run_all, "ROOT", workdir)
    monkeypatch.setattr(sys, "argv", ["run_all.py", "--collect-only"])
    run_all.main()
    assert capsys.readouterr().out.strip() == "wrote docs/RESULTS.md"

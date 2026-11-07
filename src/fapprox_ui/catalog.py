"""Read-only view of what the project already wrote to disk: docs, results, figures.

Everything is parsed on request, so a re-run of an experiment or an edit of a document
shows up at once and the docs stay the single source of truth.
"""
from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import Counter
from functools import lru_cache
from pathlib import Path

from fapprox.functions import TARGETS

from .lab import clean, make_grid, round_array

# Interface runs from a checkout (editable install): results/ and docs/ are not packaged.
ROOT = Path(__file__).resolve().parents[2]
RESULTS, DOCS, EXPERIMENTS = ROOT / "results", ROOT / "docs", ROOT / "experiments"
SEED_SPREAD = 1.5  # the project does not claim effects below this (docs/FINDINGS.md)
LEARNED_BELOW = 0.1  # U3: a frequency counts as learned below 10 % relative amplitude error


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def experiment_scripts() -> dict[str, str]:
    return {p.stem.split("_")[0]: p.stem for p in sorted(EXPERIMENTS.glob("[eu][0-9]_*.py"))}


# ------------------------------------------------------------------------ datasets
def _gradcheck(raw: dict) -> list[dict]:
    return [dict(layer=name, check=label, error=v[key], params=v["params"])
            for name, v in raw.items() for key, label in (("grad", "gradient"), ("jac", "Jacobian"))]


def _curves(raw: dict) -> list[dict]:
    ks = (1, 4, 16)  # the three components of f3
    return [dict(model=model, step=step, k=f"k = {k}", rel_error=rel[j], final_rmse=c["rmse"])
            for model, c in raw.items() for step, rel in zip(c["step"], c["rel"]) for j, k in enumerate(ks)]


def _coverage(raw: dict) -> list[dict]:
    return [dict(score=score, nominal=round(1.0 - float(alpha), 3), coverage=cov, width=raw["width"][score][alpha])
            for score, table in raw["coverage"].items() for alpha, cov in table.items()]


def _ensembles(raw: dict) -> list[dict]:
    return [dict(activation=name, region=label, rmse=v[f"rmse_{key}"], spread=v[f"sd_{key}"])
            for name, v in raw["ensembles"].items() for key, label in (("in", "inside the box"), ("out", "outside the box"))]


ADAPTERS = {"gradcheck": ("gradcheck.json", _gradcheck), "curves": ("curves.json", _curves),
            "coverage": ("summary.json", _coverage), "ensembles": ("summary.json", _ensembles)}
JSON_DATASETS = {"e0_sanity": ["gradcheck"], "u3_spectral_bias": ["curves"], "u4_uncertainty": ["coverage", "ensembles"]}


def dataset_names(script: str) -> list[str]:
    names = [p.stem for p in sorted((RESULTS / script).glob("*.csv"))]
    return names + [n for n in JSON_DATASETS.get(script, []) if (RESULTS / script / ADAPTERS[n][0]).exists()]


def _parse_cell(v: str):
    if v in ("", "null"):
        return None
    try:
        f = float(v)
    except ValueError:
        return v
    return f if math.isfinite(f) else None


def _read_rows(script: str, name: str) -> list[dict] | None:
    if name not in dataset_names(script):
        return None
    if name in ADAPTERS:
        return ADAPTERS[name][1](json.loads(_read(RESULTS / script / ADAPTERS[name][0])))
    with open(RESULTS / script / f"{name}.csv", newline="", encoding="utf-8") as fh:
        raw = list(csv.DictReader(fh))
    # A column with any text in it stays text ('[32, 32]', 'True'), even where single cells look numeric
    text = {k for r in raw for k, v in r.items() if isinstance(_parse_cell(v), str)}
    return [{k: (v if v not in ("", "null") else None) if k in text else _parse_cell(v) for k, v in r.items()} for r in raw]


def load_dataset(exp: str, name: str) -> dict | None:
    script = experiment_scripts().get(exp)
    rows = _read_rows(script, name) if script else None
    if rows is None:
        return None
    columns = []
    for key in rows[0] if rows else []:
        present = [r[key] for r in rows if r[key] is not None]
        unique = list(dict.fromkeys(present))
        columns.append(dict(name=key, type="number" if present and all(isinstance(v, (int, float)) for v in present) else "string",
                            unique=len(unique), constant=len(unique) <= 1, values=unique if len(unique) <= 24 else None))
    return clean(dict(experiment=exp, name=name, columns=columns, rows=rows))


def all_datasets() -> list[dict]:
    out = []
    for exp, script in experiment_scripts().items():
        for name in dataset_names(script):
            rows = _read_rows(script, name) or []
            out.append(dict(experiment=exp, name=name, rows=len(rows)))
    return out


# --------------------------------------------------------------------- experiments
def _sections(markdown: str) -> dict[str, tuple[str, str]]:
    found = re.finditer(r"^### ([EU]\d) · (.+?)\n(.*?)(?=^#{2,3} |\Z)", markdown, re.M | re.S)
    return {m[1].lower(): (m[2].strip(), m[3].strip().rstrip("-").strip()) for m in found}


def _table_cell(markdown: str, key: str) -> str:
    m = re.search(rf"^\|\s*{re.escape(key)}\s*\|\s*(.+?)\s*\|", markdown, re.M)
    return m[1] if m else ""


def _owners(plan: str) -> dict[str, str]:
    m = re.search(r"^\|\s*Experiments\s*\|(.+?)\|(.+?)\|", plan, re.M)
    if not m:
        return {}
    return {e.strip().lower(): owner for owner, cell in (("A", m[1]), ("B", m[2])) for e in cell.split(",")}


def _summary_parts(script: str) -> dict:
    text = _read(RESULTS / script / "summary.md")
    m = re.search(r"_Profile: \*\*(\w+)\*\* · generated ([\d-]+ [\d:]+)_", text)
    body = text[m.end():].strip() if m else text
    return dict(profile=m[1] if m else None, generated=m[2] if m else None, summary=body)


def list_experiments() -> list[dict]:
    readme, plan = _read(ROOT / "README.md"), _read(DOCS / "PROJECT_PLAN.md")
    protocol, owners = _sections(_read(DOCS / "EXPERIMENTS.md")), _owners(plan)
    out = []
    for exp, script in experiment_scripts().items():
        parts = _summary_parts(script)
        out.append(dict(id=exp, script=script, title=protocol.get(exp, (script, ""))[0], group="sensitivity" if exp[0] == "e" else "unique",
                        question=_table_cell(readme, exp.upper()), hypothesis=_table_cell(plan, exp.upper()),
                        owner=owners.get(exp), profile=parts["profile"], generated=parts["generated"]))
    return out


def experiment_detail(exp: str) -> dict | None:
    base = next((e for e in list_experiments() if e["id"] == exp), None)
    if base is None:
        return None
    script = base["script"]
    figures = sorted((DOCS / "figures").glob(f"{script}_*.png"))
    return dict(**base,
                protocol=_sections(_read(DOCS / "EXPERIMENTS.md")).get(exp, ("", ""))[1],
                findings=_sections(_read(DOCS / "FINDINGS.md")).get(exp, ("", ""))[1],
                summary=_summary_parts(script)["summary"],
                figures=[dict(name=p.stem.removeprefix(f"{script}_").replace("_", " "), url=f"/figures/{p.name}") for p in figures],
                datasets=[dict(name=n, rows=len(_read_rows(script, n) or [])) for n in dataset_names(script)])


# ------------------------------------------------------------------------ overview
def _rows(exp: str, name: str) -> list[dict]:
    script = experiment_scripts().get(exp)
    return (_read_rows(script, name) if script else None) or []


def _highlights() -> dict:
    h: dict = {}
    grad = _rows("e0", "gradcheck")
    if grad:
        h["engine"] = dict(worst_error=max(r["error"] for r in grad), layers=len({r["layer"] for r in grad}))

    u1 = _rows("u1", "runs")
    if u1:
        def median(t: str, method: str) -> float:
            return statistics.median(r["rmse"] for r in u1 if r["target"] == t and r["method"] == method)

        h["lm_vs_adam"] = {t: median(t, "Adam") / median(t, "LM") for t in sorted({r["target"] for r in u1})}

    u2 = [r for r in _rows("u2", "runs") if r["target"] == "f7"]
    kan = [r for r in u2 if r["model"] == "KAN [2,1,1]"]
    mlp = [r["rmse"] for r in u2 if r["model"].startswith("MLP")]
    if kan and mlp:
        best = min(kan, key=lambda r: r["rmse"])
        h["kan"] = dict(rmse=best["rmse"], n_params=int(best["n_params"]), best_mlp=min(mlp))

    raw = _read(RESULTS / "u3_spectral_bias" / "curves.json")
    if raw:
        curves = json.loads(raw)

        def learned(model: str) -> int | None:
            return next((s for s, rel in zip(curves[model]["step"], curves[model]["rel"]) if rel[2] < LEARNED_BELOW), None)

        h["spectral"] = dict(tanh_learned_at=learned("tanh MLP"), fourier_learned_at=learned("Fourier features + tanh"),
                             tanh_rmse=curves["tanh MLP"]["rmse"], fourier_rmse=curves["Fourier features + tanh"]["rmse"],
                             steps=curves["tanh MLP"]["step"][-1])

    raw = _read(RESULTS / "u4_uncertainty" / "summary.json")
    if raw:
        h["conformal"] = dict(nominal=0.9, coverage=json.loads(raw)["coverage"]["plain"]["0.1"])
    return h


def _verdicts() -> list[dict]:
    """Best network vs best classical approximator per target of E6, over the whole size sweep."""
    rows = _rows("e6", "runs")
    out = []
    for key in sorted({r["target"] for r in rows}):
        rs = [r for r in rows if r["target"] == key]
        net = min((r for r in rs if r["model"].startswith("MLP")), key=lambda r: r["rmse"])
        cls = min((r for r in rs if not r["model"].startswith("MLP")), key=lambda r: r["rmse"])
        ratio = max(net["rmse"], cls["rmse"]) / min(net["rmse"], cls["rmse"])
        winner = "tie" if ratio < SEED_SPREAD else "network" if net["rmse"] < cls["rmse"] else "classical"
        out.append(dict(target=key, formula=TARGETS[key].formula, winner=winner, ratio=ratio, orders=math.log10(ratio),
                        network=dict(model=net["model"], rmse=net["rmse"], n_params=int(net["n_params"])),
                        classical=dict(model=cls["model"], rmse=cls["rmse"], n_params=int(cls["n_params"]))))
    return out


def _timeline(plan: str) -> list[dict]:
    m = re.search(r"```mermaid\s+gantt(.*?)```", plan, re.S)
    out = []
    for line in (m[1].splitlines() if m else []):
        if ":" not in line or line.strip().startswith(("title", "dateFormat", "section")):
            continue
        label, _, rest = line.rpartition(":")
        fields = [f.strip() for f in rest.split(",")]
        dates = [f for f in fields if re.fullmatch(r"\d{4}-\d{2}-\d{2}", f)]
        if dates:
            out.append(dict(label=label.strip(), start=dates[0], end=dates[-1], milestone="milestone" in fields,
                            done="done" in fields))
    return sorted(out, key=lambda t: t["start"])


def overview() -> dict:
    experiments = list_experiments()
    timeline = _timeline(_read(DOCS / "PROJECT_PLAN.md"))
    deadline = next((t["end"] for t in reversed(timeline) if t["milestone"]), None)
    profiles = Counter(e["profile"] for e in experiments if e["profile"])
    stamps = [e["generated"] for e in experiments if e["generated"]]
    return clean(dict(profiles=dict(profiles), generated=max(stamps) if stamps else None,
                      run_count=sum(d["rows"] for d in all_datasets() if d["name"] in ("runs", "lr_batch", "optimizers")),
                      highlights=_highlights(), verdicts=_verdicts(), timeline=timeline, deadline=deadline))


# ------------------------------------------------------------------------- targets
@lru_cache(maxsize=1)
def targets() -> list[dict]:
    out = []
    for t in TARGETS.values():
        _, preview = make_grid(t, n1=160, n2=28)
        preview["exact"] = round_array(preview["exact"], 4)
        out.append(dict(key=t.key, name=t.name, formula=t.formula, dim=t.dim, probes=t.probes,
                        domain=[list(d) for d in t.domain], frequencies=list(t.frequencies), preview=preview))
    return out


# ----------------------------------------------------------------------- documents
def _document_paths() -> dict[str, Path]:
    paths = [ROOT / "README.md", *sorted(DOCS.glob("*.md")), ROOT / "CONTRIBUTING.md"]
    return {p.stem.lower(): p for p in paths if p.exists()}


def list_documents() -> list[dict]:
    out = []
    for slug, path in _document_paths().items():
        m = re.search(r"^#\s+(.+)$", _read(path), re.M)
        out.append(dict(slug=slug, title=m[1] if m else path.stem, path=path.relative_to(ROOT).as_posix()))
    return out


def load_document(slug: str) -> dict | None:
    path = _document_paths().get(slug)
    return dict(slug=slug, path=path.relative_to(ROOT).as_posix(), markdown=_read(path)) if path else None

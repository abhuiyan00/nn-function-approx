# Convenience targets. On Windows use the VS Code tasks (.vscode/tasks.json) or run the commands directly.
PY ?= python

.PHONY: install test quick full clean ui

install:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest

quick:            ## all experiments, quick profile (~15-25 min on 2 cores)
	$(PY) experiments/run_all.py --profile quick

full:             ## all experiments, full profile (hours; run overnight)
	$(PY) experiments/run_all.py --profile full

ui:               ## build the frontend and open the interface
	$(PY) -m pip install -e ".[ui]"
	npm --prefix ui ci
	npm --prefix ui run build
	$(PY) -m fapprox ui

clean:
	rm -rf .pytest_cache src/*.egg-info
	find . -name __pycache__ -type d -exec rm -rf {} +

# Working together (two people, one repository)

## Branches
* `main` is always green: tests pass, figures regenerate.
* Work on a branch named after the owner and the topic: `a/e4-noise-sweep`, `b/u2-kan-grid`.
* Open a pull request into `main`; the **other** member reviews and merges.

## Commits
Conventional Commits, subject ≤ 50 characters, body says *why*:
```
feat(kan): grid range update from samples
fix(train): restore best weights after LM early stop
exp(e4): add LM series to the N-sigma sweep
docs(report): draft section 2 (data sets)
```
Types used here: `feat`, `fix`, `test`, `exp` (experiment scripts / results), `docs`, `chore`, `refactor`.

## Before every push
```bash
pytest                                     # all tests green
python experiments/<changed_exp>.py        # quick profile, check the figure
npm --prefix ui test                       # if you changed the interface; more in ui/README.md
```

## Results and figures
* Only scripts in `experiments/` write to `results/` and `docs/figures/`; never edit them by hand.
* Commit CSV / summary / PNG files together with the script change that produced them.
* Full-profile results for the report: one member runs `python experiments/run_all.py --profile full`
  overnight on one machine, then commits in one `exp(all): full profile` commit.
* Model files (`*.npz`) stay out of git (`.gitignore`).

## Code style
* NumPy only in `src/funcprox` (Matplotlib only in `plotting.py`). Interface dependencies stay in
  `src/funcprox_ui` and `ui/`; `src/funcprox` must keep working without them.
* Every new layer gets a gradient test in `tests/test_engine.py`.
* Docstring at the top of each experiment: question, variables, outputs.

## Weekly sync (30 min)
1. What changed (PRs merged), 2. what the figures say, 3. next week's owner/task list, 4. questions for the supervisor.

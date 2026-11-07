# funcprox interface: frontend

React, TypeScript, Vite, Tailwind CSS and shadcn/ui, talking to the FastAPI app in
`src/funcprox_ui`. The engine in `src/funcprox` stays NumPy-only. See the project
[README](../README.md) for how to install and open the interface.

## Develop

```bash
python -m funcprox ui --no-browser      # API on :8765
npm --prefix ui run dev                # hot reload on :5173, proxies the API and images to :8765
```

## Check

```bash
npm --prefix ui run typecheck
npm --prefix ui run lint
npm --prefix ui test                   # unit tests of the logic (Vitest)
npm --prefix ui run build              # the browser tests run against this build
npm --prefix ui run e2e                # browser and axe accessibility tests (Playwright), about 5 min
python -m pytest                       # the HTTP contract, in tests/test_ui_api.py
```

Run `npx playwright install chromium` once, inside `ui/`, before the browser tests. They use
`e2e/serve.py`: the real engine and result files, with saved models in a temporary folder and a
stand-in for experiment jobs, so a test never overwrites anything in the repository.

## Layout

| Path | Contents |
|---|---|
| `src/pages/` | One file per route; the lab is in `src/pages/lab/` |
| `src/components/charts/` | Own SVG and canvas charts on d3-scale and d3-shape |
| `src/components/FunctionPicker.tsx` | Formula, points and drawing input, and the missing-middle sliders, shared by the Curve Finder and the lab |
| `src/components/ui/` | shadcn/ui components, generated: `npx shadcn@latest add <name>` |
| `src/lib/api.ts` | Types and calls of every API route |
| `src/lib/experimentCharts.ts` | The interactive chart of each experiment, as data |
| `src/hooks/useRunStream.ts` | Follows a training run over Server-Sent Events |

## Add a chart to an experiment

Add an entry to `EXPERIMENT_CHARTS` in `src/lib/experimentCharts.ts`, naming a result file and its
columns. A JSON result needs an adapter to rows first: see `ADAPTERS` in
`src/funcprox_ui/catalog.py`. CSV files work as they are.

## Design rules

* **Tokens only.** Colours are CSS variables in `src/index.css`, for both themes. No raw hex in components.
* **Quiet chrome, loud data.** Neutral surfaces with one accent, so colour on screen means data or a state.
* **Colour follows the entity.** A series keeps its slot, whichever filter is active. Slots are assigned
  in a fixed order and never cycled.
* **Colour is never alone.** Series also differ by marker shape, status also has an icon and a label, and
  every chart has a table view and works with the arrow keys.
* **Series palette.** Checked for colour-blind separation in both themes. Check again if you change it.
  Light-theme series with low contrast (aqua, yellow, magenta) rely on the legend, marker shape and table view.
* **Offline.** Fonts and libraries are bundled, so a demo needs no network. Only the Swagger page at
  `/api/docs` loads from a CDN.

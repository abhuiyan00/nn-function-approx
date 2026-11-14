import { useQueries, useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { EmptyState, ErrorState, LoadingBlock, Segmented } from "@/components/common";
import { HeatMap } from "@/components/charts/HeatMap";
import { XYChart, type XYSeries } from "@/components/charts/XYChart";
import { seriesColor } from "@/components/charts/core";
import { type KanEdge, type RunDetail, api } from "@/lib/api";
import { duration, factor, integer, percent, sci } from "@/lib/format";
import { fitRegions } from "@/lib/functions";

const SPECTRUM_FLOOR = 1e-6;

export function LossChart({ run, height = 280 }: { run: RunDetail; height?: number }) {
  const series = useMemo<XYSeries[]>(() => {
    const h = run.history;
    return [
      { id: "train", label: "Training MSE", color: seriesColor(0), x: h.epoch, y: h.train },
      { id: "val", label: "Validation MSE", color: seriesColor(1), x: h.epoch, y: h.val },
      { id: "test", label: run.grid.known === false ? "MSE on held-out points" : "MSE against exact f", color: seriesColor(2), x: h.test_epoch.filter((e) => e > 0), y: h.test_mse.slice(h.test_epoch.length - h.test_epoch.filter((e) => e > 0).length), width: 1.5 },
    ];
  }, [run.history, run.grid.known]);
  const best = run.final?.best_epoch;
  const vlines = useMemo(() => (best && best > 0 ? [{ x: best, label: `Best, epoch ${best}` }] : undefined), [best]);
  return (
    <XYChart
      series={series}
      yScale="log"
      xLabel={run.spec.optimizer === "lm" ? "Iteration" : "Epoch"}
      yLabel="MSE (scaled)"
      height={height}
      vlines={vlines}
      ariaLabel="Training, validation and exact-function error over training"
    />
  );
}

export function SpectrumPanel({ run }: { run: RunDetail }) {
  const exact = run.grid.spectrum_exact;
  const pred = (run.final?.spectrum ?? run.live?.spectrum)?.pred;
  const series = useMemo<XYSeries[]>(() => {
    if (!exact) return [];
    const k = exact.map((_, i) => i);
    const floor = (v: number | null) => (v == null ? null : Math.max(v, SPECTRUM_FLOOR));
    const out: XYSeries[] = [{ id: "exact", label: "Exact f", color: "var(--axis)", x: k, y: exact.map(floor), width: 4.5, underlay: true }];
    if (pred) out.push({ id: "pred", label: "Network", color: seriesColor(0), x: k, y: pred.map(floor), width: 1.75 });
    return out;
  }, [exact, pred]);
  const components = run.components;
  const componentSeries = useMemo<XYSeries[]>(() => {
    if (!components) return [];
    return components.k.map((k, j) => ({
      id: `k${k}`, label: `k = ${k}`, color: seriesColor(j), x: components.epoch, y: components.rel.map((r) => Math.max(r[j], SPECTRUM_FLOOR)),
    }));
  }, [components]);
  const learned = useMemo(() => [{ y: 0.1, label: "learned below 0.1" }], []);

  return (
    <div className="space-y-5">
      <div>
        <h3 className="mb-1 text-sm font-medium">Frequency content</h3>
        <p className="mb-2 text-xs text-muted-foreground">Amplitude by the number of cycles across the domain. A plain network fills in low frequencies first.</p>
        <XYChart series={series} yScale="log" xLabel="Cycles across the domain, k" yLabel="Amplitude" height={240} ariaLabel="Amplitude spectrum of the network and of the exact function" />
      </div>
      {componentSeries.length > 0 && (
        <div>
          <h3 className="mb-1 text-sm font-medium">When each component is learned</h3>
          <p className="mb-2 text-xs text-muted-foreground">Relative error of the amplitude of sin(kπx), measured as in experiment U3.</p>
          <XYChart series={componentSeries} yScale="log" xLabel={run.spec.optimizer === "lm" ? "Iteration" : "Epoch"} yLabel="Relative error" height={240} hlines={learned} ariaLabel="Relative amplitude error per frequency component over training" />
        </div>
      )}
    </div>
  );
}

export function DampingPanel({ run }: { run: RunDetail }) {
  const series = useMemo<XYSeries[]>(() => [{ id: "mu", label: "Damping μ", color: seriesColor(0), x: run.history.epoch, y: run.history.mu }], [run.history]);
  return (
    <div>
      <p className="mb-2 text-xs text-muted-foreground">Large μ takes small gradient-descent steps. As μ falls, Levenberg-Marquardt turns into Gauss-Newton.</p>
      <XYChart series={series} yScale="log" xLabel="Iteration" yLabel="μ" height={260} ariaLabel="Levenberg-Marquardt damping over the iterations" />
    </div>
  );
}

function EdgeChart({ edge, layers }: { edge: KanEdge; layers: number }) {
  const series = useMemo<XYSeries[]>(() => {
    const out: XYSeries[] = [{ id: "phi", label: "Learned edge", color: seriesColor(0), x: edge.x, y: edge.y }];
    if (edge.match) out.unshift({ id: "fit", label: `Closest form: a·${edge.match.name} + b`, color: "var(--axis)", x: edge.x, y: edge.match.fit, width: 4.5, underlay: true });
    return out;
  }, [edge]);
  const title = layers > 1 ? `Layer ${edge.layer + 1}: input ${edge.i + 1} to output ${edge.o + 1}` : `Input ${edge.i + 1} to output ${edge.o + 1}`;
  return (
    <figure className="min-w-0 rounded-md border p-3">
      <figcaption className="mb-1 flex flex-wrap justify-between gap-x-2 text-xs">
        <span className="font-medium">{title}</span>
        {edge.match && <span className="tnum text-muted-foreground">R² {edge.match.r2 > 0.9999 ? "> 0.9999" : `= ${edge.match.r2.toFixed(4)}`}</span>}
      </figcaption>
      <XYChart series={series} xLabel="Input" height={190} ariaLabel={`${title}: learned edge function`} />
    </figure>
  );
}

export function EdgesPanel({ run }: { run: RunDetail }) {
  const edges = run.final?.edges ?? run.live?.edges;
  if (!edges) return <EmptyState title="Too many edges to draw">Edge functions are shown for KANs with at most 24 edges.</EmptyState>;
  const layers = new Set(edges.map((e) => e.layer)).size;
  return (
    <div className="space-y-3">
      <p className="text-xs text-muted-foreground">
        A KAN learns a function on every edge, so its parts can be read directly.
        {" "}Each edge is compared with a short list of simple forms (x, x², x³, |x|, log, √, exp, tanh, sin) and the closest is drawn under it.
        A high R² suggests that form. It does not prove it: the same function can train into different edges from other starting weights.
      </p>
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
        {edges.map((e) => <EdgeChart key={`${e.layer}-${e.i}-${e.o}`} edge={e} layers={layers} />)}
      </div>
    </div>
  );
}

export function BaselinesPanel({ run }: { run: RunDetail }) {
  const final = run.final;
  if (!final) return <EmptyState title="Available when training ends">The classical fits are computed once for the finished network.</EmptyState>;
  const net = final.metrics.rmse;
  const rows = [
    { model: `This network (${run.label})`, n: final.metrics.n_params, rmse: net, max: final.metrics.max_err, hole: final.metrics.rmse_gap, own: true },
    ...final.baselines.map((b) => ({ model: b.model, n: b.n_coef, rmse: b.rmse, max: b.max_err, hole: b.rmse_gap, own: false })),
  ];
  const hasHole = final.metrics.rmse_gap != null;
  const best = final.baselines.reduce((a, b) => (b.rmse < a.rmse ? b : a), final.baselines[0]);
  const verdict = net == null || !best ? null
    : best.rmse < net ? `${best.model} with ${best.n_coef.toLocaleString("en-US")} coefficients beats this network by ${factor(net / best.rmse)}.`
    : `This network beats ${best.model} with ${best.n_coef.toLocaleString("en-US")} coefficients by ${factor(best.rmse / net)}.`;
  return (
    <div className="space-y-3">
      {verdict && <p className="text-sm font-medium">{verdict}</p>}
      <div className="overflow-x-auto rounded-md border">
        <table className="w-full text-[13px]">
          <caption className="sr-only">Network against least-squares approximators on the same training data</caption>
          <thead className="bg-wash text-left text-xs text-muted-foreground">
            <tr>
              <th scope="col" className="px-3 py-2 font-medium">Approximator</th>
              <th scope="col" className="px-3 py-2 text-right font-medium">Fitted values</th>
              <th scope="col" className="px-3 py-2 text-right font-medium">{run.grid.known === false ? "RMSE on held-out points" : "RMSE vs exact f"}</th>
              {hasHole && <th scope="col" className="px-3 py-2 text-right font-medium">In the missing middle</th>}
              <th scope="col" className="px-3 py-2 text-right font-medium">Max error</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.model} className={`border-t ${r.own ? "font-medium" : ""}`}>
                <td className="px-3 py-1.5">{r.model}</td>
                <td className="tnum px-3 py-1.5 text-right">{r.n.toLocaleString("en-US")}</td>
                <td className="tnum px-3 py-1.5 text-right">{sci(r.rmse)}</td>
                {hasHole && <td className="tnum px-3 py-1.5 text-right">{sci(r.hole)}</td>}
                <td className="tnum px-3 py-1.5 text-right">{sci(r.max)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-xs text-subtle">
        Least squares on the same training points. The size is the best of experiment E6's sweep no larger than the network, chosen on the validation points. Scored on the same exact grid.
        The RBF network is a network too, but nothing in it is trained by gradient steps: k-means places its Gaussian units and least squares solves the output weights.
        Polynomials and splines on a grid stop at two inputs, so with three the RBF network is the only yardstick.
      </p>
    </div>
  );
}

export function ComparePanel({ ids }: { ids: string[] }) {
  const [axis, setAxis] = useState<"epoch" | "time">("epoch");
  const results = useQueries({
    queries: ids.map((id) => ({ queryKey: ["lab-run", id], queryFn: () => api.run(id), staleTime: 2000 })),
  });
  const runs = results.map((r) => r.data).filter((r): r is RunDetail => !!r);
  const key = runs.map((r) => `${r.id}:${r.history.epoch.length}`).join();
  const series = useMemo<XYSeries[]>(() => runs.map((r, i) => ({
    id: r.id,
    label: `${r.spec.target} · ${r.label}`,
    color: seriesColor(i),
    x: axis === "epoch" ? r.history.epoch : r.history.wall,
    y: r.history.val,
    // eslint-disable-next-line react-hooks/exhaustive-deps
  })), [key, axis]);

  if (ids.length === 0) {
    return <EmptyState title="Nothing selected yet">Tick up to four runs in the table below to compare their validation curves.</EmptyState>;
  }
  return (
    <div className="space-y-2">
      <div className="flex justify-end">
        <Segmented label="Horizontal axis" value={axis} onChange={setAxis} options={[{ value: "epoch", label: "By epoch" }, { value: "time", label: "By time" }]} />
      </div>
      <XYChart
        series={series}
        yScale="log"
        xLabel={axis === "epoch" ? "Epoch or iteration" : "Seconds"}
        yLabel="Validation MSE (scaled)"
        formatX={axis === "time" ? duration : undefined}
        height={300}
        ariaLabel="Validation curves of the selected runs"
      />
    </div>
  );
}

export function UnitsPanel({ run }: { run: RunDetail }) {
  const units = run.final?.units ?? run.live?.units;
  const x = run.grid.kind === "1d" ? run.grid.units_x : null;
  const series = useMemo<XYSeries[]>(() => {
    if (!units || !x) return [];
    const total = x.map((_, i) => {
      const sum = units.curves.reduce((s, c) => s + (c[i] ?? NaN), (units.rest?.[i] ?? 0) + (units.offset ?? 0));
      return Number.isFinite(sum) ? sum : null;
    });
    return [
      { id: "sum", label: "All units + bias = the network", color: "var(--foreground)", x, y: total, width: 2.5 },
      ...units.curves.map((c, k) => ({ id: `u${k}`, label: `Unit ${units.index[k] + 1}`, color: seriesColor(k), x, y: c, width: 1.25 })),
      ...(units.rest ? [{ id: "rest", label: "The other units together", color: "var(--subtle)", x, y: units.rest, width: 1.25 }] : []),
    ];
  }, [units, x]);
  if (!units || !x) return <EmptyState title="Not available for this network">Unit shares are drawn for MLPs with one input.</EmptyState>;
  return (
    <div>
      <p className="mb-2 text-xs text-muted-foreground">
        The answer is a weighted sum of the last hidden layer. A thin curve is one unit's share, its weight times its value.
        The thick curve adds them all and the bias: that sum is the network. The {units.curves.length} largest shares are drawn one by one.
      </p>
      <XYChart series={series} xLabel="x" yLabel="Share of the answer" height={300} ariaLabel="Share of each hidden unit in the answer of the network" />
    </div>
  );
}

export function SlopePanel({ run }: { run: RunDetail }) {
  const slope = run.final?.slope ?? run.live?.slope;
  const grid = run.grid;
  const exact = grid.kind === "1d" ? grid.slope_exact : null;
  const series = useMemo<XYSeries[]>(() => {
    if (!exact || grid.kind !== "1d") return [];
    const out: XYSeries[] = [{ id: "exact", label: "Slope of the exact f", color: "var(--axis)", x: grid.x, y: exact, width: 4.5, underlay: true }];
    if (slope) out.push({ id: "pred", label: "Slope of the network", color: seriesColor(0), x: grid.x, y: slope.pred, width: 1.75 });
    return out;
  }, [exact, slope, grid]);
  const regions = useMemo(() => (grid.kind === "1d" ? fitRegions(grid) : []), [grid]);
  if (!exact) return <EmptyState title="Not available for this function">The slope check needs one input and an exact function.</EmptyState>;
  return (
    <div>
      <p className="mb-2 text-xs text-muted-foreground">
        The slope of the network comes from the chain rule through its layers, not from differences. A network can sit close to f and still get f′ wrong:
        small ripples in the fit become large in the slope.
        {slope?.rmse != null && <> RMSE of the slope inside the box: <span className="tnum font-medium text-foreground" data-testid="slope-rmse">{sci(slope.rmse)}</span>.</>}
      </p>
      <XYChart series={series} xLabel="x" yLabel="df/dx" height={280} regions={regions} ariaLabel="Slope of the network against the slope of the exact function" />
    </div>
  );
}

export function LandscapePanel({ run }: { run: RunDetail }) {
  const done = !!run.final;
  const query = useQuery({ queryKey: ["landscape", run.id], queryFn: () => api.landscape(run.id), enabled: done, staleTime: Infinity });
  if (!done) return <EmptyState title="Available when training ends">The surface is measured around the final weights.</EmptyState>;
  if (query.isError) return <ErrorState error={query.error} onRetry={() => query.refetch()} />;
  const d = query.data;
  if (!d) return <LoadingBlock className="h-72" />;
  return (
    <div className="grid grid-cols-1 items-start gap-4 md:grid-cols-[minmax(0,420px)_minmax(0,1fr)]">
      <HeatMap
        title="log₁₀ of the training error"
        x={d.alpha}
        y={d.beta}
        values={d.loss}
        height={320}
        path={{ x: d.path.a, y: d.path.b }}
        axes={["direction 1", "direction 2"]}
      />
      <div className="space-y-2 text-xs text-muted-foreground">
        <p>
          The error over a plane in weight space. The plane passes through the final weights at (0, 0) and is spanned by the two directions that hold
          {d.explained != null ? <> <span className="tnum font-medium text-foreground">{percent(d.explained)}</span> of </> : " most of "}
          the path training took (its principal components, as in Li et al. 2018).
        </p>
        <p>The line is that path: the hollow dot is the starting weights, the filled dot the trained network. The scale above the map runs from the lowest error on the left to the highest on the right.</p>
        <p>
          A long path along a narrow valley is what slows first-order methods down. Levenberg-Marquardt rescales the valley with curvature and crosses it in a few steps:
          train the same network with Adam and with LM and compare the two paths.
        </p>
        <p className="text-subtle">Measured on {integer(Math.min(512, run.spec.n_train))} training points, {d.alpha.length} × {d.beta.length} forward passes. Everything off this plane is not shown.</p>
      </div>
    </div>
  );
}

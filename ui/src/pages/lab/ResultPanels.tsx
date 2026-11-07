import { useQueries } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { EmptyState, Segmented } from "@/components/common";
import { XYChart, type XYSeries } from "@/components/charts/XYChart";
import { seriesColor } from "@/components/charts/core";
import { type KanEdge, type RunDetail, api } from "@/lib/api";
import { duration, factor, sci } from "@/lib/format";

const SPECTRUM_FLOOR = 1e-6;

export function LossChart({ run, height = 280 }: { run: RunDetail; height?: number }) {
  const series = useMemo<XYSeries[]>(() => {
    const h = run.history;
    return [
      { id: "train", label: "Training MSE", color: seriesColor(0), x: h.epoch, y: h.train },
      { id: "val", label: "Validation MSE", color: seriesColor(1), x: h.epoch, y: h.val },
      { id: "test", label: "MSE against exact f", color: seriesColor(2), x: h.test_epoch.filter((e) => e > 0), y: h.test_mse.slice(h.test_epoch.length - h.test_epoch.filter((e) => e > 0).length), width: 1.5 },
    ];
  }, [run.history]);
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
    if (edge.match) out.unshift({ id: "fit", label: `Best a·${edge.match.name} + b`, color: "var(--axis)", x: edge.x, y: edge.match.fit, width: 4.5, underlay: true });
    return out;
  }, [edge]);
  const title = layers > 1 ? `Layer ${edge.layer + 1}: input ${edge.i + 1} to output ${edge.o + 1}` : `Input ${edge.i + 1} to output ${edge.o + 1}`;
  return (
    <figure className="min-w-0 rounded-md border p-3">
      <figcaption className="mb-1 flex flex-wrap justify-between gap-x-2 text-xs">
        <span className="font-medium">{title}</span>
        {edge.match && <span className="tnum text-muted-foreground">R² = {edge.match.r2 > 0.9999 ? "> 0.9999" : edge.match.r2.toFixed(4)}</span>}
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
        {edges[0]?.match ? " Here each edge is compared with the part of exp(sin πx + y²) it should have found." : ""}
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
  const rows = [{ model: `This network (${run.label})`, n: final.metrics.n_params, rmse: net, max: final.metrics.max_err, own: true }, ...final.baselines.map((b) => ({ model: b.model, n: b.n_coef, rmse: b.rmse, max: b.max_err, own: false }))];
  const best = final.baselines.reduce((a, b) => (b.rmse < a.rmse ? b : a), final.baselines[0]);
  const verdict = net == null || !best ? null
    : best.rmse < net ? `${best.model} with ${best.n_coef.toLocaleString("en-US")} coefficients beats this network by ${factor(net / best.rmse)}.`
    : `This network beats ${best.model} with ${best.n_coef.toLocaleString("en-US")} coefficients by ${factor(best.rmse / net)}.`;
  return (
    <div className="space-y-3">
      {verdict && <p className="text-sm font-medium">{verdict}</p>}
      <div className="overflow-x-auto rounded-md border">
        <table className="w-full text-[13px]">
          <caption className="sr-only">Network against classical approximators on the same training data</caption>
          <thead className="bg-wash text-left text-xs text-muted-foreground">
            <tr>
              <th scope="col" className="px-3 py-2 font-medium">Approximator</th>
              <th scope="col" className="px-3 py-2 text-right font-medium">Fitted values</th>
              <th scope="col" className="px-3 py-2 text-right font-medium">RMSE vs exact f</th>
              <th scope="col" className="px-3 py-2 text-right font-medium">Max error</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.model} className={`border-t ${r.own ? "font-medium" : ""}`}>
                <td className="px-3 py-1.5">{r.model}</td>
                <td className="tnum px-3 py-1.5 text-right">{r.n.toLocaleString("en-US")}</td>
                <td className="tnum px-3 py-1.5 text-right">{sci(r.rmse)}</td>
                <td className="tnum px-3 py-1.5 text-right">{sci(r.max)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-xs text-subtle">Least squares on the same training points. The size is the best of experiment E6's sweep no larger than the network, chosen on the validation points. Scored on the same exact grid.</p>
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

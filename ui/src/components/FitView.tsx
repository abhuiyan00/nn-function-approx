import { useMemo } from "react";
import type { Grid } from "@/lib/api";
import { HeatMap } from "./charts/HeatMap";
import { XYChart, type XYSeries } from "./charts/XYChart";

const EXACT_COLOR = "var(--axis)";
const PRED_COLOR = "var(--series-1)";

interface Props {
  grid: Grid;
  pred?: (number | null)[] | null;
  residual?: (number | null)[] | null;
  samples?: { x: number[]; y: number[] } | null;
  height?: number;
}

function Fit1D({ grid, pred, residual, samples, height = 280 }: Props & { grid: Extract<Grid, { kind: "1d" }> }) {
  const fit = useMemo<XYSeries[]>(() => {
    const series: XYSeries[] = [{ id: "exact", label: "Exact f", color: EXACT_COLOR, x: grid.x, y: grid.exact, width: 4.5, underlay: true }];
    if (pred) series.push({ id: "pred", label: "Network", color: PRED_COLOR, x: grid.x, y: pred, width: 1.75 });
    return series;
  }, [grid, pred]);
  const error = useMemo<XYSeries[]>(() => (residual ? [{ id: "res", label: "Network − exact f", color: PRED_COLOR, x: grid.x, y: residual, width: 1.5 }] : []), [grid, residual]);
  const regions = useMemo(() => {
    const [lo, hi] = grid.box;
    const out = [];
    if (grid.x[0] < lo) out.push({ x0: grid.x[0], x1: lo, label: "No training data" });
    if (grid.x[grid.x.length - 1] > hi) out.push({ x0: hi, x1: grid.x[grid.x.length - 1], label: "No training data" });
    return out;
  }, [grid]);
  const dots = useMemo(() => (samples ? { x: samples.x, y: samples.y, label: "Training samples" } : undefined), [samples]);
  const zero = useMemo(() => [{ y: 0, label: "" }], []);

  return (
    <div className="space-y-3">
      <XYChart series={fit} xLabel="x" yLabel="f(x)" height={height} regions={regions} dots={dots} ariaLabel="Network against the exact function" />
      {residual && <XYChart series={error} xLabel="x" yLabel="Error" height={150} hlines={zero} regions={regions} ariaLabel="Error of the network across x" />}
    </div>
  );
}

function Fit2D({ grid, pred, residual, samples }: Props & { grid: Extract<Grid, { kind: "2d" }> }) {
  const domain = useMemo<[number, number]>(() => {
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of grid.exact) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    return [lo, hi];
  }, [grid]);
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3">
      <HeatMap title="Exact f" x={grid.x} y={grid.y} values={grid.exact} domain={domain} dots={samples ?? undefined} />
      <HeatMap title="Network" x={grid.x} y={grid.y} values={pred ?? []} domain={domain} />
      <HeatMap title="Network − exact f" x={grid.x} y={grid.y} values={residual ?? []} kind="diverging" />
    </div>
  );
}

/** Network against the exact function: curve and error in 1-D, three surfaces in 2-D. */
export function FitView(props: Props) {
  return props.grid.kind === "1d" ? <Fit1D {...props} grid={props.grid} /> : <Fit2D {...props} grid={props.grid} />;
}

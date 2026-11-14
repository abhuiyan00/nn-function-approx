import type { Grid, RunSpec, SearchState } from "./api";
import { factor, sci } from "./format";
import { type Row, toCsv } from "./runs";

export type Range = [number, number];

// ---------------------------------------------------------------- a user's function
export const isCustom = (spec: Pick<RunSpec, "formula" | "points">) => spec.formula != null || spec.points != null;

/** Only the function: this is what the preview needs, and what makes two searches comparable. */
export function functionFields(spec: RunSpec): Partial<RunSpec> {
  const { target, formula, domain, points, gap } = spec;
  return { target, ...(formula != null && { formula }), ...(domain != null && { domain }), ...(points != null && { points }), ...(gap != null && { gap }) };
}

export const AXES = ["x", "y", "z"];

/** A formula has as many inputs as the last of x, y, z it names. */
export const formulaDim = (formula: string) => (/\bz\b/.test(formula) ? 3 : /\by\b/.test(formula) ? 2 : 1);

export function functionDim(spec: RunSpec, targets: { key: string; dim: number }[]): number {
  if (spec.formula != null) return formulaDim(spec.formula);
  if (spec.points != null) return 1;
  return targets.find((t) => t.key === spec.target)?.dim ?? 1;
}

export const resizeRanges = (ranges: Range[] | null, dim: number): Range[] =>
  Array.from({ length: dim }, (_, i) => ranges?.[i] ?? [-1, 1]);

export const boxOf = (grid: Grid): Range[] => (grid.kind === "1d" ? [grid.box] : grid.box);

const tidy = (v: number) => Number(v.toPrecision(6));

/** The middle 30 % of every range. */
export const defaultGap = (box: Range[]): Range[] =>
  box.map(([lo, hi]) => [tidy((lo + hi) / 2 - 0.15 * (hi - lo)), tidy((lo + hi) / 2 + 0.15 * (hi - lo))]);

// ----------------------------------------------------------------- measured points
/** One pair per line, split by commas, semicolons or spaces. A first line of words is a header. */
export function parsePoints(text: string): { points: Range[]; error: string | null } {
  const points: Range[] = [];
  const lines = text.split(/\r?\n/);
  for (const [i, raw] of lines.entries()) {
    const line = raw.trim();
    if (!line) continue;
    const cells = line.split(/[,;\s]+/).filter(Boolean).map(Number);
    if (cells.length === 2 && cells.every(Number.isFinite)) points.push([cells[0], cells[1]]);
    else if (!(i === 0 && cells.some(Number.isNaN))) return { points, error: `Line ${i + 1} is not a pair of numbers: ${line}` };
  }
  return { points, error: null };
}

export const pointsToText = (points: Range[]) => points.map((p) => p.join(", ")).join("\n");

export const examplePoints = (): Range[] =>
  Array.from({ length: 41 }, (_, i) => {
    const x = tidy(-2 + i * 0.1);
    return [x, Number((Math.sin(2 * x) * Math.exp((-x * x) / 6)).toFixed(4))];
  });

// --------------------------------------------------------------------- the draw pad
export const PAD_COLUMNS = 120;

/** Fill the columns between two pen positions, so that a fast stroke leaves no holes. */
export function paintColumns(columns: (number | null)[], from: { col: number; y: number }, to: { col: number; y: number }): (number | null)[] {
  const next = [...columns];
  const steps = Math.abs(to.col - from.col);
  for (let k = 0; k <= steps; k++) {
    const t = steps ? k / steps : 1;
    next[from.col + Math.sign(to.col - from.col) * k] = from.y + (to.y - from.y) * t;
  }
  return next;
}

/** The pad's x of a column, and the column of an x: both run across -1 to 1. */
export const padX = (col: number, n = PAD_COLUMNS) => tidy(-1 + (2 * col) / (n - 1));
export const padCol = (x: number, n = PAD_COLUMNS) => ((x + 1) / 2) * (n - 1);

export function columnsToPoints(columns: (number | null)[]): Range[] {
  const points: Range[] = [];
  columns.forEach((y, col) => {
    if (y != null) points.push([padX(col, columns.length), tidy(y)]);
  });
  return points;
}

export function pointsToColumns(points: Range[] | null, n = PAD_COLUMNS): (number | null)[] {
  const columns = Array<number | null>(n).fill(null);
  for (const [x, y] of points ?? []) {
    const col = Math.round(padCol(x, n));
    if (col >= 0 && col < n) columns[col] = y;
  }
  return columns;
}

/** Columns on each side of the eraser's tip, so that one click still hides a stretch. */
export const ERASER = 2;

/** The stretch an eraser stroke hid, joined with what was hidden before: there is one missing middle. */
export function eraseSpan(gap: Range[] | null | undefined, from: number, to: number, n = PAD_COLUMNS): Range[] {
  const round = (v: number) => Math.round(v * 100) / 100; // the sliders show these; the brush is wider than the rounding
  const lo = round(padX(Math.max(0, Math.min(from, to) - ERASER), n));
  const hi = round(padX(Math.min(n - 1, Math.max(from, to) + ERASER), n));
  const old = gap?.length === 1 ? gap[0] : null;
  return [[old ? Math.min(old[0], lo) : lo, old ? Math.max(old[1], hi) : hi]];
}

// ------------------------------------------------------------------- chart regions
/** Runs of missing values, where the function is undefined. All missing means measured data instead. */
export function undefinedRegions(x: number[], y: (number | null)[]): { x0: number; x1: number }[] {
  if (y.every((v) => v == null)) return [];
  const out: { x0: number; x1: number }[] = [];
  let start = -1;
  for (let i = 0; i <= y.length; i++) {
    const missing = i < y.length && y[i] == null;
    if (missing && start < 0) start = i;
    if (!missing && start >= 0) {
      out.push({ x0: x[start], x1: x[i - 1] });
      start = -1;
    }
  }
  return out;
}

export function fitRegions(grid: Extract<Grid, { kind: "1d" }>): { x0: number; x1: number; label: string }[] {
  const [lo, hi] = grid.box;
  const out = undefinedRegions(grid.x, grid.exact).map((r) => ({ ...r, label: "Undefined" }));
  if (grid.gap) out.push({ x0: grid.gap[0][0], x1: grid.gap[0][1], label: "Missing middle" });
  if (grid.x[0] < lo) out.push({ x0: grid.x[0], x1: lo, label: "No training data" });
  if (grid.x[grid.x.length - 1] > hi) out.push({ x0: hi, x1: grid.x[grid.x.length - 1], label: "No training data" });
  return out;
}

// ------------------------------------------------------------------- search results
export function curvesCsv(state: SearchState): string {
  const { grid, known } = state;
  const curves = state.rows.filter((r) => r.pred);
  const inputs = grid.kind === "1d" ? ["x"] : ["x", "y"];
  const columns = [...inputs, ...(known ? ["exact"] : []), ...curves.map((r) => r.key)];
  const rows = grid.exact.map((exact, i) => {
    const place: Row = grid.kind === "1d" ? { x: grid.x[i] } : { x: grid.x[Math.floor(i / grid.y.length)], y: grid.y[i % grid.y.length] };
    return { ...place, exact, ...Object.fromEntries(curves.map((r) => [r.key, r.pred![i]])) };
  });
  return toCsv(columns, rows);
}

/** What the ranking means, in plain words. */
export function describeSearch(state: SearchState): string[] {
  const lines: string[] = [];
  const network = state.rows.find((r) => r.key === state.best_network);
  const winner = state.rows.find((r) => r.key === state.winner);
  const truth = state.known ? "the exact function" : "your points";
  if (network?.rmse_gap != null && network.rmse_seen != null) {
    lines.push(`${network.key} misses ${truth} by ${sci(network.rmse_seen)} where it had data, and by ${sci(network.rmse_gap)} inside the missing middle.`);
  }
  if (winner && network?.rmse && winner.rmse && winner.kind === "classical") {
    lines.push(winner.key.startsWith("RBF")
      ? `An RBF network solved by least squares gets closer than every network trained step by step, by ${factor(network.rmse / winner.rmse)}.`
      : `${winner.key} gets closer than any network here, by ${factor(network.rmse / winner.rmse)}. For this function a network is not the best tool.`);
  }
  if (state.blind_pick && state.blind_pick !== state.winner && state.rows.some((r) => r.rmse_gap != null)) {
    lines.push(`Judged by its validation points alone, a user would pick ${state.blind_pick}: those points say nothing about the missing middle.`);
  }
  if (!state.known) {
    lines.push("Measured data has no exact function: where you gave no points, nothing can be scored, so each curve is a guess there.");
  }
  return lines;
}

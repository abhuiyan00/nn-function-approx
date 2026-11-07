import type { RunSpec } from "./api";

export const MAX_PARAMS_LM = 1500; // same limit as funcprox_ui.schemas

/** CLI defaults of ``python -m funcprox train``: a command lists only what differs. */
export const defaultSpec: RunSpec = {
  target: "f1",
  arch: "mlp",
  hidden: [32, 32],
  act: "tanh",
  bias_init: null,
  fourier_m: 0,
  fourier_scale: 2,
  w0: 30,
  widths: null,
  grid: 5,
  grid_updates: [],
  optimizer: "adam",
  lr: 1e-3,
  batch: 64,
  epochs: 300,
  patience: 50,
  l2: 0,
  n_train: 2000,
  n_val: 500,
  sigma: 0,
  seed: 1,
  data_seed: 0,
};

/** "16,16", "8 16" or the shape "3x64"; null when it is not a list of positive integers. */
export function parseIntList(text: string): number[] | null {
  const trimmed = text.trim();
  const shape = trimmed.match(/^(\d+)\s*[x×]\s*(\d+)$/i);
  const values = shape
    ? Array<number>(Number(shape[1])).fill(Number(shape[2]))
    : trimmed.split(/[\s,]+/).filter(Boolean).map(Number);
  return values.length > 0 && values.every((v) => Number.isInteger(v) && v > 0) ? values : null;
}

const pairs = (w: number[]) => w.slice(0, -1).map((a, i) => [a, w[i + 1]] as const);

/** Mirrors mlp_param_count / kan_param_count in funcprox.model; the server reports the real count. */
export function paramCount(spec: RunSpec, dim: number): number {
  if (spec.arch === "kan") {
    const widths = spec.widths ?? [dim, 5, 1];
    return pairs(widths).reduce((n, [a, b]) => n + a * b * (spec.grid + 3) + a * b + b, 0);
  }
  const first = spec.fourier_m ? 2 * spec.fourier_m : dim;
  return pairs([first, ...spec.hidden, 1]).reduce((n, [a, b]) => n + a * b + b, 0);
}

export type SpecErrors = Partial<Record<keyof RunSpec, string>>;

export function validateSpec(spec: RunSpec, dim: number): SpecErrors {
  const errors: SpecErrors = {};
  if (spec.arch === "kan") {
    const w = spec.widths ?? [dim, 5, 1];
    if (w[0] !== dim) errors.widths = `KAN widths must start with ${dim}, the input size of this target.`;
    else if (w[w.length - 1] !== 1) errors.widths = "KAN widths must end with 1: the engine has scalar output.";
  } else if (spec.hidden.length > 6) {
    errors.hidden = "At most 6 hidden layers.";
  }
  const n = paramCount(spec, dim);
  if (spec.optimizer === "lm" && n > MAX_PARAMS_LM) {
    const field = spec.arch === "kan" ? "widths" : "hidden";
    errors[field] = `Levenberg-Marquardt solves a P × P system each iteration: ${n.toLocaleString("en-US")} parameters exceed its limit of ${MAX_PARAMS_LM.toLocaleString("en-US")}.`;
  }
  return errors;
}

export function cliCommand(spec: RunSpec): string {
  const d = defaultSpec;
  const parts = ["python -m funcprox train"];
  const add = (flag: string, value: unknown) => parts.push(`--${flag} ${value}`);
  add("target", spec.target);
  if (spec.arch === "kan") {
    add("arch", "kan");
    if (spec.widths) add("widths", spec.widths.join(","));
    if (spec.grid !== d.grid) add("grid", spec.grid);
    if (spec.grid_updates.length) add("grid-updates", spec.grid_updates.join(","));
  } else {
    if (spec.hidden.join() !== d.hidden.join()) add("hidden", spec.hidden.join(","));
    if (spec.act !== d.act) add("act", spec.act);
    if (spec.bias_init) add("bias-init", spec.bias_init);
    if (spec.fourier_m) {
      add("fourier-m", spec.fourier_m);
      if (spec.fourier_scale !== d.fourier_scale) add("fourier-scale", spec.fourier_scale);
    }
    if (spec.act === "sine" && spec.w0 !== d.w0) add("w0", spec.w0);
  }
  if (spec.optimizer !== d.optimizer) add("optimizer", spec.optimizer);
  if (spec.lr !== d.lr) add("lr", spec.lr);
  if (spec.batch !== d.batch) add("batch", spec.batch);
  if (spec.epochs !== d.epochs) add("epochs", spec.epochs);
  if (spec.patience !== d.patience) add("patience", spec.patience);
  if (spec.l2 !== d.l2) add("l2", spec.l2);
  if (spec.n_train !== d.n_train) add("n-train", spec.n_train);
  if (spec.n_val !== d.n_val) add("n-val", spec.n_val);
  if (spec.sigma !== d.sigma) add("sigma", spec.sigma);
  if (spec.seed !== d.seed) add("seed", spec.seed);
  if (spec.data_seed !== d.data_seed) add("data-seed", spec.data_seed);
  return parts.join(" ");
}

// ---------------------------------------------------------------------- result rows
export type Row = Record<string, string | number | null>;

export interface AggregatePoint {
  x: number;
  y: number;
  lo: number;
  hi: number;
  n: number;
}

export interface AggregateSeries {
  key: string;
  points: AggregatePoint[];
}

function median(sorted: number[]): number {
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/** Median and min-max per x within each colour group: seeds become one point and a band. */
export function aggregate(
  rows: Row[],
  x: string,
  y: string,
  color?: string,
  options: { positiveY?: boolean; positiveX?: boolean; order?: string[] } = {},
): AggregateSeries[] {
  const groups = new Map<string, Map<number, number[]>>();
  for (const row of rows) {
    const xv = row[x];
    const yv = row[y];
    if (typeof xv !== "number" || typeof yv !== "number") continue;
    if ((options.positiveX && xv <= 0) || (options.positiveY && yv <= 0)) continue;
    const key = color ? String(row[color] ?? "n/a") : "";
    if (!groups.has(key)) groups.set(key, new Map());
    const byX = groups.get(key)!;
    if (!byX.has(xv)) byX.set(xv, []);
    byX.get(xv)!.push(yv);
  }
  const keys = options.order ? options.order.filter((k) => groups.has(k)) : [...groups.keys()];
  return keys.map((key) => ({
    key,
    points: [...groups.get(key)!.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([px, values]) => {
        const sorted = [...values].sort((a, b) => a - b);
        return { x: px, y: median(sorted), lo: sorted[0], hi: sorted[sorted.length - 1], n: sorted.length };
      }),
  }));
}

export function medianOf(values: number[]): number {
  return median([...values].sort((a, b) => a - b));
}

export function toCsv(columns: string[], rows: Row[]): string {
  const cell = (v: string | number | null | undefined) => {
    const s = v == null ? "" : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return [columns.join(","), ...rows.map((r) => columns.map((c) => cell(r[c])).join(","))].join("\n") + "\n";
}

export function downloadText(filename: string, text: string, type = "text/csv"): void {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

import type { Row } from "./runs";

export type Optimizer = "adam" | "sgd" | "momentum" | "nesterov" | "lm";
export type Activation = "tanh" | "sigmoid" | "relu" | "silu" | "sine";

export interface RunSpec {
  target: string;
  formula: string | null;
  domain: [number, number][] | null;
  points: [number, number][] | null;
  /** Null keeps the target's own missing middle, empty removes it. */
  gap: [number, number][] | null;
  arch: "mlp" | "kan";
  hidden: number[];
  act: Activation;
  bias_init: "zero" | "kink" | null;
  fourier_m: number;
  fourier_scale: number;
  w0: number;
  widths: number[] | null;
  grid: number;
  grid_updates: number[];
  optimizer: Optimizer;
  lr: number;
  batch: number;
  epochs: number;
  patience: number;
  l2: number;
  n_train: number;
  n_val: number;
  sigma: number;
  seed: number;
  data_seed: number;
}

interface GridCommon {
  /** Null where the function is undefined; all null for measured data. */
  exact: (number | null)[];
  gap?: [number, number][] | null;
  /** False for measured data: there is no exact function behind it. */
  known?: boolean;
}

export type Grid =
  | (GridCommon & { kind: "1d"; x: number[]; box: [number, number]; spectrum_exact?: (number | null)[] | null })
  | (GridCommon & { kind: "2d"; x: number[]; y: number[]; box: [number, number][]; spectrum_exact?: null });

export interface FunctionPreview {
  key: string;
  formula: string;
  dim: number;
  grid: Grid;
  samples: { x: number[]; y: number[] };
}

export interface SearchRow {
  key: string;
  /** Colour slot: fixed per model, whatever its rank. */
  slot: number;
  kind: "network" | "classical";
  status: RunStatus;
  seeds: number;
  n_params: number;
  best_run: string | null;
  rmse?: number | null;
  rmse_lo?: number | null;
  rmse_hi?: number | null;
  rmse_gap?: number | null;
  rmse_seen?: number | null;
  max_err?: number | null;
  val_mse?: number | null;
  seconds?: number | null;
  pred?: (number | null)[] | null;
  lo?: (number | null)[] | null;
  hi?: (number | null)[] | null;
}

export interface SearchState {
  id: string;
  status: "running" | "done" | "stopped";
  done: number;
  total: number;
  formula: string;
  dim: number;
  known: boolean;
  grid: Grid;
  samples: { x: number[]; y: number[] };
  rows: SearchRow[];
  winner: string | null;
  best_network: string | null;
  blind_pick: string | null;
}

export interface Target {
  key: string;
  name: string;
  formula: string;
  dim: number;
  probes: string;
  domain: [number, number][];
  frequencies: number[];
  preview: Grid;
}

export type RunStatus = "queued" | "running" | "done" | "stopped" | "failed";

export interface RunSummary {
  id: string;
  status: RunStatus;
  label: string;
  spec: RunSpec;
  n_params: number;
  created: number;
  error: string | null;
  saved_as: string | null;
  rmse: number | null;
  max_err: number | null;
  r2: number | null;
  seconds: number | null;
  epochs_run: number;
  best_epoch: number | null;
  stop: string | null;
}

export interface KanEdge {
  layer: number;
  i: number;
  o: number;
  x: number[];
  y: (number | null)[];
  match?: { name: string; r2: number; fit: (number | null)[] };
}

export interface Snapshot {
  epoch: number;
  pred: (number | null)[];
  residual: (number | null)[];
  rmse: number | null;
  rmse_gap?: number | null;
  max_err: number | null;
  r2: number | null;
  spectrum: { pred: (number | null)[] } | null;
  edges: KanEdge[] | null;
}

export interface Baseline {
  model: string;
  n_coef: number;
  rmse: number;
  max_err: number;
  rmse_gap?: number | null;
  rmse_seen?: number | null;
}

export interface RunFinal {
  metrics: {
    rmse: number | null;
    rmse_gap?: number | null;
    rmse_seen?: number | null;
    max_err_gap?: number | null;
    mae: number | null;
    max_err: number | null;
    r2: number | null;
    nrmse: number | null;
    rmse_edge?: number | null;
    rmse_interior?: number | null;
    argmax_in_edge?: boolean;
    n_params: number;
  };
  best_epoch: number;
  epochs_run: number;
  stop: string;
  seconds: number;
  pred: (number | null)[];
  residual: (number | null)[];
  spectrum: { pred: (number | null)[] } | null;
  edges: KanEdge[] | null;
  baselines: Baseline[];
}

export interface RunHistory {
  epoch: number[];
  train: (number | null)[];
  val: (number | null)[];
  wall: number[];
  updates: number[];
  mu: (number | null)[];
  test_epoch: number[];
  test_mse: number[];
}

export interface RunDetail extends RunSummary {
  architecture: string;
  formula: string;
  grid: Grid;
  samples: { x: number[]; y: number[] };
  history: RunHistory;
  components: { k: number[]; epoch: number[]; rel: number[][] } | null;
  live: Snapshot | null;
  final: RunFinal | null;
}

export interface RunFrame {
  history?: Omit<RunHistory, "test_epoch" | "test_mse">;
  test?: { epoch: number[]; mse: number[] };
  components?: { epoch: number[]; rel: number[][] };
  live?: Snapshot;
  status?: RunStatus;
}

export interface ExperimentSummary {
  id: string;
  script: string;
  title: string;
  group: "sensitivity" | "unique";
  question: string;
  hypothesis: string;
  owner: "A" | "B" | null;
  profile: string | null;
  generated: string | null;
}

export interface ExperimentDetail extends ExperimentSummary {
  protocol: string;
  findings: string;
  summary: string;
  figures: { name: string; url: string }[];
  datasets: { name: string; rows: number }[];
}

export interface DatasetColumn {
  name: string;
  type: "number" | "string";
  unique: number;
  constant: boolean;
  values: (string | number)[] | null;
}

export interface Dataset {
  experiment: string;
  name: string;
  columns: DatasetColumn[];
  rows: Row[];
}

export interface Verdict {
  target: string;
  formula: string;
  winner: "network" | "classical" | "tie";
  ratio: number;
  orders: number;
  network: { model: string; rmse: number; n_params: number };
  classical: { model: string; rmse: number; n_params: number };
}

export interface TimelineItem {
  label: string;
  start: string;
  end: string;
  milestone: boolean;
  done: boolean;
}

export interface Overview {
  profiles: Record<string, number>;
  generated: string | null;
  run_count: number;
  highlights: {
    engine?: { worst_error: number; layers: number };
    lm_vs_adam?: Record<string, number>;
    kan?: { rmse: number; n_params: number; best_mlp: number };
    spectral?: { tanh_learned_at: number | null; fourier_learned_at: number | null; tanh_rmse: number; fourier_rmse: number; steps: number };
    conformal?: { nominal: number; coverage: number };
  };
  verdicts: Verdict[];
  timeline: TimelineItem[];
  deadline: string | null;
}

export interface SavedModel {
  name: string;
  size: number;
  modified: number;
  error?: string;
  target?: string | null;
  arch?: string | null;
  label?: string | null;
  rmse?: number | null;
  max_err?: number | null;
  r2?: number | null;
  n_params?: number | null;
}

export interface ModelDetail extends SavedModel {
  formula?: string;
  dim?: number;
  grid: Grid | null;
  pred: (number | null)[] | null;
  residual: (number | null)[] | null;
}

export interface Prediction {
  y_hat: number[];
  exact: number[] | null;
  abs_error: number[] | null;
  outside_box: boolean[];
  in_gap: boolean[];
}

export type JobStatus = "queued" | "running" | "done" | "failed" | "cancelled";

export interface Job {
  id: string;
  experiment: string;
  profile: "quick" | "full";
  workers: number;
  status: JobStatus;
  returncode: number | null;
  created: number;
  started: number | null;
  finished: number | null;
  last_line: string | null;
  log?: string[];
}

export interface Health {
  status: string;
  version: string;
  python: string;
  numpy: string;
  cpus: number;
}

export interface DocumentInfo {
  slug: string;
  title: string;
  path: string;
}

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

function errorMessage(detail: unknown, fallback: string): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail) && detail[0]?.msg) {
    const first = detail[0];
    const field = (first.loc ?? []).filter((p: unknown) => p !== "body").join(".");
    const msg = String(first.msg).replace(/^Value error, /, "");
    return field ? `${field}: ${msg}` : msg;
  }
  return fallback;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, init);
  } catch {
    throw new ApiError("The funcprox server is not responding. Is `python -m funcprox ui` still running?", 0);
  }
  if (response.status === 204) return undefined as T;
  const body = await response.json().catch(() => null);
  if (!response.ok) throw new ApiError(errorMessage(body?.detail, `Request failed (${response.status})`), response.status);
  return body as T;
}

const post = <T>(path: string, body?: unknown) =>
  request<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });

export const api = {
  health: () => request<Health>("/api/health"),
  overview: () => request<Overview>("/api/overview"),
  targets: () => request<Target[]>("/api/targets"),
  experiments: () => request<ExperimentSummary[]>("/api/experiments"),
  experiment: (id: string) => request<ExperimentDetail>(`/api/experiments/${id}`),
  datasets: () => request<{ experiment: string; name: string; rows: number }[]>("/api/datasets"),
  dataset: (exp: string, name: string) => request<Dataset>(`/api/datasets/${exp}/${name}`),
  documents: () => request<DocumentInfo[]>("/api/documents"),
  document: (slug: string) => request<{ slug: string; path: string; markdown: string }>(`/api/documents/${slug}`),
  runs: () => request<RunSummary[]>("/api/lab/runs"),
  run: (id: string) => request<RunDetail>(`/api/lab/runs/${id}`),
  startRun: (spec: RunSpec) => post<RunSummary>("/api/lab/runs", spec),
  stopRun: (id: string) => post<RunSummary>(`/api/lab/runs/${id}/stop`),
  deleteRun: (id: string) => request<void>(`/api/lab/runs/${id}`, { method: "DELETE" }),
  saveRun: (id: string, name: string) => post<{ name: string; path: string }>(`/api/lab/runs/${id}/save`, { name }),
  predictRun: (id: string, inputs: number[][]) => post<Prediction>(`/api/lab/runs/${id}/predict`, { inputs }),
  previewFunction: (fields: Partial<RunSpec>) => post<FunctionPreview>("/api/functions/preview", fields),
  startSearch: (body: { base: RunSpec; effort: "quick" | "thorough"; seeds: number }) => post<SearchState>("/api/search", body),
  search: (id: string) => request<SearchState>(`/api/search/${id}`),
  stopSearch: (id: string) => post<SearchState>(`/api/search/${id}/stop`),
  models: () => request<SavedModel[]>("/api/models"),
  model: (name: string) => request<ModelDetail>(`/api/models/${name}`),
  predict: (name: string, inputs: number[][]) => post<Prediction>(`/api/models/${name}/predict`, { inputs }),
  jobs: () => request<Job[]>("/api/jobs"),
  job: (id: string) => request<Job>(`/api/jobs/${id}`),
  startJob: (experiment: string, profile: string, workers: number) => post<Job>("/api/jobs", { experiment, profile, workers }),
  cancelJob: (id: string) => post<Job>(`/api/jobs/${id}/cancel`),
};

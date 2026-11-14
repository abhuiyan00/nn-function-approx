import type { RunSpec, SweepParam, SweepState } from "./api";
import { factor } from "./format";

export interface Setting {
  label: string;
  values: number[];
  /** What the experiments found about this setting. */
  hint: string;
  only?: "mlp" | "kan";
  /** Levenberg-Marquardt has no learning rate and no batches. */
  firstOrder?: boolean;
  /** Measured points come as they are: their number and noise cannot be changed. */
  needsFormula?: boolean;
}

export const SETTINGS: Record<SweepParam, Setting> = {
  lr: { label: "Learning rate", values: [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2], hint: "E3 found a broad stable band, then divergence.", firstOrder: true },
  batch: { label: "Batch size", values: [16, 32, 64, 128, 256], hint: "E3: with a fixed number of epochs a large batch takes too few steps.", firstOrder: true },
  width: { label: "Units per hidden layer", values: [4, 8, 16, 32, 64], hint: "E1: capacity against the same data.", only: "mlp" },
  depth: { label: "Hidden layers", values: [1, 2, 3, 4], hint: "E1: depth against width.", only: "mlp" },
  grid: { label: "KAN grid intervals", values: [3, 5, 8, 12, 16], hint: "U2: a finer grid resolves more detail and has more to fit.", only: "kan" },
  n_train: { label: "Training points", values: [50, 100, 200, 500, 1000, 2000], hint: "E4: the error falls with more data.", needsFormula: true },
  sigma: { label: "Label noise σ", values: [0, 0.05, 0.1, 0.2, 0.4], hint: "E4: noise bounds the error from below.", needsFormula: true },
  l2: { label: "Weight decay", values: [0, 1e-6, 1e-5, 1e-4, 1e-3], hint: "E5: moderate decay helps, strong decay underfits." },
  patience: { label: "Early-stopping patience", values: [5, 10, 25, 50, 100], hint: "E5: too little patience stops on a plateau." },
  epochs: { label: "Epochs or iterations", values: [50, 100, 200, 400, 800], hint: "The training budget." },
};

/** The settings that mean something for this run. */
export function settingsFor(base: Pick<RunSpec, "arch" | "optimizer" | "points">): SweepParam[] {
  return (Object.keys(SETTINGS) as SweepParam[]).filter((key) => {
    const s = SETTINGS[key];
    return (!s.only || s.only === base.arch) && !(s.firstOrder && base.optimizer === "lm") && !(s.needsFormula && base.points != null);
  });
}

/** A value as the form and the axis write it: 0.0001 becomes 1e-4. */
export const showValue = (v: number) => (v !== 0 && (Math.abs(v) < 1e-2 || Math.abs(v) >= 1e5) ? v.toExponential() : String(v));

/** 2 to 12 different numbers, split by commas, semicolons or spaces; null otherwise. */
export function parseValues(text: string): number[] | null {
  const values = text.split(/[\s,;]+/).filter(Boolean).map(Number);
  const fine = values.length >= 2 && values.length <= 12 && values.every(Number.isFinite) && new Set(values).size === values.length;
  return fine ? values : null;
}

/** What the numbers say, by the rule of the project plan: an effect counts only when it exceeds the seed spread. */
export function sweepVerdict(state: Pick<SweepState, "effect" | "spread" | "param" | "best" | "rows">): string | null {
  if (state.effect == null || state.spread == null || state.best == null) return null;
  const name = SETTINGS[state.param].label.toLowerCase();
  const head = `Across these values the error changes by ${factor(state.effect)}.`;
  if (state.rows[0]?.seeds === 1) return `${head} One seed gives no spread to compare with: use three or more before reading an effect into this.`;
  const seeds = `Between seeds at the best value it changes by ${factor(state.spread)}.`;
  return state.effect > 1.5 * state.spread
    ? `${head} ${seeds} The ${name} matters more than the starting weights here; best at ${showValue(state.best)}.`
    : `${head} ${seeds} That is within the luck of the starting weights: this study does not show that the ${name} matters.`;
}

/** Interactive chart of each experiment, described over its result files. */
export interface ChartSpec {
  id: string;
  title: string;
  note?: string;
  dataset: string;
  kind: "line" | "dot" | "heat";
  x: string;
  y: string;
  /** Column that separates the series. */
  color?: string;
  /** Column shown as a selector above the chart. */
  facet?: string;
  where?: Record<string, (string | number)[]>;
  /** Plot several columns as one measure: long form with columns 'measure' and 'value'. */
  melt?: string[];
  xScale?: "linear" | "log" | "ordinal";
  yScale?: "linear" | "log";
  xLabel: string;
  yLabel?: string;
  /** Display names: category values of the colour, facet and heat axes. */
  names?: Record<string, string>;
  colorPrefix?: string;
  reference?: { value: number; label: string };
}

const TEST_RMSE = "Test RMSE vs exact f";
const TARGET_NAMES = { f1: "f1 · sin 2πx", f4: "f4 · Runge", f5: "f5 · Franke", f6: "f6 · |x|", f7: "f7 · exp(sin πx + y²)" };
const HOLE_NAMES = { m1: "m1 · sine", m2: "m2 · hidden peak", m3: "m3 · hidden kink", m4: "m4 · cubic", m5: "m5 · surface" };

export const EXPERIMENT_CHARTS: Record<string, ChartSpec[]> = {
  e0: [{
    id: "gradcheck", title: "Analytic derivatives against finite differences", note: "Each dot is one check of one layer type. The test suite fails above 1e-6.",
    dataset: "gradcheck", kind: "dot", x: "error", y: "layer", color: "check", xScale: "log", xLabel: "Largest relative finite-difference error",
    reference: { value: 1e-6, label: "test limit 1e-6" },
  }],
  e1: [{
    id: "params", title: "Depth against width at equal parameter count", note: "Median over seeds; the band spans seed minimum to maximum.",
    dataset: "runs", kind: "line", x: "n_params", y: "rmse", color: "depth", facet: "target", xScale: "log", yScale: "log",
    xLabel: "Parameters", yLabel: TEST_RMSE, colorPrefix: "Hidden layers: ", names: TARGET_NAMES,
  }],
  e2: [{
    id: "variants", title: "Activation and initialisation", note: "One dot per seed; the tick marks the median.",
    dataset: "runs", kind: "dot", x: "rmse", y: "variant", facet: "target", xScale: "log", xLabel: TEST_RMSE, names: TARGET_NAMES,
  }],
  e3: [{
    id: "lr-batch", title: "Learning rate against batch size", note: "Median test RMSE over seeds, Runge function, Adam.",
    dataset: "lr_batch", kind: "heat", x: "lr", y: "batch", xLabel: "Learning rate", yLabel: "Batch size", names: { "0": "full" },
  }, {
    id: "optimizers", title: "Optimisers on the same network", note: "2 × 16 tanh on the Runge function; one dot per learning rate tried.",
    dataset: "optimizers", kind: "dot", x: "rmse", y: "optimizer", xScale: "log", xLabel: TEST_RMSE,
  }],
  e4: [{
    id: "n-sigma", title: "Error against data size and noise", note: "Equal update budget for every N. Noise σ is in units of the target.",
    dataset: "runs", kind: "line", x: "n_train", y: "rmse", color: "sigma", facet: "lr_schedule", xScale: "log", yScale: "log",
    xLabel: "Training points N", yLabel: TEST_RMSE, colorPrefix: "σ = ",
    names: { constant: "Adam, constant lr", cosine: "Adam, cosine lr", lm: "Levenberg-Marquardt" },
  }],
  e5: [{
    id: "patience", title: "Early-stopping patience", note: "f1, N = 250, σ = 0.1, a 2 × 128 network with 68× more parameters than points.",
    dataset: "runs", kind: "line", x: "patience", y: "rmse", where: { kind: ["patience"] }, xScale: "ordinal", yScale: "log", xLabel: "Patience (epochs)", yLabel: TEST_RMSE,
  }, {
    id: "l2", title: "L2 weight decay", note: "Same setting as the patience sweep.",
    dataset: "runs", kind: "line", x: "l2", y: "rmse", where: { kind: ["l2"] }, xScale: "ordinal", yScale: "log", xLabel: "Weight decay λ", yLabel: TEST_RMSE,
  }],
  e6: [{
    id: "scaling", title: "Networks against classical approximators at equal size", note: "Network curves are medians over seeds. Classical fits are deterministic.",
    dataset: "runs", kind: "line", x: "n_params", y: "rmse", color: "model", facet: "target", xScale: "log", yScale: "log",
    xLabel: "Parameters or coefficients", yLabel: TEST_RMSE, names: TARGET_NAMES,
  }],
  e7: [{
    id: "hole-clean", title: "Error inside the hole against its width, no noise", note: "Median over seeds. No training or validation point falls inside the hole.",
    dataset: "runs", kind: "line", x: "gap_pct", y: "rmse_gap", color: "model", facet: "target", where: { sigma: [0] }, yScale: "log",
    xLabel: "Hole width, % of each axis", yLabel: "RMSE inside the hole", names: HOLE_NAMES,
  }, {
    id: "hole-noisy", title: "The same with noise σ = 0.05", note: "With noise, high-order fits lose their exact recovery of the hole.",
    dataset: "runs", kind: "line", x: "gap_pct", y: "rmse_gap", color: "model", facet: "target", where: { sigma: [0.05] }, yScale: "log",
    xLabel: "Hole width, % of each axis", yLabel: "RMSE inside the hole", names: HOLE_NAMES,
  }],
  u1: [{
    id: "methods", title: "The same network with three training methods", note: "2 × 16 tanh. One dot per seed; the tick marks the median.",
    dataset: "runs", kind: "dot", x: "rmse", y: "method", facet: "target", xScale: "log", xLabel: TEST_RMSE, names: TARGET_NAMES,
  }],
  u2: [{
    id: "params", title: "KAN against MLP at equal size", note: "Both trained with Levenberg-Marquardt. KAN points vary the grid size G.",
    dataset: "runs", kind: "line", x: "n_params", y: "rmse", color: "model", facet: "target", xScale: "log", yScale: "log",
    xLabel: "Parameters", yLabel: TEST_RMSE, names: TARGET_NAMES,
  }],
  u3: [{
    id: "components", title: "When each frequency of f3 is learned", note: "Relative error of the amplitude of sin(kπx). A component counts as learned below 0.1.",
    dataset: "curves", kind: "line", x: "step", y: "rel_error", color: "k", facet: "model", yScale: "log", xLabel: "Training step", yLabel: "Relative amplitude error",
    reference: { value: 0.1, label: "learned below 0.1" },
  }],
  u4: [{
    id: "coverage", title: "Conformal coverage against the promise", note: "Mean over 200 random calibration splits. On target means on the diagonal.",
    dataset: "coverage", kind: "line", x: "nominal", y: "coverage", color: "score", xLabel: "Nominal coverage", yLabel: "Measured coverage",
  }, {
    id: "width", title: "Interval width", note: "The weighted score spends width where the noise is, and is narrower at high coverage.",
    dataset: "coverage", kind: "line", x: "nominal", y: "width", color: "score", xLabel: "Nominal coverage", yLabel: "Mean interval width",
  }, {
    id: "ensembles", title: "Ensemble spread against real error", note: "Outside the training box the spread grows, yet stays far below the error.",
    dataset: "ensembles", kind: "dot", x: "value", y: "region", color: "measure", facet: "activation", melt: ["rmse", "spread"], xScale: "log", xLabel: "RMSE or ensemble spread",
    names: { rmse: "Real error (RMSE)", spread: "Ensemble spread" },
  }],
};

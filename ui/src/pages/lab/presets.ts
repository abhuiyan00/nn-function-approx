import type { RunSpec } from "@/lib/api";
import { defaultSpec } from "@/lib/runs";

export interface Preset {
  id: string;
  label: string;
  source: string;
  hint: string;
  spec: RunSpec;
}

const make = (id: string, label: string, source: string, hint: string, spec: Partial<RunSpec>): Preset => ({
  id, label, source, hint, spec: { ...defaultSpec, ...spec },
});

const SPECTRAL = { target: "f3", optimizer: "adam", lr: 1e-3, batch: 0, epochs: 3000, patience: 0, n_train: 1024, n_val: 256 } as const;

/** Each setting comes from the experiment named in ``source``. */
export const PRESETS: Preset[] = [
  make("adam-runge", "Adam on Runge", "U1", "The baseline for the next preset: same network, first-order training.",
    { target: "f4", hidden: [16, 16], optimizer: "adam", epochs: 600, patience: 50 }),
  make("lm-runge", "LM on Runge", "U1", "Same network, Levenberg-Marquardt: compare the final RMSE with Adam.",
    { target: "f4", hidden: [16, 16], optimizer: "lm", epochs: 300, patience: 30 }),
  make("spectral-tanh", "Spectral bias", "U3", "Watch the fit: the tanh network learns the slow waves and misses sin 16πx.",
    { ...SPECTRAL, hidden: [64, 64, 64] }),
  make("spectral-fourier", "Fourier features", "U3", "The same target with an input encoding: all frequencies arrive at once.",
    { ...SPECTRAL, hidden: [64, 64], fourier_m: 32, fourier_scale: 4 }),
  make("kan-f7", "KAN reads the formula", "U2", "The Edges panel below shows it: the learned edges match sin(πx), y² and exp(·).",
    { target: "f7", arch: "kan", widths: [2, 1, 1], grid: 10, grid_updates: [25, 75], optimizer: "lm", epochs: 300, patience: 30 }),
  make("relu-kink", "ReLU on |x|", "E2", "A ReLU network is piecewise linear, so it can represent the kink exactly.",
    { target: "f6", hidden: [32, 32], act: "relu", bias_init: "kink", epochs: 600 }),
  make("overfit", "Overfitting", "E4", "Few noisy points and a large network: past its minimum the error against exact f doubles, while training MSE keeps falling.",
    { target: "f1", hidden: [128, 128], n_train: 100, n_val: 25, sigma: 0.2, epochs: 6000, patience: 0 }),
  make("franke", "Franke in 2-D", "E6", "A 2-D benchmark where networks hold their own against classical methods.",
    { target: "f5", hidden: [16, 16], optimizer: "lm", epochs: 300, patience: 30 }),
];

import { scaleLinear, scaleLog } from "d3-scale";
import { useLayoutEffect, useRef, useState } from "react";

export type AxisKind = "linear" | "log";

/** Colours follow the entity: slots are assigned once, in fixed order, and never cycled. */
export const seriesColor = (index: number) => `var(--series-${(index % 8) + 1})`;

export const MARKERS = ["circle", "square", "diamond", "triangle", "triangle-down", "pentagon", "hexagon", "star"] as const;
export type Marker = (typeof MARKERS)[number];

/** Marker outline centred on 0,0: shape is the second channel next to hue, for colour-blind readers. */
export function markerPath(marker: Marker, r: number): string {
  const poly = (points: [number, number][]) => `M${points.map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join("L")}Z`;
  const regular = (n: number, start = -Math.PI / 2) =>
    poly(Array.from({ length: n }, (_, i) => {
      const a = start + (i * 2 * Math.PI) / n;
      return [r * 1.12 * Math.cos(a), r * 1.12 * Math.sin(a)] as [number, number];
    }));
  switch (marker) {
    case "square":
      return poly([[-r * 0.86, -r * 0.86], [r * 0.86, -r * 0.86], [r * 0.86, r * 0.86], [-r * 0.86, r * 0.86]]);
    case "diamond":
      return poly([[0, -r * 1.2], [r * 1.2, 0], [0, r * 1.2], [-r * 1.2, 0]]);
    case "triangle":
      return regular(3);
    case "triangle-down":
      return regular(3, Math.PI / 2);
    case "pentagon":
      return regular(5);
    case "hexagon":
      return regular(6);
    case "star":
      return poly(Array.from({ length: 10 }, (_, i) => {
        const a = -Math.PI / 2 + (i * Math.PI) / 5;
        const R = i % 2 ? r * 0.55 : r * 1.25;
        return [R * Math.cos(a), R * Math.sin(a)] as [number, number];
      }));
    default:
      return `M${-r},0a${r},${r} 0 1,0 ${2 * r},0a${r},${r} 0 1,0 ${-2 * r},0Z`;
  }
}

export function useSize<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(0);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    setWidth(el.clientWidth);
    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width));
    observer.observe(el);
    return () => observer.disconnect();
  }, []);
  return { ref, width };
}

export function makeScale(kind: AxisKind, domain: [number, number], range: [number, number]) {
  return kind === "log"
    ? scaleLog().domain(domain).range(range).clamp(false)
    : scaleLinear().domain(domain).range(range);
}

function finite(values: Iterable<number | null | undefined>, kind: AxisKind): number[] {
  const out: number[] = [];
  for (const v of values) if (v != null && Number.isFinite(v) && (kind === "linear" || v > 0)) out.push(v);
  return out;
}

export function domainOf(values: Iterable<number | null | undefined>, kind: AxisKind, pad = 0.04): [number, number] {
  const v = finite(values, kind);
  if (!v.length) return kind === "log" ? [1e-3, 1] : [0, 1];
  let lo = Infinity;
  let hi = -Infinity;
  for (const value of v) {
    if (value < lo) lo = value;
    if (value > hi) hi = value;
  }
  if (kind === "log") {
    if (lo === hi) return [lo / 2, hi * 2];
    return [10 ** (Math.log10(lo) - 0.04), 10 ** (Math.log10(hi) + 0.04)];
  }
  if (lo === hi) return [lo - 1, hi + 1];
  const span = (hi - lo) * pad;
  return [lo - span, hi + span];
}

/** Decade ticks on log axes, thinned to fit; ordinary round ticks on linear axes. */
export function ticksOf(kind: AxisKind, domain: [number, number], length: number): number[] {
  const count = Math.max(2, Math.floor(length / 56));
  if (kind === "linear") return scaleLinear().domain(domain).ticks(Math.min(count, 8));
  const lo = Math.ceil(Math.log10(domain[0]));
  const hi = Math.floor(Math.log10(domain[1]));
  if (hi - lo < 2) {
    const ticks: number[] = [];
    for (let e = lo - 1; e <= hi; e++) {
      for (const m of [1, 2, 5]) {
        const v = Number((m * 10 ** e).toPrecision(1));
        if (v >= domain[0] && v <= domain[1]) ticks.push(v);
      }
    }
    return ticks;
  }
  const step = Math.max(1, Math.ceil((hi - lo + 1) / count));
  const ticks: number[] = [];
  for (let e = lo; e <= hi; e += step) ticks.push(10 ** e);
  return ticks;
}

export function tickLabel(kind: AxisKind, value: number): string {
  if (kind === "log") {
    const e = Math.log10(value);
    if (Math.abs(e - Math.round(e)) < 1e-9) {
      const r = Math.round(e);
      if (r >= 0 && r <= 4) return (10 ** r).toLocaleString("en-US");
      if (r === -1) return "0.1";
      if (r === -2) return "0.01";
      return `1e${r}`;
    }
    return String(Number(value.toPrecision(2)));
  }
  const abs = Math.abs(value);
  if (abs !== 0 && (abs >= 1e5 || abs < 1e-3)) return value.toExponential(0);
  return String(Number(value.toPrecision(4))).replace(/^-0$/, "0");
}

/** Keep at most about ``limit`` points; each bucket keeps its min and max, so spikes survive. */
export function decimate(x: number[], y: (number | null)[], limit: number): number[] {
  const n = x.length;
  if (n <= limit) return Array.from({ length: n }, (_, i) => i);
  const buckets = Math.max(1, Math.floor(limit / 2));
  const size = n / buckets;
  const keep: number[] = [0];
  for (let b = 0; b < buckets; b++) {
    const start = Math.floor(b * size);
    const end = Math.min(n, Math.floor((b + 1) * size));
    let lo = -1;
    let hi = -1;
    for (let i = start; i < end; i++) {
      const v = y[i];
      if (v == null || !Number.isFinite(v)) continue;
      if (lo < 0 || v < (y[lo] as number)) lo = i;
      if (hi < 0 || v > (y[hi] as number)) hi = i;
    }
    if (lo >= 0 && hi >= 0) {
      if (lo < hi) keep.push(lo, hi);
      else if (hi < lo) keep.push(hi, lo);
      else keep.push(lo);
    }
  }
  keep.push(n - 1);
  return [...new Set(keep)];
}

/** Index of the entry of a sorted array nearest to ``value``. */
export function nearest(sorted: number[], value: number): number {
  let lo = 0;
  let hi = sorted.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (sorted[mid] <= value) lo = mid;
    else hi = mid;
  }
  return Math.abs(sorted[hi] - value) < Math.abs(sorted[lo] - value) ? hi : lo;
}

// ------------------------------------------------------------------------ colour ramps
const SEQUENTIAL = {
  light: ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
  dark: ["#104281", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef", "#b7d3f6", "#cde2fb"],
};
const DIVERGING = {
  light: ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f2b0ae", "#e34948", "#9c2424"],
  dark: ["#b7d3f6", "#5598e7", "#1c5cab", "#383835", "#9c2f2f", "#e66767", "#f5b3b3"],
};

type RGB = [number, number, number];
const hexToRgb = (hex: string): RGB => [parseInt(hex.slice(1, 3), 16), parseInt(hex.slice(3, 5), 16), parseInt(hex.slice(5, 7), 16)];

function ramp(stops: string[]) {
  const rgb = stops.map(hexToRgb);
  return (t: number): RGB => {
    const clamped = Math.max(0, Math.min(1, t));
    const p = clamped * (rgb.length - 1);
    const i = Math.min(rgb.length - 2, Math.floor(p));
    const f = p - i;
    return [0, 1, 2].map((c) => rgb[i][c] + (rgb[i + 1][c] - rgb[i][c]) * f) as RGB;
  };
}

export type Theme = "light" | "dark";
export const sequentialRamp = (theme: Theme) => ramp(SEQUENTIAL[theme]);
export const divergingRamp = (theme: Theme) => ramp(DIVERGING[theme]);
export const rampStops = (kind: "sequential" | "diverging", theme: Theme) => (kind === "sequential" ? SEQUENTIAL : DIVERGING)[theme];

/** Black or white, whichever has more WCAG contrast on this fill: one of them always reaches 4.5:1. */
export function inkOn(rgb: RGB): string {
  const [r, g, b] = rgb.map((v) => {
    const c = v / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });
  const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  return 1.05 / (lum + 0.05) > (lum + 0.05) / 0.05 ? "#ffffff" : "#000000";
}

import { type PointerEvent, useEffect, useMemo, useRef, useState } from "react";
import { useTheme } from "@/hooks/useTheme";
import { sci } from "@/lib/format";
import { divergingRamp, rampStops, sequentialRamp, tickLabel, ticksOf, useSize } from "./core";
import { scaleLinear } from "d3-scale";

interface Props {
  title: string;
  x: number[];
  y: number[];
  /** Flat, x index first: value(i, j) = values[i * y.length + j]. */
  values: (number | null)[];
  kind?: "sequential" | "diverging";
  domain?: [number, number];
  height?: number;
  dots?: { x: number[]; y: number[] };
}

const MARGIN = { top: 4, right: 8, bottom: 28, left: 36 };

/** One function value per pixel of a small canvas, which the browser smooths when scaling up. */
export function HeatMap({ title, x, y, values, kind = "sequential", domain, height = 260, dots }: Props) {
  const { theme } = useTheme();
  const { ref, width } = useSize<HTMLDivElement>();
  const canvas = useRef<HTMLCanvasElement>(null);
  const [hover, setHover] = useState<{ i: number; j: number } | null>(null);
  const nx = x.length;
  const ny = y.length;

  const range = useMemo<[number, number]>(() => {
    if (domain) return domain;
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of values) {
      if (v == null) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if (!Number.isFinite(lo)) return [0, 1];
    if (kind === "diverging") {
      const m = Math.max(Math.abs(lo), Math.abs(hi), 1e-300);
      return [-m, m];
    }
    return lo === hi ? [lo - 1, hi + 1] : [lo, hi];
  }, [values, domain, kind]);

  const measured = width > 0; // the canvas only exists once the container has a size
  useEffect(() => {
    const ctx = canvas.current?.getContext("2d");
    if (!ctx || !nx || !ny) return;
    const colour = kind === "diverging" ? divergingRamp(theme) : sequentialRamp(theme);
    const image = ctx.createImageData(nx, ny);
    const span = range[1] - range[0] || 1;
    for (let i = 0; i < nx; i++) {
      for (let j = 0; j < ny; j++) {
        const v = values[i * ny + j];
        const p = ((ny - 1 - j) * nx + i) * 4; // canvas rows run down, y runs up
        if (v == null) continue;
        const [r, g, b] = colour((v - range[0]) / span);
        image.data[p] = r;
        image.data[p + 1] = g;
        image.data[p + 2] = b;
        image.data[p + 3] = 255;
      }
    }
    ctx.putImageData(image, 0, 0);
  }, [values, nx, ny, range, kind, theme, measured]);

  const plotW = Math.max(20, width - MARGIN.left - MARGIN.right);
  const plotH = Math.max(20, height - MARGIN.top - MARGIN.bottom);
  const sx = scaleLinear().domain([x[0], x[nx - 1]]).range([0, plotW]);
  const sy = scaleLinear().domain([y[0], y[ny - 1]]).range([plotH, 0]);

  const onMove = (event: PointerEvent<HTMLDivElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const i = Math.round(((event.clientX - rect.left) / rect.width) * (nx - 1));
    const j = Math.round((1 - (event.clientY - rect.top) / rect.height) * (ny - 1));
    if (i >= 0 && i < nx && j >= 0 && j < ny) setHover({ i, j });
  };

  const value = hover ? values[hover.i * ny + hover.j] : null;
  const stops = rampStops(kind, theme);

  return (
    <figure className="min-w-0">
      <figcaption className="mb-1 space-y-0.5 text-xs">
        <div className="font-medium text-foreground">{title}</div>
        <div className="flex items-center gap-1.5 text-subtle" aria-label={`Colour scale from ${sci(range[0])} to ${sci(range[1])}`}>
          <span className="tnum whitespace-nowrap">{sci(range[0])}</span>
          <span className="inline-block h-2 min-w-6 flex-1 rounded-sm" style={{ background: `linear-gradient(90deg, ${stops.join(",")})` }} />
          <span className="tnum whitespace-nowrap">{sci(range[1])}</span>
        </div>
      </figcaption>
      <div ref={ref} className="relative" style={{ height }}>
        {width > 0 && (
          <>
            <div
              className="absolute cursor-crosshair overflow-hidden rounded-sm"
              style={{ left: MARGIN.left, top: MARGIN.top, width: plotW, height: plotH }}
              onPointerMove={onMove}
              onPointerLeave={() => setHover(null)}
            >
              <canvas ref={canvas} width={nx} height={ny} role="img" aria-label={`${title}: heat map over x and y`} className="block size-full" />
            </div>
            <svg width={width} height={height} aria-hidden="true" className="pointer-events-none absolute inset-0">
              <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
                {dots?.x.map((px, k) => <circle key={k} cx={sx(px)} cy={sy(dots.y[k])} r={1} fill="var(--foreground)" opacity={0.25} />)}
                {ticksOf("linear", [x[0], x[nx - 1]], plotW * 1.2).map((t) => (
                  <text key={t} x={sx(t)} y={plotH + 15} textAnchor="middle" className="tnum fill-subtle text-[11px]">{tickLabel("linear", t)}</text>
                ))}
                {ticksOf("linear", [y[0], y[ny - 1]], plotH).map((t) => (
                  <text key={t} x={-6} y={sy(t)} dy="0.32em" textAnchor="end" className="tnum fill-subtle text-[11px]">{tickLabel("linear", t)}</text>
                ))}
                {hover && <circle cx={sx(x[hover.i])} cy={sy(y[hover.j])} r={4} fill="none" stroke="var(--foreground)" strokeWidth={1.5} />}
              </g>
            </svg>
            {hover && (
              <div className="pointer-events-none absolute top-1 right-2 z-10 rounded-md border bg-popover px-2 py-1 text-xs shadow-sm">
                <span className="tnum font-medium text-foreground">{sci(value)}</span>
                <span className="tnum ml-2 text-subtle">x {x[hover.i].toFixed(2)}, y {y[hover.j].toFixed(2)}</span>
              </div>
            )}
          </>
        )}
      </div>
    </figure>
  );
}

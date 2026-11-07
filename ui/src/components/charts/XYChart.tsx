import { area, line } from "d3-shape";
import { type KeyboardEvent, type PointerEvent, memo, useMemo, useState } from "react";
import { sci } from "@/lib/format";
import { Legend } from "./Legend";
import { type AxisKind, type Marker, decimate, domainOf, makeScale, markerPath, nearest, tickLabel, ticksOf, useSize } from "./core";

export interface XYSeries {
  id: string;
  label: string;
  color: string;
  x: number[];
  y: (number | null)[];
  lo?: (number | null)[];
  hi?: (number | null)[];
  width?: number;
  marker?: Marker | null;
  /** Painted under the rest and left out of nothing: used for the exact function. */
  underlay?: boolean;
}

interface Props {
  series: XYSeries[];
  xScale?: AxisKind;
  yScale?: AxisKind;
  xLabel?: string;
  yLabel?: string;
  height?: number;
  /** Category names when x holds positions 0, 1, 2, ... */
  xCategories?: string[];
  xDomain?: [number, number];
  yDomain?: [number, number];
  /** Shaded spans with a caption, such as the extrapolation zone. */
  regions?: { x0: number; x1: number; label: string }[];
  vlines?: { x: number; label: string }[];
  hlines?: { y: number; label: string }[];
  /** Quiet context dots without a tooltip, such as training samples. */
  dots?: { x: number[]; y: number[]; label: string };
  formatX?: (v: number) => string;
  formatY?: (v: number | null | undefined) => string;
  ariaLabel: string;
}

const MARGIN = { top: 10, right: 14, bottom: 40, left: 58 };
const MAX_PATH_FACTOR = 2;

function XYChartImpl({
  series, xScale = "linear", yScale = "linear", xLabel, yLabel, height = 280, xCategories, xDomain, yDomain,
  regions = [], vlines = [], hlines = [], dots, formatX, formatY = sci, ariaLabel,
}: Props) {
  const { ref, width } = useSize<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null); // x value under the crosshair

  const plotW = Math.max(40, width - MARGIN.left - MARGIN.right);
  const plotH = Math.max(40, height - MARGIN.top - MARGIN.bottom);

  const { sx, sy, xd, yd } = useMemo(() => {
    const xs = xDomain ?? (xCategories
      ? ([-0.3, xCategories.length - 0.7] as [number, number])
      : domainOf([...series.flatMap((s) => s.x), ...(dots?.x ?? [])], xScale, 0));
    const yValues = function* () {
      for (const s of series) {
        yield* s.y;
        if (s.lo) yield* s.lo;
        if (s.hi) yield* s.hi;
      }
      for (const h of hlines) yield h.y;
      if (dots) yield* dots.y;
    };
    const ys = yDomain ?? domainOf(yValues(), yScale);
    return { xd: xs, yd: ys, sx: makeScale(xScale, xs, [0, plotW]), sy: makeScale(yScale, ys, [plotH, 0]) };
  }, [series, xScale, yScale, xDomain, yDomain, xCategories, hlines, dots, plotW, plotH]);

  const valid = (v: number | null | undefined, kind: AxisKind): v is number =>
    v != null && Number.isFinite(v) && (kind === "linear" || v > 0);

  const drawn = useMemo(() => {
    if (!width) return [];
    const limit = Math.ceil(plotW * MAX_PATH_FACTOR);
    return series.map((s) => {
      const idx = decimate(s.x, s.y, limit);
      const path = line<number>()
        .defined((i) => valid(s.y[i], yScale) && valid(s.x[i], xScale))
        .x((i) => sx(s.x[i]))
        .y((i) => sy(s.y[i] as number))(idx) ?? "";
      const band = s.lo && s.hi
        ? area<number>()
            .defined((i) => valid(s.lo![i], yScale) && valid(s.hi![i], yScale))
            .x((i) => sx(s.x[i]))
            .y0((i) => sy(s.lo![i] as number))
            .y1((i) => sy(s.hi![i] as number))(idx) ?? ""
        : "";
      return { ...s, path, band };
    });
  }, [series, sx, sy, width, plotW, xScale, yScale]);

  const allX = useMemo(() => {
    const longest = series.reduce<XYSeries | null>((best, s) => (!best || s.x.length > best.x.length ? s : best), null);
    return longest?.x ?? [];
  }, [series]);

  const onMove = (event: PointerEvent<SVGSVGElement>) => {
    if (!allX.length) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const px = event.clientX - rect.left - MARGIN.left;
    if (px < -8 || px > plotW + 8) return setHover(null);
    setHover(allX[nearest(allX, sx.invert(px))]);
  };

  const onKey = (event: KeyboardEvent<SVGSVGElement>) => {
    if (!allX.length) return;
    const current = hover == null ? -1 : nearest(allX, hover);
    const step = event.shiftKey ? Math.max(1, Math.round(allX.length / 20)) : 1;
    const moves: Record<string, number> = {
      ArrowRight: Math.min(allX.length - 1, current + step),
      ArrowLeft: Math.max(0, (current < 0 ? allX.length : current) - step),
      Home: 0,
      End: allX.length - 1,
    };
    if (event.key === "Escape") return setHover(null);
    if (event.key in moves) {
      event.preventDefault();
      setHover(allX[moves[event.key]]);
    }
  };

  const readout = hover == null ? [] : series.map((s) => {
    if (!s.x.length) return { s, value: null as number | null, lo: null as number | null | undefined, hi: null as number | null | undefined };
    const i = nearest(s.x, hover);
    return { s, value: s.y[i], lo: s.lo?.[i], hi: s.hi?.[i] };
  });
  const crossX = hover == null ? null : sx(hover);
  const label = (v: number) => (xCategories ? xCategories[Math.round(v)] ?? "" : formatX ? formatX(v) : sci(v));

  const xTicks = xCategories ? xCategories.map((_, i) => i) : ticksOf(xScale, xd, plotW * 1.1);
  const yTicks = ticksOf(yScale, yd, plotH);
  const legendItems = series.map((s) => ({ label: s.label, color: s.color, marker: s.marker }));

  return (
    <div className="flex min-w-0 flex-col gap-2">
      <Legend items={legendItems} />
      <div ref={ref} className="relative min-w-0" style={{ height }}>
        {width > 0 && (
          <svg
            width={width}
            height={height}
            role="img"
            aria-label={ariaLabel}
            tabIndex={0}
            className="block touch-pan-y rounded-sm"
            onPointerMove={onMove}
            onPointerLeave={() => setHover(null)}
            onKeyDown={onKey}
            onBlur={() => setHover(null)}
          >
            <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
              {regions.map((r) => {
                const x0 = Math.max(0, sx(r.x0));
                const x1 = Math.min(plotW, sx(r.x1));
                return x1 > x0 ? (
                  <g key={`${r.x0}`}>
                    <rect x={x0} y={0} width={x1 - x0} height={plotH} fill="var(--wash)" />
                    <text x={x0 + 6} y={14} className="fill-subtle text-[11px]">{r.label}</text>
                  </g>
                ) : null;
              })}
              {yTicks.map((t) => (
                <g key={t} transform={`translate(0,${sy(t)})`}>
                  <line x2={plotW} stroke="var(--border)" />
                  <text x={-8} dy="0.32em" textAnchor="end" className="tnum fill-subtle text-[11px]">{tickLabel(yScale, t)}</text>
                </g>
              ))}
              <line y1={plotH} y2={plotH} x2={plotW} stroke="var(--axis)" />
              {xTicks.map((t) => (
                <text key={t} x={sx(t)} y={plotH + 16} textAnchor="middle" className="tnum fill-subtle text-[11px]">
                  {xCategories ? xCategories[t] : tickLabel(xScale, t)}
                </text>
              ))}
              {xLabel && <text x={plotW / 2} y={plotH + 34} textAnchor="middle" className="fill-muted-foreground text-xs">{xLabel}</text>}
              {yLabel && (
                <text transform={`translate(-46,${plotH / 2}) rotate(-90)`} textAnchor="middle" className="fill-muted-foreground text-xs">{yLabel}</text>
              )}

              {hlines.map((h) => (
                <g key={h.label}>
                  <line x2={plotW} y1={sy(h.y)} y2={sy(h.y)} stroke="var(--subtle)" />
                  <text x={plotW - 4} y={sy(h.y) - 5} textAnchor="end" className="fill-muted-foreground text-[11px]">{h.label}</text>
                </g>
              ))}
              {vlines.map((v) => (
                <g key={v.label}>
                  <line x1={sx(v.x)} x2={sx(v.x)} y2={plotH} stroke="var(--subtle)" />
                  <text x={Math.min(sx(v.x) + 5, plotW - 90)} y={12} className="fill-muted-foreground text-[11px]">{v.label}</text>
                </g>
              ))}

              <defs>
                <clipPath id={`clip-${ariaLabel.replace(/\W/g, "")}`}>
                  <rect x={-4} y={-4} width={plotW + 8} height={plotH + 8} />
                </clipPath>
              </defs>
              <g clipPath={`url(#clip-${ariaLabel.replace(/\W/g, "")})`}>
                {drawn.filter((s) => s.band).map((s) => <path key={`b${s.id}`} d={s.band} fill={s.color} opacity={0.12} />)}
                {drawn.filter((s) => s.underlay).map((s) => (
                  <path key={s.id} d={s.path} fill="none" stroke={s.color} strokeWidth={s.width ?? 4} strokeLinecap="round" strokeLinejoin="round" />
                ))}
                {dots && dots.x.map((x, i) => ( // above the thick exact curve, or noise-free samples vanish under it
                  <circle key={i} cx={sx(x)} cy={sy(dots.y[i])} r={1.6} fill="var(--foreground)" opacity={0.45} />
                ))}
                {drawn.filter((s) => !s.underlay).map((s) => (
                  <path key={s.id} d={s.path} fill="none" stroke={s.color} strokeWidth={s.width ?? 2} strokeLinecap="round" strokeLinejoin="round" />
                ))}
                {drawn.filter((s) => s.marker).map((s) => s.x.map((x, i) => (valid(s.y[i], yScale) && valid(x, xScale) ? (
                  <path key={`${s.id}${i}`} d={markerPath(s.marker!, 4)} transform={`translate(${sx(x)},${sy(s.y[i] as number)})`} fill={s.color} stroke="var(--surface)" strokeWidth={2} />
                ) : null)))}
              </g>

              {crossX != null && (
                <g pointerEvents="none">
                  <line x1={crossX} x2={crossX} y2={plotH} stroke="var(--subtle)" />
                  {readout.map(({ s, value }) => (valid(value, yScale) ? (
                    <circle key={s.id} cx={crossX} cy={sy(value)} r={4} fill={s.color} stroke="var(--surface)" strokeWidth={2} />
                  ) : null))}
                </g>
              )}
            </g>
          </svg>
        )}
        {crossX != null && (
          <div
            className="pointer-events-none absolute top-2 z-10 max-w-[280px] rounded-md border bg-popover px-2.5 py-2 text-xs shadow-sm"
            style={crossX + MARGIN.left > width / 2 ? { right: width - (crossX + MARGIN.left) + 12 } : { left: crossX + MARGIN.left + 12 }}
          >
            <div className="tnum mb-1 text-subtle">{xLabel ? `${xLabel}: ` : ""}{label(hover!)}</div>
            <ul className="space-y-0.5">
              {readout.map(({ s, value, lo, hi }) => (
                <li key={s.id} className="flex items-baseline gap-1.5">
                  <span aria-hidden="true" className="relative top-[-2px] inline-block h-0.5 w-3 shrink-0 rounded" style={{ background: s.color }} />
                  <span className="tnum font-medium text-foreground">{formatY(value)}</span>
                  <span className="truncate text-muted-foreground">{s.label}</span>
                  {lo != null && hi != null && lo !== hi && (
                    <span className="tnum shrink-0 text-subtle">({formatY(lo)} to {formatY(hi)})</span>
                  )}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}

export const XYChart = memo(XYChartImpl);

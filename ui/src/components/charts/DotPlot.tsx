import { type KeyboardEvent, useMemo, useState } from "react";
import { sci } from "@/lib/format";
import { medianOf } from "@/lib/runs";
import { Legend } from "./Legend";
import { type AxisKind, type Marker, domainOf, makeScale, markerPath, tickLabel, ticksOf, useSize } from "./core";

export interface Dot {
  category: string;
  value: number;
  series: string;
  detail?: string;
}

interface Props {
  dots: Dot[];
  categories: string[];
  seriesStyle: Record<string, { color: string; marker: Marker }>;
  scale?: AxisKind;
  xLabel: string;
  reference?: { value: number; label: string };
  ariaLabel: string;
}

const ROW = 30;
const CHAR_WIDTH = 7.2; // upper bound for 12px Inter, so a label is measured generously rather than clipped
const MARGIN = { top: 8, right: 16, bottom: 40 };

/** One row per category, one dot per run, a tick at the median of each row. */
export function DotPlot({ dots, categories, seriesStyle, scale = "log", xLabel, reference, ariaLabel }: Props) {
  const { ref, width } = useSize<HTMLDivElement>();
  const [active, setActive] = useState<number | null>(null);

  const drawable = useMemo(() => dots.filter((d) => Number.isFinite(d.value) && (scale === "linear" || d.value > 0)), [dots, scale]);
  const longest = Math.max(...categories.map((c) => c.length), 4);
  const left = Math.min(Math.max(70, longest * CHAR_WIDTH + 18), width * 0.42 || 180);
  const maxChars = Math.floor((left - 16) / CHAR_WIDTH);
  const height = MARGIN.top + categories.length * ROW + MARGIN.bottom;
  const plotW = Math.max(40, width - left - MARGIN.right);
  const plotH = categories.length * ROW;

  const domain = useMemo(
    () => domainOf([...drawable.map((d) => d.value), ...(reference ? [reference.value] : [])], scale),
    [drawable, reference, scale],
  );
  const sx = makeScale(scale, domain, [0, plotW]);
  const rowY = (category: string) => categories.indexOf(category) * ROW + ROW / 2;
  const medians = categories.map((category) => {
    const values = drawable.filter((d) => d.category === category).map((d) => d.value);
    return values.length ? { category, value: medianOf(values) } : null;
  });

  const onKey = (event: KeyboardEvent<SVGSVGElement>) => {
    if (!drawable.length) return;
    if (event.key === "Escape") return setActive(null);
    const delta = event.key === "ArrowRight" || event.key === "ArrowDown" ? 1 : event.key === "ArrowLeft" || event.key === "ArrowUp" ? -1 : 0;
    if (delta) {
      event.preventDefault();
      setActive((a) => (a == null ? 0 : (a + delta + drawable.length) % drawable.length));
    }
  };

  const shown = active == null ? null : drawable[active];
  const legend = Object.entries(seriesStyle).map(([label, s]) => ({ label, color: s.color, marker: s.marker }));

  return (
    <div className="flex min-w-0 flex-col gap-2">
      <Legend items={legend} />
      <div ref={ref} className="relative min-w-0" style={{ height }}>
        {width > 0 && (
          <svg width={width} height={height} role="img" aria-label={ariaLabel} tabIndex={0} onKeyDown={onKey} onBlur={() => setActive(null)} className="block rounded-sm">
            <g transform={`translate(${left},${MARGIN.top})`}>
              {ticksOf(scale, domain, plotW * 1.1).map((t) => (
                <g key={t}>
                  <line x1={sx(t)} x2={sx(t)} y2={plotH} stroke="var(--border)" />
                  <text x={sx(t)} y={plotH + 16} textAnchor="middle" className="tnum fill-subtle text-[11px]">{tickLabel(scale, t)}</text>
                </g>
              ))}
              <text x={plotW / 2} y={plotH + 34} textAnchor="middle" className="fill-muted-foreground text-xs">{xLabel}</text>
              {reference && (
                <g>
                  <line x1={sx(reference.value)} x2={sx(reference.value)} y2={plotH} stroke="var(--subtle)" />
                  <text x={sx(reference.value) - 5} y={11} textAnchor="end" className="fill-muted-foreground text-[11px]">{reference.label}</text>
                </g>
              )}
              {categories.map((category) => (
                <g key={category}>
                  <line x2={plotW} y1={rowY(category)} y2={rowY(category)} stroke="var(--border)" strokeOpacity={0.6} />
                  <text x={-10} y={rowY(category)} dy="0.32em" textAnchor="end" className="fill-foreground text-xs">
                    {category.length > maxChars ? `${category.slice(0, maxChars - 1)}…` : category}
                    <title>{category}</title>
                  </text>
                </g>
              ))}
              {medians.map((m) => m && (
                <line key={m.category} x1={sx(m.value)} x2={sx(m.value)} y1={rowY(m.category) - 9} y2={rowY(m.category) + 9} stroke="var(--foreground)" strokeWidth={2} strokeLinecap="round" />
              ))}
              {drawable.map((d, i) => {
                const s = seriesStyle[d.series];
                return (
                  <g key={i} transform={`translate(${sx(d.value)},${rowY(d.category)})`} onPointerEnter={() => setActive(i)} onPointerLeave={() => setActive(null)}>
                    <circle r={12} fill="transparent" />
                    <path d={markerPath(s.marker, active === i ? 5.2 : 4)} fill={s.color} stroke="var(--surface)" strokeWidth={2} />
                  </g>
                );
              })}
            </g>
          </svg>
        )}
        {shown && (
          <div
            className="pointer-events-none absolute z-10 rounded-md border bg-popover px-2.5 py-1.5 text-xs shadow-sm"
            style={{
              top: Math.min(rowY(shown.category) + MARGIN.top + 12, height - 56),
              ...(left + sx(shown.value) > width / 2
                ? { right: width - (left + sx(shown.value)) + 8 }
                : { left: left + sx(shown.value) + 8 }),
            }}
          >
            <div className="tnum font-medium text-foreground">{sci(shown.value)}</div>
            <div className="text-muted-foreground">{shown.category}{shown.detail ? ` · ${shown.detail}` : ""}</div>
          </div>
        )}
      </div>
    </div>
  );
}

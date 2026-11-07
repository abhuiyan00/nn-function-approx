import { useTheme } from "@/hooks/useTheme";
import { sci } from "@/lib/format";
import { inkOn, rampStops, sequentialRamp } from "./core";

interface Props {
  xLabel: string;
  yLabel: string;
  columns: { key: string | number; label: string }[];
  rows: { key: string | number; label: string }[];
  value: (row: string | number, column: string | number) => number | null;
  caption: string;
}

/** Two factors and one error value: a real table, with the value printed in every cell. */
export function HeatTable({ xLabel, yLabel, columns, rows, value, caption }: Props) {
  const { theme } = useTheme();
  const colour = sequentialRamp(theme);
  const cells = rows.flatMap((r) => columns.map((c) => value(r.key, c.key))).filter((v): v is number => v != null && v > 0);
  const lo = Math.log10(Math.min(...cells));
  const hi = Math.log10(Math.max(...cells));
  const stops = rampStops("sequential", theme);

  return (
    <div className="min-w-0 overflow-x-auto">
      <table className="w-full min-w-[440px] border-separate border-spacing-0.5 text-xs">
        <caption className="sr-only">{caption}</caption>
        <thead>
          <tr>
            <th scope="col" className="px-2 py-1 text-left font-normal text-subtle">{yLabel} ↓ · {xLabel} →</th>
            {columns.map((c) => <th key={c.key} scope="col" className="tnum px-2 py-1 font-medium text-muted-foreground">{c.label}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.key}>
              <th scope="row" className="tnum px-2 py-1 text-left font-medium text-muted-foreground">{r.label}</th>
              {columns.map((c) => {
                const v = value(r.key, c.key);
                if (v == null || v <= 0) return <td key={c.key} className="rounded-sm bg-wash px-2 py-2 text-center text-subtle">—</td>;
                const rgb = colour(hi > lo ? (Math.log10(v) - lo) / (hi - lo) : 0.5);
                return (
                  <td key={c.key} className="tnum rounded-sm px-2 py-2 text-center whitespace-nowrap" style={{ background: `rgb(${rgb.join(",")})`, color: inkOn(rgb) }}>
                    {sci(v)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mt-2 flex items-center gap-2 text-xs text-subtle">
        <span>Lower error</span>
        <span className="inline-block h-2 w-24 rounded-sm" style={{ background: `linear-gradient(90deg, ${stops.join(",")})` }} />
        <span>Higher error (log scale)</span>
      </div>
    </div>
  );
}

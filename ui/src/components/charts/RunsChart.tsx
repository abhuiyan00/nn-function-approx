import { useMemo, useState } from "react";
import { BarChart3, Table2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Segmented } from "@/components/common";
import type { Dataset } from "@/lib/api";
import type { ChartSpec } from "@/lib/experimentCharts";
import { sci } from "@/lib/format";
import { type Row, aggregate, medianOf } from "@/lib/runs";
import { DotPlot, type Dot } from "./DotPlot";
import { HeatTable } from "./HeatTable";
import { XYChart, type XYSeries } from "./XYChart";
import { MARKERS, seriesColor, tickLabel } from "./core";

const show = (spec: ChartSpec, value: string | number | null | undefined) => {
  const key = String(value ?? "n/a");
  return spec.names?.[key] ?? key;
};

function melted(rows: Row[], columns: string[]): Row[] {
  return rows.flatMap((row) => columns.map((measure) => ({ ...row, measure, value: row[measure] })));
}

/** Median table behind every chart: no value is reachable through hover only. */
function TableView({ spec, rows, seriesKeys }: { spec: ChartSpec; rows: Row[]; seriesKeys: string[] }) {
  const grouped = useMemo(() => {
    const groups = new Map<string, { series: string; x: string | number; values: number[] }>();
    for (const row of rows) {
      const y = spec.kind === "dot" ? row[spec.x] : spec.kind === "heat" ? row.rmse : row[spec.y];
      const x = spec.kind === "dot" ? row[spec.y] : row[spec.x];
      if (typeof y !== "number" || x == null) continue;
      const series = spec.color ? show(spec, row[spec.color]) : "";
      const axis = spec.kind === "heat" ? `${x} / ${show(spec, row[spec.y])}` : x;
      const key = `${series}|${axis}`;
      if (!groups.has(key)) groups.set(key, { series, x: axis, values: [] });
      groups.get(key)!.values.push(y);
    }
    return [...groups.values()];
  }, [rows, spec]);

  return (
    <div className="max-h-80 overflow-auto rounded-md border">
      <table className="w-full text-xs">
        <caption className="sr-only">{spec.title}: medians of the plotted values</caption>
        <thead className="sticky top-0 bg-wash text-left text-muted-foreground">
          <tr>
            {seriesKeys.length > 0 && <th scope="col" className="px-3 py-2 font-medium">Series</th>}
            <th scope="col" className="px-3 py-2 font-medium">{spec.kind === "dot" ? "Category" : spec.xLabel}</th>
            <th scope="col" className="px-3 py-2 text-right font-medium">Median</th>
            <th scope="col" className="px-3 py-2 text-right font-medium">Min</th>
            <th scope="col" className="px-3 py-2 text-right font-medium">Max</th>
            <th scope="col" className="px-3 py-2 text-right font-medium">Runs</th>
          </tr>
        </thead>
        <tbody>
          {grouped.map((g) => (
            <tr key={`${g.series}${g.x}`} className="border-t">
              {seriesKeys.length > 0 && <td className="px-3 py-1.5">{g.series}</td>}
              <td className="tnum px-3 py-1.5">{typeof g.x === "number" ? sci(g.x) : g.x}</td>
              <td className="tnum px-3 py-1.5 text-right font-medium">{sci(medianOf(g.values))}</td>
              <td className="tnum px-3 py-1.5 text-right">{sci(Math.min(...g.values))}</td>
              <td className="tnum px-3 py-1.5 text-right">{sci(Math.max(...g.values))}</td>
              <td className="tnum px-3 py-1.5 text-right">{g.values.length}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function buildPlot(spec: ChartSpec, rows: Row[], seriesKeys: string[]) {
  const label = (key: string) => (spec.color ? `${spec.colorPrefix ?? ""}${show(spec, key)}` : spec.title);
  const index = (key: string) => Math.max(0, seriesKeys.indexOf(key));

  if (spec.kind === "dot") {
    const categories = [...new Set(rows.map((r) => String(r[spec.y])))];
    const dots: Dot[] = rows
      .filter((r) => typeof r[spec.x] === "number")
      .map((r) => {
        const series = spec.color ? String(r[spec.color]) : "";
        const seed = r.seed != null ? `seed ${r.seed}` : undefined;
        return { category: String(r[spec.y]), value: r[spec.x] as number, series: label(series), detail: seed ?? (spec.color ? show(spec, series) : undefined) };
      });
    const seriesStyle = Object.fromEntries(
      (seriesKeys.length ? seriesKeys : [""]).map((key) => [label(key), { color: seriesColor(index(key)), marker: MARKERS[index(key) % MARKERS.length] }]),
    );
    return { kind: "dot" as const, dots, categories, seriesStyle };
  }

  const ordinal = spec.xScale === "ordinal";
  const categories = ordinal ? [...new Set(rows.map((r) => r[spec.x]))].filter((v): v is number => typeof v === "number").sort((a, b) => a - b) : [];
  const positioned = ordinal ? rows.map((r) => ({ ...r, __x: categories.indexOf(r[spec.x] as number) })) : rows;
  const series: XYSeries[] = aggregate(positioned, ordinal ? "__x" : spec.x, spec.y, spec.color, {
    positiveX: spec.xScale === "log",
    positiveY: spec.yScale === "log",
    order: spec.color ? seriesKeys : undefined,
  }).map((s) => ({
    id: s.key,
    label: label(s.key),
    color: seriesColor(index(s.key)),
    x: s.points.map((p) => p.x),
    y: s.points.map((p) => p.y),
    lo: s.points.map((p) => p.lo),
    hi: s.points.map((p) => p.hi),
    marker: s.points.length <= 40 ? MARKERS[index(s.key) % MARKERS.length] : null,
  }));
  return { kind: "line" as const, series, xCategories: ordinal ? categories.map((v) => tickLabel("linear", v)) : undefined };
}

export function RunsChart({ spec, data }: { spec: ChartSpec; data: Dataset }) {
  const base = useMemo(() => {
    const filtered = data.rows.filter((r) => Object.entries(spec.where ?? {}).every(([k, values]) => values.includes(r[k] as string | number)));
    return spec.melt ? melted(filtered, spec.melt) : filtered;
  }, [data, spec]);

  const facets = useMemo(() => (spec.facet ? [...new Set(base.map((r) => String(r[spec.facet!])))] : []), [base, spec]);
  const [facetChoice, setFacet] = useState<string | null>(null);
  const facet = facetChoice != null && facets.includes(facetChoice) ? facetChoice : facets[0];
  const [view, setView] = useState<"chart" | "table">("chart");

  // Assigned before the facet filter: a series keeps its colour whichever facet is showing
  const seriesKeys = useMemo(() => (spec.color ? [...new Set(base.map((r) => String(r[spec.color!] ?? "n/a")))] : []), [base, spec]);
  const rows = useMemo(() => (spec.facet ? base.filter((r) => String(r[spec.facet!]) === facet) : base), [base, spec, facet]);
  const plot = useMemo(() => (spec.kind === "heat" ? null : buildPlot(spec, rows, seriesKeys)), [spec, rows, seriesKeys]);

  const heat = useMemo(() => {
    if (spec.kind !== "heat") return null;
    const xs = [...new Set(rows.map((r) => r[spec.x] as number))].sort((a, b) => a - b);
    // batch 0 means full batch, which is the largest
    const ys = [...new Set(rows.map((r) => r[spec.y] as number))].sort((a, b) => (a || Infinity) - (b || Infinity));
    return {
      columns: xs.map((v) => ({ key: v, label: show(spec, v) })),
      rows: ys.map((v) => ({ key: v, label: show(spec, v) })),
      value: (row: string | number, column: string | number) => {
        const values = rows.filter((r) => r[spec.y] === row && r[spec.x] === column && typeof r.rmse === "number").map((r) => r.rmse as number);
        return values.length ? medianOf(values) : null;
      },
    };
  }, [rows, spec]);

  return (
    <Card className="gap-3 p-4">
      <header className="flex flex-wrap items-start justify-between gap-x-4 gap-y-2">
        <div className="min-w-0 flex-1 basis-56">
          <h3 className="text-sm font-semibold">{spec.title}</h3>
          {spec.note && <p className="text-xs text-muted-foreground">{spec.note}</p>}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {facets.length > 1 && (
            <Segmented
              label={`Filter ${spec.title}`}
              value={facet}
              onChange={setFacet}
              options={facets.map((f) => ({ value: f, label: show(spec, f) }))}
            />
          )}
          <Button
            variant="ghost"
            size="icon-sm"
            aria-pressed={view === "table"}
            aria-label={view === "chart" ? "Show as table" : "Show as chart"}
            title={view === "chart" ? "Show as table" : "Show as chart"}
            onClick={() => setView(view === "chart" ? "table" : "chart")}
          >
            {view === "chart" ? <Table2 /> : <BarChart3 />}
          </Button>
        </div>
      </header>
      {view === "table" ? (
        <TableView spec={spec} rows={rows} seriesKeys={seriesKeys} />
      ) : plot?.kind === "line" ? (
        <XYChart
          series={plot.series}
          xScale={spec.xScale === "log" ? "log" : "linear"}
          yScale={spec.yScale ?? "linear"}
          xCategories={plot.xCategories}
          xLabel={spec.xLabel}
          yLabel={spec.yLabel}
          height={320}
          hlines={spec.reference ? [{ y: spec.reference.value, label: spec.reference.label }] : undefined}
          ariaLabel={`${spec.title}${facet ? `, ${show(spec, facet)}` : ""}`}
        />
      ) : plot?.kind === "dot" ? (
        <DotPlot
          dots={plot.dots}
          categories={plot.categories}
          seriesStyle={plot.seriesStyle}
          scale={spec.xScale === "log" ? "log" : "linear"}
          xLabel={spec.xLabel}
          reference={spec.reference}
          ariaLabel={`${spec.title}${facet ? `, ${show(spec, facet)}` : ""}`}
        />
      ) : heat ? (
        <HeatTable xLabel={spec.xLabel} yLabel={spec.yLabel ?? spec.y} columns={heat.columns} rows={heat.rows} value={heat.value} caption={spec.title} />
      ) : null}
    </Card>
  );
}

import { useQuery } from "@tanstack/react-query";
import { useEffect, useMemo } from "react";
import { useSearchParams } from "react-router";
import { RunsChart } from "@/components/charts/RunsChart";
import { DataTable } from "@/components/DataTable";
import { EmptyState, ErrorState, LoadingBlock, PageHeader } from "@/components/common";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import type { Dataset, DatasetColumn } from "@/lib/api";
import { api } from "@/lib/api";
import type { ChartSpec } from "@/lib/experimentCharts";
import type { Row } from "@/lib/runs";

const ANY = "__any";
const NONE = "__none";
const MAX_FILTERS = 6;
const MAX_SERIES = 8;
const PREFERRED_Y = ["rmse", "error", "rel_error", "coverage", "spread"];
const PREFERRED_X = ["gap_pct", "n_params", "n_train", "step", "lr", "nominal", "seconds"];
const FILTERS_FIRST = ["target", "sigma"]; // a result file can have more columns than there is room for filters
const PREFERRED_COLOR = ["model", "method", "variant", "act", "optimizer", "depth", "sigma", "check", "k", "score"];

function Control({ label, children, htmlFor }: { label: string; children: React.ReactNode; htmlFor: string }) {
  return (
    <div className="min-w-32 flex-1 space-y-1 sm:max-w-52">
      <Label htmlFor={htmlFor} className="text-xs font-medium text-muted-foreground">{label}</Label>
      {children}
    </div>
  );
}

function Picker({ id, value, onChange, options }: { id: string; value: string; onChange: (v: string) => void; options: { value: string; label: string }[] }) {
  return (
    <Select value={value} onValueChange={onChange}>
      <SelectTrigger id={id} size="sm" className="w-full"><SelectValue /></SelectTrigger>
      <SelectContent>{options.map((o) => <SelectItem key={o.value} value={o.value}>{o.label}</SelectItem>)}</SelectContent>
    </Select>
  );
}

const firstOf = (preferred: string[], columns: DatasetColumn[]) => preferred.find((n) => columns.some((c) => c.name === n));
const seriesColumns = (columns: DatasetColumn[]) => columns.filter((c) => !c.constant && c.values && c.unique >= 2 && c.unique <= MAX_SERIES);

export function Explorer() {
  const [params, setParams] = useSearchParams();
  const catalog = useQuery({ queryKey: ["datasets"], queryFn: api.datasets });
  const chosen = params.get("data") ?? "e6/runs";
  const [exp, name] = chosen.split("/");
  const dataset = useQuery({ queryKey: ["dataset", exp, name], queryFn: () => api.dataset(exp, name) });

  const update = (patch: Record<string, string | null>) => setParams((p) => {
    const next = new URLSearchParams(p);
    for (const [k, v] of Object.entries(patch)) {
      if (v == null) next.delete(k);
      else next.set(k, v);
    }
    return next;
  }, { replace: true });

  const data = dataset.data;
  const columns = useMemo(() => data?.columns ?? [], [data]);
  const numeric = useMemo(() => columns.filter((c) => c.type === "number" && !c.constant), [columns]);
  const y = params.get("y") ?? firstOf(PREFERRED_Y, numeric) ?? numeric[0]?.name ?? "";
  const x = params.get("x") ?? firstOf(PREFERRED_X.filter((n) => n !== y), numeric) ?? numeric.find((c) => c.name !== y)?.name ?? "";
  const seriesChoices = useMemo(() => seriesColumns(columns), [columns]);
  const color = params.get("color") ?? firstOf(PREFERRED_COLOR, seriesChoices) ?? NONE;
  const logX = params.get("logx") != null ? params.get("logx") === "1" : x === "n_params" || x === "n_train" || x === "lr";
  const logY = params.get("logy") != null ? params.get("logy") === "1" : /rmse|err|mse|max/.test(y);

  const filterColumns = useMemo(() => {
    const later = (c: DatasetColumn) => (FILTERS_FIRST.includes(c.name) ? 0 : 1);
    return seriesChoices.filter((c) => c.name !== color).sort((a, b) => later(a) - later(b)).slice(0, MAX_FILTERS);
  }, [seriesChoices, color]);
  const filterValue = (c: DatasetColumn) => params.get(`f.${c.name}`) ?? (c.name === "target" ? String(c.values![0]) : ANY); // medians across different functions mean nothing, so start on one
  const filters = useMemo(() => filterColumns.flatMap((c) => {
    const value = filterValue(c);
    return value === ANY ? [] : [{ column: c.name, value }];
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }), [filterColumns, params]);

  const rows = useMemo<Row[]>(() => (data?.rows ?? []).filter((r) => filters.every((f) => String(r[f.column]) === f.value)), [data, filters]);
  const filtered = useMemo<Dataset | null>(() => (data ? { ...data, rows } : null), [data, rows]);

  const spec = useMemo<ChartSpec | null>(() => {
    if (!x || !y) return null;
    const label = (n: string) => n.replace(/_/g, " ");
    return {
      id: `${chosen}-${x}-${y}-${color}`,
      title: `${label(y)} against ${label(x)}`,
      note: "Median per x within each series; the band spans the minimum to the maximum.",
      dataset: name,
      kind: "line",
      x,
      y,
      color: color === NONE ? undefined : color,
      xScale: logX ? "log" : "linear",
      yScale: logY ? "log" : "linear",
      xLabel: label(x),
      yLabel: label(y),
      colorPrefix: color === NONE ? undefined : `${label(color)}: `,
    };
  }, [chosen, name, x, y, color, logX, logY]);

  useEffect(() => {
    if (catalog.data && !catalog.data.some((d) => `${d.experiment}/${d.name}` === chosen)) update({ data: "e6/runs" });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [catalog.data, chosen]);

  if (catalog.isError) return <ErrorState error={catalog.error} onRetry={() => catalog.refetch()} />;

  return (
    <div className="space-y-4">
      <PageHeader title="Run Explorer" description="Plot any measurement against any setting, across every recorded run. The address keeps your choices, so a view can be shared." />

      <div className="space-y-3 rounded-lg border bg-card p-3" role="group" aria-label="Chart settings">
        <div className="flex flex-wrap items-end gap-3">
          <Control label="Result file" htmlFor="ex-data">
            <Picker
              id="ex-data"
              value={chosen}
              onChange={(v) => setParams({ data: v }, { replace: true })}
              options={(catalog.data ?? [{ experiment: exp, name, rows: 0 }]).map((d) => ({ value: `${d.experiment}/${d.name}`, label: `${d.experiment.toUpperCase()} · ${d.name}${d.rows ? ` (${d.rows})` : ""}` }))}
            />
          </Control>
          <Control label="Horizontal axis" htmlFor="ex-x">
            <Picker id="ex-x" value={x} onChange={(v) => update({ x: v, logx: null })} options={numeric.map((c) => ({ value: c.name, label: c.name }))} />
          </Control>
          <Control label="Vertical axis" htmlFor="ex-y">
            <Picker id="ex-y" value={y} onChange={(v) => update({ y: v, logy: null })} options={numeric.map((c) => ({ value: c.name, label: c.name }))} />
          </Control>
          <Control label="Separate series by" htmlFor="ex-color">
            <Picker
              id="ex-color"
              value={color}
              onChange={(v) => update({ color: v })}
              options={[{ value: NONE, label: "One series" }, ...seriesChoices.map((c) => ({ value: c.name, label: `${c.name} (${c.unique})` }))]}
            />
          </Control>
          <div className="flex items-center gap-4 pb-1.5">
            <div className="flex items-center gap-2">
              <Switch id="ex-logx" checked={logX} onCheckedChange={(v) => update({ logx: v ? "1" : "0" })} />
              <Label htmlFor="ex-logx" className="text-xs font-normal">Log x</Label>
            </div>
            <div className="flex items-center gap-2">
              <Switch id="ex-logy" checked={logY} onCheckedChange={(v) => update({ logy: v ? "1" : "0" })} />
              <Label htmlFor="ex-logy" className="text-xs font-normal">Log y</Label>
            </div>
          </div>
        </div>
        {filterColumns.length > 0 && (
          <div className="flex flex-wrap items-end gap-3 border-t pt-3">
            {filterColumns.map((c) => (
              <Control key={c.name} label={`Only ${c.name}`} htmlFor={`ex-f-${c.name}`}>
                <Picker
                  id={`ex-f-${c.name}`}
                  value={filterValue(c)}
                  onChange={(v) => update({ [`f.${c.name}`]: v })}
                  options={[{ value: ANY, label: "All" }, ...(c.values ?? []).map((v) => ({ value: String(v), label: String(v) }))]}
                />
              </Control>
            ))}
          </div>
        )}
      </div>

      {dataset.isError && <ErrorState error={dataset.error} onRetry={() => dataset.refetch()} />}
      {!data && !dataset.isError && <LoadingBlock />}
      {filtered && numeric.length < 2 && (
        <EmptyState title="Nothing to plot in this file">It has fewer than two numeric columns that vary.</EmptyState>
      )}
      {filtered && spec && numeric.length >= 2 && <RunsChart key={spec.id} spec={spec} data={filtered} />}
      {filtered && <DataTable columns={columns} rows={rows} filename={`${exp}_${name}`} />}
    </div>
  );
}

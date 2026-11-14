import { useMutation, useQuery } from "@tanstack/react-query";
import { Download, Play, SlidersHorizontal, Square } from "lucide-react";
import { useId, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router";
import { EmptyState, ErrorState, Notice, PageHeader, Segmented, StatusPill } from "@/components/common";
import { XYChart, type XYSeries } from "@/components/charts/XYChart";
import { seriesColor } from "@/components/charts/core";
import { Field, SelectField } from "@/components/fields";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { ApiError, type RunSpec, type SweepParam, type SweepState, api } from "@/lib/api";
import { duration, integer, sci } from "@/lib/format";
import { type Row, downloadText, loadLabConfig, toCsv } from "@/lib/runs";
import { SETTINGS, parseValues, settingsFor, showValue as show, sweepVerdict } from "@/lib/sweep";

const SEEDS = ["1", "2", "3", "5"] as const;

const describe = (spec: RunSpec) =>
  `${spec.arch === "kan" ? `KAN [${(spec.widths ?? []).join(", ")}], grid ${spec.grid}` : `MLP ${spec.hidden.join("-")} ${spec.act}`} · ${spec.optimizer === "lm" ? "Levenberg-Marquardt" : spec.optimizer}`;

function Results({ state }: { state: SweepState }) {
  const setting = SETTINGS[state.param];
  const labels = useMemo(() => state.rows.map((r) => show(r.value)), [state.rows]);
  const hasHole = state.rows.some((r) => r.rmse_gap != null);
  const series = useMemo<XYSeries[]>(() => {
    const at = state.rows.map((_, i) => i);
    const out: XYSeries[] = [{
      id: "rmse", label: state.known ? "RMSE against the exact f: median and range of the seeds" : "RMSE on held-out points: median and range of the seeds",
      color: seriesColor(0), x: at, y: state.rows.map((r) => r.rmse), lo: state.rows.map((r) => r.rmse_lo), hi: state.rows.map((r) => r.rmse_hi), marker: "circle",
    }];
    if (hasHole) out.push({ id: "gap", label: "In the missing middle, median", color: seriesColor(1), x: at, y: state.rows.map((r) => r.rmse_gap), marker: "square" });
    return out;
  }, [state.rows, state.known, hasHole]);
  const verdict = state.status === "running" ? null : sweepVerdict(state);
  const csv = () => downloadText(`sweep-${state.param}.csv`, toCsv(
    ["value", "n_params", "seeds", "rmse", "rmse_lo", "rmse_hi", "rmse_gap", "epochs", "seconds"], state.rows as unknown as Row[]));

  return (
    <div className="stagger space-y-4">
      <div className="space-y-1.5">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
          <h2 className="text-sm font-semibold" data-testid="sweep-title">{setting.label} · {state.target} · {state.label}</h2>
          <StatusPill status={state.status} />
          <span className="tnum text-xs text-muted-foreground" data-testid="sweep-progress">{state.done} of {state.total} trainings finished</span>
        </div>
        <div role="progressbar" aria-label="Trainings finished" aria-valuemin={0} aria-valuemax={state.total} aria-valuenow={state.done} className="h-1.5 overflow-hidden rounded-full bg-wash">
          <div className={`h-full rounded-full bg-primary transition-[width] duration-500 ${state.status === "running" ? "shimmer" : ""}`} style={{ width: `${(100 * state.done) / state.total}%` }} />
        </div>
      </div>
      {verdict && <Notice>{verdict}</Notice>}
      <Card className="gap-2 p-4">
        <h3 className="text-sm font-semibold">Error against {setting.label.toLowerCase()}</h3>
        <XYChart series={series} xCategories={labels} yScale="log" xLabel={setting.label} yLabel="RMSE" height={300} ariaLabel={`Error against ${setting.label.toLowerCase()}`} />
      </Card>
      <Card className="gap-0 overflow-hidden p-0">
        <div className="flex items-center justify-between gap-2 border-b px-4 py-2">
          <h3 className="text-sm font-semibold">Every value</h3>
          <Button variant="outline" size="sm" onClick={csv}><Download /> Download as CSV</Button>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-[13px]">
            <caption className="sr-only">Results of the study, one row per value</caption>
            <thead className="bg-wash text-left text-xs text-muted-foreground">
              <tr>
                <th scope="col" className="px-3 py-2 font-medium">{setting.label}</th>
                <th scope="col" className="px-3 py-2 text-right font-medium">Parameters</th>
                <th scope="col" className="px-3 py-2 text-right font-medium">RMSE, median</th>
                <th scope="col" className="px-3 py-2 text-right font-medium">Lowest to highest seed</th>
                {hasHole && <th scope="col" className="px-3 py-2 text-right font-medium">In the missing middle</th>}
                <th scope="col" className="px-3 py-2 text-right font-medium">Epochs run</th>
                <th scope="col" className="px-3 py-2 text-right font-medium">Time per run</th>
                <th scope="col" className="px-3 py-2 text-right font-medium">Best run</th>
              </tr>
            </thead>
            <tbody>
              {state.rows.map((r) => (
                <tr key={r.value} className={`border-t ${r.value === state.best && state.status !== "running" ? "bg-wash font-medium" : ""}`}>
                  <td className="tnum px-3 py-1.5">{show(r.value)}</td>
                  <td className="tnum px-3 py-1.5 text-right">{integer(r.n_params)}</td>
                  <td className="tnum px-3 py-1.5 text-right">{r.rmse == null ? `${r.done} of ${r.seeds} done` : sci(r.rmse)}</td>
                  <td className="tnum px-3 py-1.5 text-right whitespace-nowrap">{r.rmse_lo == null ? "—" : `${sci(r.rmse_lo)} to ${sci(r.rmse_hi)}`}</td>
                  {hasHole && <td className="tnum px-3 py-1.5 text-right">{sci(r.rmse_gap)}</td>}
                  <td className="tnum px-3 py-1.5 text-right">{integer(r.epochs)}</td>
                  <td className="tnum px-3 py-1.5 text-right whitespace-nowrap">{duration(r.seconds)}</td>
                  <td className="px-3 py-1.5 text-right">
                    {r.best_run ? <Link className="underline underline-offset-2" to={`/lab?run=${r.best_run}`}>Open in the Lab</Link> : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}

export function Sweep() {
  const [params, setParams] = useSearchParams();
  const [base] = useState(loadLabConfig);
  const options = useMemo(() => settingsFor(base), [base]);
  const [param, setParam] = useState<SweepParam>(options[0]);
  const [text, setText] = useState(SETTINGS[options[0]].values.map(show).join(", "));
  const [seeds, setSeeds] = useState<(typeof SEEDS)[number]>("3");
  const valuesId = useId();
  const values = parseValues(text);

  const id = params.get("sweep");
  const sweep = useQuery({
    queryKey: ["sweep", id],
    queryFn: () => api.sweep(id!),
    enabled: !!id,
    retry: false,
    refetchInterval: (query) => (query.state.data?.status === "running" ? 800 : false),
  });
  const start = useMutation({
    mutationFn: () => api.startSweep({ base, param, values: values!, seeds: Number(seeds) }),
    onSuccess: (state) => setParams({ sweep: state.id }, { replace: true }),
  });
  const stop = useMutation({ mutationFn: api.stopSweep, onSuccess: () => sweep.refetch() });
  const running = sweep.data?.status === "running";
  const error = start.error instanceof ApiError ? start.error.message : start.error ? "Could not start the study." : null;

  return (
    <div className="space-y-4">
      <PageHeader
        title="Sensitivity"
        description="Change one setting, keep the rest, and train every value on several seeds. The seeds show how much of a difference is only the luck of the starting weights."
      />
      <div className="grid grid-cols-1 items-start gap-4 lg:grid-cols-[300px_minmax(0,1fr)]">
        <Card className="gap-4 p-4">
          <div className="space-y-1">
            <h2 className="text-sm font-semibold">Everything that stays the same</h2>
            <p className="text-xs text-muted-foreground" data-testid="sweep-base">
              {base.formula ?? (base.points ? `${base.points.length} measured points` : base.target)} · {describe(base)}
            </p>
            <p className="text-xs text-subtle">
              These are the settings of the <Link className="underline underline-offset-2" to="/lab">Training Lab</Link>. Change them there.
            </p>
          </div>
          <SelectField
            label="Setting to vary"
            value={param}
            onChange={(next) => { setParam(next); setText(SETTINGS[next].values.map(show).join(", ")); }}
            options={options.map((key) => ({ value: key, label: SETTINGS[key].label }))}
            help={SETTINGS[param].hint}
          />
          <Field label="Values" htmlFor={valuesId} error={values ? undefined : "Give 2 to 12 different numbers, separated by commas."}>
            <Input id={valuesId} value={text} className="h-8 font-mono text-xs" spellCheck={false} autoComplete="off" aria-invalid={!values} aria-describedby={values ? undefined : `${valuesId}-error`} onChange={(e) => setText(e.target.value)} />
          </Field>
          <div className="space-y-1">
            <div className="text-xs font-medium text-muted-foreground">Seeds per value</div>
            <Segmented label="Seeds per value" value={seeds} onChange={setSeeds} options={SEEDS.map((s) => ({ value: s, label: s }))} />
            <p className="text-xs text-subtle">{values ? `${values.length * Number(seeds)} trainings, one after another.` : "Three seeds are the least that show a spread."}</p>
          </div>
          {error && <p role="alert" className="text-xs text-critical-ink">{error}</p>}
          {running ? (
            <Button type="button" variant="secondary" onClick={() => stop.mutate(id!)}><Square /> Stop</Button>
          ) : (
            <Button type="button" disabled={!values || start.isPending} onClick={() => start.mutate()}><Play /> Run the study</Button>
          )}
        </Card>

        <div className="min-w-0">
          {sweep.data ? <Results state={sweep.data} />
            : id && sweep.isError ? (
              <ErrorState error={new Error("This study is no longer available: studies live in the server's memory and end when it restarts.")} onRetry={() => setParams({}, { replace: true })} />
            ) : (
              <EmptyState icon={<SlidersHorizontal className="size-6" />} title="No study yet">
                Pick a setting and press Run. A flat line means the device does not care about that setting; a steep one shows where it breaks.
                Every point is the median of the seeds, and the band is their range.
              </EmptyState>
            )}
        </div>
      </div>
    </div>
  );
}

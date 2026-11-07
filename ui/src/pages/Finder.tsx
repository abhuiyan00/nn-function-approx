import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Download, ExternalLink, Search, Square } from "lucide-react";
import { type ReactNode, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router";
import { ErrorState, LoadingBlock, Notice, PageHeader, Segmented, Stat, StatusPill, useTargets } from "@/components/common";
import { XYChart, type XYSeries } from "@/components/charts/XYChart";
import { seriesColor } from "@/components/charts/core";
import { NumberField } from "@/components/fields";
import { FitView } from "@/components/FitView";
import { FunctionPicker, MissingMiddle } from "@/components/FunctionPicker";
import { PredictForm } from "@/components/PredictForm";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { useFunctionPreview } from "@/hooks/useFunctionPreview";
import { ApiError, type RunSpec, type SearchRow, type SearchState, api } from "@/lib/api";
import { integer, sci } from "@/lib/format";
import { boxOf, curvesCsv, describeSearch, fitRegions, functionFields, isCustom } from "@/lib/functions";
import { defaultSpec, downloadText } from "@/lib/runs";

const CONFIG_KEY = "funcprox-finder-config";
const START: RunSpec = { ...defaultSpec, target: "m1", n_train: 1000, n_val: 250 };
const DEFAULT_CURVES = 3;

function loadConfig(): RunSpec {
  try {
    const raw = localStorage.getItem(CONFIG_KEY);
    if (raw) return { ...START, ...Object.fromEntries(Object.entries(JSON.parse(raw)).filter(([k]) => k in START)) };
  } catch {
    /* no stored configuration, or storage is blocked */
  }
  return START;
}

function Step({ n, title, children }: { n: number; title: string; children: ReactNode }) {
  return (
    <Card className="gap-3 p-4">
      <h2 className="flex items-center gap-2 text-sm font-semibold">
        <span aria-hidden="true" className="tnum flex size-5 items-center justify-center rounded-full bg-primary text-xs text-primary-foreground">{n}</span>
        {title}
      </h2>
      {children}
    </Card>
  );
}

function CurveChart({ state, shown }: { state: SearchState; shown: (row: SearchRow) => boolean }) {
  const grid = state.grid;
  const focus = state.best_network;
  const series = useMemo<XYSeries[]>(() => {
    if (grid.kind !== "1d") return [];
    const out: XYSeries[] = state.known ? [{ id: "exact", label: "Exact f", color: "var(--axis)", x: grid.x, y: grid.exact, width: 4.5, underlay: true }] : [];
    for (const r of state.rows) {
      if (!r.pred || !shown(r)) continue;
      const main = r.key === focus;
      out.push({ id: r.key, label: r.key, color: seriesColor(r.slot), x: grid.x, y: r.pred, width: main ? 2.25 : 1.25, lo: main ? r.lo ?? undefined : undefined, hi: main ? r.hi ?? undefined : undefined });
    }
    return out;
  }, [state, grid, focus, shown]);
  // a poor fit can leave the plot by orders of magnitude inside the hole: keep the function readable
  const yDomain = useMemo<[number, number] | undefined>(() => {
    const values = [...grid.exact, ...state.samples.y, ...(state.rows.find((r) => r.key === focus)?.pred ?? [])].filter((v): v is number => v != null);
    if (!values.length) return undefined;
    const lo = Math.min(...values);
    const hi = Math.max(...values);
    const pad = (hi - lo || 1) * 0.15;
    return [lo - pad, hi + pad];
  }, [grid, state.samples, state.rows, focus]);
  const regions = useMemo(() => (grid.kind === "1d" ? fitRegions(grid) : []), [grid]);
  const dots = useMemo(() => ({ ...state.samples, label: "Training samples" }), [state.samples]);

  if (grid.kind === "2d") {
    const row = state.rows.find((r) => r.key === focus);
    const residual = row?.pred?.map((v, i) => (v == null || grid.exact[i] == null ? null : v - grid.exact[i]!));
    return <FitView grid={grid} pred={row?.pred} residual={residual} samples={state.samples} />;
  }
  return <XYChart series={series} xLabel="x" yLabel="f(x)" height={340} yDomain={yDomain} regions={regions} dots={dots} ariaLabel="The curves found against the function" />;
}

function Ranking({ state, shown, onToggle }: { state: SearchState; shown: (row: SearchRow) => boolean; onToggle: (row: SearchRow, on: boolean) => void }) {
  const hole = state.rows.some((r) => r.rmse_gap != null);
  const seeds = state.rows.some((r) => r.seeds > 1);
  return (
    <div className="overflow-x-auto rounded-lg border bg-card">
      <table className="w-full text-[13px]">
        <caption className="sr-only">Every model, closest curve first</caption>
        <thead className="bg-wash text-left text-xs text-muted-foreground">
          <tr>
            <th scope="col" className="px-2 py-2 text-right font-medium"><abbr title="Rank" className="no-underline">#</abbr></th>
            {state.grid.kind === "1d" && <th scope="col" className="px-1 py-2 font-medium">Show</th>}
            <th scope="col" className="px-2 py-2 font-medium">Model</th>
            <th scope="col" className="px-2 py-2 text-right font-medium">RMSE</th>
            {seeds && <th scope="col" className="px-2 py-2 text-right font-medium whitespace-nowrap">Across starts</th>}
            {hole && <th scope="col" className="px-2 py-2 text-right font-medium whitespace-nowrap" title="RMSE in the missing middle">In the hole</th>}
            {hole && <th scope="col" className="px-2 py-2 text-right font-medium whitespace-nowrap" title="RMSE where it had training data">With data</th>}
            <th scope="col" className="px-2 py-2 text-right font-medium" title="Fitted parameters or coefficients">Size</th>
            <th scope="col" className="px-1 py-2 font-medium"><span className="sr-only">Open</span></th>
          </tr>
        </thead>
        <tbody>
          {state.rows.map((r, i) => (
            <tr key={r.key} data-testid="ranking-row" data-state={r.key === state.best_network ? "selected" : undefined} className="border-t data-[state=selected]:bg-wash">
              <td className="tnum px-2 py-1.5 text-right text-muted-foreground">{r.status === "done" ? i + 1 : "—"}</td>
              {state.grid.kind === "1d" && (
                <td className="px-1 py-1.5">
                  <Checkbox aria-label={`Show ${r.key} on the chart`} checked={shown(r)} disabled={!r.pred} onCheckedChange={(on) => onToggle(r, on === true)} />
                </td>
              )}
              <td className="px-2 py-1.5">
                <span className="flex flex-wrap items-center gap-x-2 gap-y-0.5">
                  <span aria-hidden="true" className="inline-block h-0.5 w-3 shrink-0 rounded" style={{ background: seriesColor(r.slot) }} />
                  <span className="font-medium">{r.key}</span>
                  {r.status !== "done" && <StatusPill status={r.status} />}
                  {r.key === state.winner && <span className="rounded-full bg-good-wash px-2 py-0.5 text-xs font-medium whitespace-nowrap text-good">Closest</span>}
                  {r.key === state.best_network && r.key !== state.winner && <span className="rounded-full bg-info-wash px-2 py-0.5 text-xs font-medium whitespace-nowrap text-info-ink">Closest network</span>}
                  {r.key === state.blind_pick && r.key !== state.winner && <span className="rounded-full bg-wash px-2 py-0.5 text-xs whitespace-nowrap text-muted-foreground">Best on validation</span>}
                </span>
              </td>
              <td className="tnum px-2 py-1.5 text-right font-medium whitespace-nowrap">{sci(r.rmse)}</td>
              {seeds && <td className="tnum px-2 py-1.5 text-right whitespace-nowrap text-muted-foreground">{r.seeds > 1 ? `${sci(r.rmse_lo)} to ${sci(r.rmse_hi)}` : "—"}</td>}
              {hole && <td className="tnum px-2 py-1.5 text-right whitespace-nowrap">{sci(r.rmse_gap)}</td>}
              {hole && <td className="tnum px-2 py-1.5 text-right whitespace-nowrap">{sci(r.rmse_seen)}</td>}
              <td className="tnum px-2 py-1.5 text-right">{integer(r.n_params)}</td>
              <td className="px-1 py-1 text-right">
                {r.best_run && (
                  <Button variant="ghost" size="icon-sm" asChild aria-label={`Open ${r.key} in the Training Lab`} title="Open in the Training Lab">
                    <Link to={`/lab?run=${r.best_run}`}><ExternalLink /></Link>
                  </Button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Results({ state, onStop }: { state: SearchState; onStop: () => void }) {
  const [choices, setChoices] = useState<Record<string, boolean>>({});
  const running = state.status === "running";
  const focus = state.rows.find((r) => r.key === state.best_network);
  const closest = useMemo(() => new Set(state.rows.filter((r) => r.pred).slice(0, DEFAULT_CURVES).map((r) => r.key)), [state.rows]);
  const shown = useMemo(() => (r: SearchRow) => choices[r.key] ?? (closest.has(r.key) || r.key === state.best_network), [choices, closest, state.best_network]);
  const answer = focus?.best_run && focus.status === "done" ? focus.best_run : null;
  const score = state.known ? "RMSE vs exact f" : "RMSE on held-out points";

  return (
    <div className="space-y-4" data-testid="search-results">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="min-w-0">
          <h2 className="text-sm font-semibold">{state.formula}</h2>
          <p className="tnum text-xs text-muted-foreground" data-testid="search-progress">{state.done} of {state.total} trainings finished</p>
        </div>
        <div className="flex items-center gap-2">
          <span data-testid="search-status"><StatusPill status={state.status} /></span>
          {running && <Button variant="secondary" size="sm" onClick={onStop}><Square /> Stop</Button>}
          {!running && (
            <Button variant="outline" size="sm" onClick={() => downloadText("closest-curves.csv", curvesCsv(state))}><Download /> Download curves</Button>
          )}
        </div>
      </div>
      {running && (
        <div role="progressbar" aria-label="Trainings finished" aria-valuemin={0} aria-valuemax={state.total} aria-valuenow={state.done} className="h-1.5 overflow-hidden rounded-full bg-wash">
          <div className="h-full bg-primary transition-[width]" style={{ width: `${(state.done / state.total) * 100}%` }} />
        </div>
      )}
      {focus && (
        <div className="grid grid-cols-2 gap-2 xl:grid-cols-4">
          <Stat label="Closest network" value={<span className="block text-sm leading-snug whitespace-normal">{focus.key}</span>} detail={running ? "So far" : state.winner === focus.key ? "The closest curve of all" : `${state.winner} is closer`} />
          <Stat label={score} value={<span data-testid="best-rmse">{sci(focus.rmse)}</span>} detail="Over the whole range" />
          {focus.rmse_gap != null && <Stat label="In the missing middle" value={sci(focus.rmse_gap)} detail="It saw no data here" />}
          {focus.rmse_seen != null && <Stat label="Where it had data" value={sci(focus.rmse_seen)} />}
        </div>
      )}
      <Card className="gap-2 p-4"><CurveChart state={state} shown={shown} /></Card>
      {!running && describeSearch(state).map((line) => <Notice key={line}>{line}</Notice>)}
      <Ranking state={state} shown={shown} onToggle={(row, on) => setChoices((c) => ({ ...c, [row.key]: on }))} />
      <p className="text-xs text-subtle">
        {state.known ? "RMSE is the error against the exact function over the whole range, hole included, and sets the ranking. " : "RMSE is the error on points the models never trained on, including any points you hid, and sets the ranking. "}
        Size counts fitted parameters or coefficients. Models pick their stopping point from validation points outside the hole, and classical fits choose their size the same way.
      </p>
      {answer && <PredictForm key={answer} title={`Ask ${focus!.key}`} dim={state.dim} onPredict={(rows) => api.predictRun(answer, rows)} />}
    </div>
  );
}

export function Finder() {
  const queryClient = useQueryClient();
  const targets = useTargets();
  const [params, setParams] = useSearchParams();
  const [spec, setSpecState] = useState<RunSpec>(loadConfig);
  const [seeds, setSeeds] = useState("1");
  const [effort, setEffort] = useState<"quick" | "thorough">("quick");
  const [searched, setSearched] = useState<string | null>(null);
  const { preview, error, checking } = useFunctionPreview(spec);

  const setSpec = (patch: Partial<RunSpec>) => setSpecState((s) => {
    const next = { ...s, ...patch };
    try {
      localStorage.setItem(CONFIG_KEY, JSON.stringify(next));
    } catch {
      /* only a convenience */
    }
    return next;
  });

  const searchId = params.get("search");
  const search = useQuery({
    queryKey: ["search", searchId],
    queryFn: () => api.search(searchId!),
    enabled: !!searchId,
    retry: false,
    refetchInterval: (query) => (query.state.data?.status === "running" ? 800 : false),
  });
  const start = useMutation({
    mutationFn: () => api.startSearch({ base: spec, effort, seeds: Number(seeds) }),
    onSuccess: (state) => {
      setSearched(JSON.stringify(functionFields(spec)));
      queryClient.setQueryData(["search", state.id], state);
      setParams({ search: state.id }, { replace: true });
    },
  });
  const stop = useMutation({
    mutationFn: () => api.stopSearch(searchId!),
    onSuccess: (state) => queryClient.setQueryData(["search", state.id], state),
  });

  if (targets.isError) return <ErrorState error={targets.error} onRetry={() => targets.refetch()} />;
  if (!targets.data) return <LoadingBlock className="h-[600px]" />;

  const state = search.data;
  const running = state?.status === "running";
  const changed = searched != null && searched !== JSON.stringify(functionFields(spec));
  const lost = search.error instanceof ApiError && search.error.status === 404;
  const serverError = start.error instanceof Error ? start.error.message : null;

  return (
    <div className="space-y-4">
      <PageHeader
        title="Curve Finder"
        description="Give a function, hide a part of it if you like, and find the closest curve. Every model trains on the same points, and every model is scored on the whole range."
      />
      <div className="grid grid-cols-1 items-start gap-4 lg:grid-cols-[340px_minmax(0,1fr)]">
        <div className="space-y-3">
          <Step n={1} title="Your function">
            <FunctionPicker spec={spec} targets={targets.data} error={error} onChange={setSpec} />
          </Step>
          <Step n={2} title="Missing middle">
            {preview ? <MissingMiddle box={boxOf(preview.grid)} gap={preview.grid.gap} onChange={(gap) => setSpec({ gap })} /> : <LoadingBlock className="h-12" />}
          </Step>
          <Step n={3} title="Find the closest curve">
            {spec.points == null && (
              <div className="grid grid-cols-2 gap-2">
                <NumberField label="Training points" value={spec.n_train} min={10} max={20000} integerOnly onChange={(n_train) => setSpec({ n_train })} />
                <NumberField label="Noise σ" value={spec.sigma} min={0} max={5} onChange={(sigma) => setSpec({ sigma })} />
              </div>
            )}
            <div className="space-y-1">
              <p className="text-xs font-medium text-muted-foreground">Starting points per model</p>
              <Segmented label="Starting points per model" value={seeds} onChange={setSeeds} options={[{ value: "1", label: "1" }, { value: "3", label: "3" }, { value: "5", label: "5" }]} />
              <p className="text-xs text-subtle">With more than one, a band on the chart shows how far the curve moves when only the starting weights change.</p>
            </div>
            <div className="space-y-1">
              <p className="text-xs font-medium text-muted-foreground">Training effort</p>
              <Segmented label="Training effort" value={effort} onChange={setEffort} options={[{ value: "quick", label: "Quick" }, { value: "thorough", label: "Thorough" }]} />
            </div>
            {serverError && <Notice tone="critical">{serverError}</Notice>}
            <Button disabled={!!error || checking || running || start.isPending} onClick={() => start.mutate()}><Search /> Find the closest curve</Button>
          </Step>
        </div>

        <div className="min-w-0 space-y-4">
          {(!state || changed) && preview && (
            <Card className="gap-2 p-4">
              <div>
                <h2 className="text-sm font-semibold" data-testid="preview-title">{isCustom(spec) ? preview.formula : `${preview.key} · ${preview.formula}`}</h2>
                <p className="text-xs text-muted-foreground">{changed ? "Changed since the search below. Find again to compare." : "Dots show where the training data falls. Nothing trained yet."}</p>
              </div>
              <FitView grid={preview.grid} samples={preview.samples} />
            </Card>
          )}
          {lost && <Notice tone="warning" title="This search is no longer available">Searches live in the server's memory and end when it restarts.</Notice>}
          {search.isError && !lost && <ErrorState error={search.error} onRetry={() => search.refetch()} />}
          {state && <Results key={state.id} state={state} onStop={() => stop.mutate()} />}
        </div>
      </div>
    </div>
  );
}

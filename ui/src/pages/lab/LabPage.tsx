import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { FlaskConical, RotateCcw, Save, Trash2 } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router";
import { toast } from "sonner";
import { EmptyState, ErrorState, LoadingBlock, Notice, PageHeader, Segmented, Stat, StatusPill, useTargets } from "@/components/common";
import { FitView } from "@/components/FitView";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { useRunStream } from "@/hooks/useRunStream";
import { ApiError, type RunDetail, type RunSpec, type RunSummary, api } from "@/lib/api";
import { duration, integer, sci } from "@/lib/format";
import { defaultSpec, validateSpec } from "@/lib/runs";
import { ConfigPanel } from "./ConfigPanel";
import { PRESETS, type Preset } from "./presets";
import { BaselinesPanel, ComparePanel, DampingPanel, EdgesPanel, LossChart, SpectrumPanel } from "./ResultPanels";

const CONFIG_KEY = "funcprox-lab-config";
const MAX_COMPARE = 4;
type Panel = "spectrum" | "damping" | "edges" | "baselines" | "compare";

function loadConfig(): RunSpec {
  try {
    const raw = localStorage.getItem(CONFIG_KEY);
    // only settings this version knows: the server refuses unknown ones
    if (raw) return { ...defaultSpec, ...Object.fromEntries(Object.entries(JSON.parse(raw)).filter(([k]) => k in defaultSpec)) };
  } catch {
    /* no stored configuration, or storage is blocked */
  }
  return defaultSpec;
}

function Metrics({ run }: { run: RunDetail }) {
  const m = run.final?.metrics;
  const live = run.live;
  const h = run.history;
  const lm = run.spec.optimizer === "lm";
  const epoch = h.epoch.length ? h.epoch[h.epoch.length - 1] : 0;
  const elapsed = run.final?.seconds ?? (h.wall.length ? h.wall[h.wall.length - 1] : null);
  return (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 xl:grid-cols-6">
      <Stat label="RMSE vs exact f" value={<span data-testid="metric-rmse">{sci(m?.rmse ?? live?.rmse)}</span>} detail={m ? "10,000-point grid" : "Live, coarse grid"} />
      <Stat label="Max error" value={sci(m?.max_err ?? live?.max_err)} detail={m?.argmax_in_edge ? "At the edge of the box" : m ? "Inside the box" : "Live"} />
      <Stat label="R²" value={(m?.r2 ?? live?.r2) == null ? "—" : Math.max(m?.r2 ?? live?.r2 ?? 0, -9.99).toFixed(5)} />
      <Stat label="Parameters" value={integer(run.n_params)} />
      <Stat label={lm ? "Iteration" : "Epoch"} value={integer(epoch)} detail={run.final ? `Best at ${run.final.best_epoch}` : `of ${integer(run.spec.epochs)}`} />
      <Stat label="Training time" value={duration(elapsed)} detail={run.final?.stop ? run.final.stop : undefined} />
    </div>
  );
}

function SaveDialog({ run, onClose }: { run: RunSummary; onClose: () => void }) {
  const queryClient = useQueryClient();
  const [name, setName] = useState(`${run.spec.target}-${run.label.split(" · ")[0].toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/-+$/, "")}`);
  const valid = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(name);
  const save = useMutation({
    mutationFn: () => api.saveRun(run.id, name),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["models"] });
      queryClient.invalidateQueries({ queryKey: ["lab-runs"] });
      toast.success(`Saved as ${name}`, { action: { label: "Open", onClick: () => { window.location.href = `/models?model=${name}`; } } });
      onClose();
    },
  });
  return (
    <Dialog open onOpenChange={(open) => !open && onClose()}>
      <DialogContent>
        <form onSubmit={(event) => { event.preventDefault(); if (valid) save.mutate(); }} className="space-y-4">
          <DialogHeader>
            <DialogTitle>Save this model</DialogTitle>
            <DialogDescription>Writes runs/{valid ? name : "…"}.npz, the same format as the command line. The runs folder is not tracked by git.</DialogDescription>
          </DialogHeader>
          <div className="space-y-1.5">
            <Label htmlFor="model-name">Name</Label>
            <Input id="model-name" value={name} autoFocus aria-invalid={!valid} onChange={(e) => setName(e.target.value)} />
            {!valid && <p className="text-xs text-critical-ink">Use letters, digits, dot, dash or underscore, up to 64 characters, starting with a letter or digit.</p>}
            {save.error && <p className="text-xs text-critical-ink">{(save.error as Error).message}</p>}
          </div>
          <DialogFooter>
            <Button type="button" variant="outline" onClick={onClose}>Cancel</Button>
            <Button type="submit" disabled={!valid || save.isPending}>Save model</Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

function History({ runs, selected, onSelect, compare, onCompare, onReuse, onSave, onDelete }: {
  runs: RunSummary[];
  selected: string | null;
  onSelect: (id: string) => void;
  compare: string[];
  onCompare: (id: string, on: boolean) => void;
  onReuse: (run: RunSummary) => void;
  onSave: (run: RunSummary) => void;
  onDelete: (run: RunSummary) => void;
}) {
  return (
    <Card className="gap-0 overflow-hidden p-0">
      <h2 className="border-b px-4 py-2.5 text-sm font-semibold">Runs this session</h2>
      <div className="overflow-x-auto">
        <table className="w-full text-[13px]">
          <caption className="sr-only">Training runs of this session</caption>
          <thead className="bg-wash text-left text-xs text-muted-foreground">
            <tr>
              <th scope="col" className="w-10 px-3 py-2 font-medium"><span className="sr-only">Compare</span></th>
              <th scope="col" className="px-3 py-2 font-medium">Run</th>
              <th scope="col" className="px-3 py-2 font-medium">Status</th>
              <th scope="col" className="px-3 py-2 text-right font-medium">RMSE</th>
              <th scope="col" className="px-3 py-2 text-right font-medium">Max error</th>
              <th scope="col" className="px-3 py-2 text-right font-medium">Params</th>
              <th scope="col" className="px-3 py-2 text-right font-medium">Epochs</th>
              <th scope="col" className="px-3 py-2 text-right font-medium">Time</th>
              <th scope="col" className="px-3 py-2 text-right font-medium">Actions</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => (
              <tr key={r.id} data-state={r.id === selected ? "selected" : undefined} className="border-t data-[state=selected]:bg-wash">
                <td className="px-3 py-1.5">
                  <Checkbox
                    aria-label={`Compare ${r.label}`}
                    checked={compare.includes(r.id)}
                    disabled={!compare.includes(r.id) && compare.length >= MAX_COMPARE}
                    onCheckedChange={(on) => onCompare(r.id, on === true)}
                  />
                </td>
                <td className="px-3 py-1.5">
                  <button type="button" onClick={() => onSelect(r.id)} className="text-left font-medium whitespace-nowrap underline-offset-2 hover:underline" aria-current={r.id === selected}>
                    {r.spec.target} · {r.label}
                  </button>
                </td>
                <td className="px-3 py-1.5"><StatusPill status={r.status} /></td>
                <td className="tnum px-3 py-1.5 text-right font-medium">{sci(r.rmse)}</td>
                <td className="tnum px-3 py-1.5 text-right">{sci(r.max_err)}</td>
                <td className="tnum px-3 py-1.5 text-right">{integer(r.n_params)}</td>
                <td className="tnum px-3 py-1.5 text-right">{integer(r.epochs_run)}</td>
                <td className="tnum px-3 py-1.5 text-right whitespace-nowrap">{duration(r.seconds)}</td>
                <td className="px-3 py-1 text-right whitespace-nowrap">
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button variant="ghost" size="icon-sm" aria-label={`Reuse the settings of ${r.label}`} onClick={() => onReuse(r)}><RotateCcw /></Button>
                    </TooltipTrigger>
                    <TooltipContent>Reuse settings</TooltipContent>
                  </Tooltip>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button variant="ghost" size="icon-sm" aria-label={r.saved_as ? `Saved as ${r.saved_as}` : `Save ${r.label}`} disabled={!(r.status === "done" || r.status === "stopped") || !!r.saved_as || r.rmse == null} onClick={() => onSave(r)}><Save /></Button>
                    </TooltipTrigger>
                    <TooltipContent>{r.saved_as ? `Saved as ${r.saved_as}` : "Save model"}</TooltipContent>
                  </Tooltip>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button variant="ghost" size="icon-sm" aria-label={`Remove ${r.label} from this list`} onClick={() => onDelete(r)}><Trash2 /></Button>
                    </TooltipTrigger>
                    <TooltipContent>Remove from list</TooltipContent>
                  </Tooltip>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

function PresetRow({ onRun, disabled }: { onRun: (preset: Preset) => void; disabled: boolean }) {
  return (
    <div className="space-y-1.5">
      <p className="text-xs font-medium text-muted-foreground">Reproduce a finding: one click loads the settings and trains</p>
      <div className="flex flex-wrap gap-1.5">
        {PRESETS.map((p) => (
          <Tooltip key={p.id}>
            <TooltipTrigger asChild>
              <Button variant="outline" size="sm" disabled={disabled} onClick={() => onRun(p)} data-testid={`preset-${p.id}`}>
                <span className="tnum text-xs text-subtle">{p.source}</span>
                {p.label}
              </Button>
            </TooltipTrigger>
            <TooltipContent className="max-w-xs">{p.hint}</TooltipContent>
          </Tooltip>
        ))}
      </div>
    </div>
  );
}

export function LabPage() {
  const queryClient = useQueryClient();
  const targets = useTargets();
  const [params, setParams] = useSearchParams();
  const [spec, setSpec] = useState<RunSpec>(() => {
    const config = loadConfig();
    const asked = params.get("target");
    return asked ? { ...config, target: asked, arch: "mlp", widths: null, grid_updates: [] } : config;
  });
  const [panel, setPanel] = useState<Panel>("baselines");
  const [compare, setCompare] = useState<string[]>([]);
  const [saving, setSaving] = useState<RunSummary | null>(null);
  const [hint, setHint] = useState<string | null>(null);
  const results = useRef<HTMLDivElement>(null);

  const runId = params.get("run");
  const run = useRunStream(runId);
  const runs = useQuery({
    queryKey: ["lab-runs"],
    queryFn: api.runs,
    refetchInterval: (query) => (query.state.data?.some((r) => r.status === "queued" || r.status === "running") ? 1500 : false),
  });

  useEffect(() => {
    try {
      localStorage.setItem(CONFIG_KEY, JSON.stringify(spec));
    } catch {
      /* only a convenience */
    }
  }, [spec]);

  const select = (id: string | null) => setParams((p) => {
    const next = new URLSearchParams(p);
    next.delete("target");
    if (id) next.set("run", id);
    else next.delete("run");
    return next;
  }, { replace: true });

  const start = useMutation({
    mutationFn: api.startRun,
    onSuccess: (summary) => {
      queryClient.invalidateQueries({ queryKey: ["lab-runs"] });
      select(summary.id);
      // on a phone the results sit below the whole form
      if (window.matchMedia("(max-width: 1023px)").matches) results.current?.scrollIntoView();
    },
  });
  const stop = useMutation({ mutationFn: api.stopRun });
  const remove = useMutation({
    mutationFn: api.deleteRun,
    onSuccess: (_void, id) => {
      setCompare((c) => c.filter((x) => x !== id));
      if (id === runId) select(null);
      queryClient.invalidateQueries({ queryKey: ["lab-runs"] });
    },
  });

  const known = targets.data?.find((t) => t.key === spec.target);
  const target = known ?? targets.data?.[0];
  useEffect(() => {
    if (targets.data && !known) setSpec((s) => ({ ...s, target: targets.data[0].key })); // stale storage or a bad address
  }, [targets.data, known]);
  const errors = useMemo(() => (target ? validateSpec(spec, target.dim) : {}), [spec, target]);
  const active = run && (run.status === "queued" || run.status === "running");
  const serverError = start.error instanceof ApiError ? start.error.message : start.error ? "Could not start the run." : null;

  const panels = useMemo(() => {
    const options: { value: Panel; label: string }[] = [{ value: "baselines", label: "Classical baselines" }];
    if (run?.grid.spectrum_exact) options.unshift({ value: "spectrum", label: "Spectrum" });
    if (run?.spec.optimizer === "lm") options.push({ value: "damping", label: "Damping μ" });
    if (run?.spec.arch === "kan") options.push({ value: "edges", label: "Edges" });
    options.push({ value: "compare", label: `Compare${compare.length ? ` (${compare.length})` : ""}` });
    return options;
  }, [run?.grid.spectrum_exact, run?.spec.optimizer, run?.spec.arch, compare.length]);
  const shownPanel = panels.some((p) => p.value === panel) ? panel : panels[0].value;

  const runPreset = (preset: Preset) => {
    setSpec(preset.spec);
    setHint(preset.hint);
    setPanel(preset.id === "kan-f7" ? "edges" : preset.id.startsWith("spectral") ? "spectrum" : "baselines");
    start.mutate(preset.spec);
  };

  if (targets.isError) return <ErrorState error={targets.error} onRetry={() => targets.refetch()} />;
  if (!targets.data || !target) return <LoadingBlock className="h-[600px]" />;

  const pred = run?.final?.pred ?? run?.live?.pred;
  const residual = run?.final?.residual ?? run?.live?.residual;

  return (
    <div className="space-y-4">
      <PageHeader
        title="Training Lab"
        description="Train a network on the engine and watch it approximate the function. Every error is measured against the exact f."
      />
      <PresetRow onRun={runPreset} disabled={start.isPending} />

      <div className="grid grid-cols-1 items-start gap-4 lg:grid-cols-[300px_minmax(0,1fr)]">
        <ConfigPanel
          spec={spec}
          onChange={(next) => { setSpec(next); setHint(null); }}
          targets={targets.data}
          errors={errors}
          serverError={serverError}
          running={!!active}
          onStart={() => { setHint(null); start.mutate(spec); }}
          onStop={() => run && stop.mutate(run.id)}
        />

        <div ref={results} className="min-w-0 scroll-mt-4 space-y-4">
          {run ? (
            <>
              <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
                <h2 className="text-sm font-semibold" data-testid="run-title">{run.spec.target} · {run.label}</h2>
                <span className="text-xs text-muted-foreground">f = {run.formula}</span>
                <span data-testid="run-status"><StatusPill status={run.status} /></span>
                {run.saved_as && <Link className="text-xs underline underline-offset-2" to={`/models?model=${run.saved_as}`}>Saved as {run.saved_as}</Link>}
              </div>
              {hint && <Notice>{hint}</Notice>}
              {run.status === "failed" && <Notice tone="critical" title="Training failed">{run.error}</Notice>}
              {run.final?.stop === "diverged" && (
                <Notice tone="warning" title="Training diverged">The loss stopped being a finite number. Try a smaller learning rate.</Notice>
              )}
              <Metrics run={run} />
              <div className="grid grid-cols-1 gap-4 2xl:grid-cols-2">
                <Card className="gap-2 p-4">
                  <h3 className="text-sm font-semibold">Fit against the exact function</h3>
                  <FitView grid={run.grid} pred={pred} residual={residual} samples={run.samples} />
                </Card>
                <Card className="gap-2 p-4">
                  <h3 className="text-sm font-semibold">Learning curves</h3>
                  <p className="-mt-1 text-xs text-muted-foreground">The exact-f curve is what a real user of the device cannot see: it reveals overfitting directly.</p>
                  <LossChart run={run} />
                </Card>
              </div>
              <Card className="gap-3 p-4">
                <Segmented label="Analysis" value={shownPanel} onChange={setPanel} options={panels} className="self-start" />
                {shownPanel === "spectrum" && <SpectrumPanel run={run} />}
                {shownPanel === "damping" && <DampingPanel run={run} />}
                {shownPanel === "edges" && <EdgesPanel run={run} />}
                {shownPanel === "baselines" && <BaselinesPanel run={run} />}
                {shownPanel === "compare" && <ComparePanel ids={compare} />}
              </Card>
            </>
          ) : runId && !runs.data?.some((r) => r.id === runId) && runs.isSuccess ? (
            <EmptyState
              icon={<FlaskConical className="size-6" />}
              title="This run is no longer available"
              action={<Button variant="outline" onClick={() => select(null)}>Back to the lab</Button>}
            >
              Runs live in the server's memory and end when it restarts. Saved models stay.
            </EmptyState>
          ) : (
            <Card className="gap-3 p-4">
              <div>
                <h2 className="text-sm font-semibold">{target.key} · {target.formula}</h2>
                <p className="text-xs text-muted-foreground">Nothing trained yet. Press Train, or pick a finding above.</p>
              </div>
              <FitView grid={target.preview} />
            </Card>
          )}

          {!!runs.data?.length && (
            <History
              runs={runs.data}
              selected={runId}
              onSelect={select}
              compare={compare}
              onCompare={(id, on) => setCompare((c) => (on ? [...c, id] : c.filter((x) => x !== id)))}
              onReuse={(r) => { setSpec({ ...defaultSpec, ...r.spec }); toast.success("Settings loaded into the form"); }}
              onSave={setSaving}
              onDelete={(r) => remove.mutate(r.id)}
            />
          )}
        </div>
      </div>
      {saving && <SaveDialog run={saving} onClose={() => setSaving(null)} />}
    </div>
  );
}

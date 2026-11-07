import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Play, Square, Terminal } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router";
import { EmptyState, ErrorState, LoadingBlock, Notice, PageHeader, Segmented, StatusPill } from "@/components/common";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { type Job, api } from "@/lib/api";
import { clock, duration } from "@/lib/format";

function useJobLog(job: Job | null) {
  const [lines, setLines] = useState<string[]>([]);
  const queryClient = useQueryClient();
  const id = job?.id;

  useEffect(() => {
    setLines([]);
    if (!id) return;
    const source = new EventSource(`/api/jobs/${id}/stream`);
    source.addEventListener("lines", (e) => setLines((old) => old.concat(JSON.parse(e.data))));
    source.addEventListener("done", () => {
      source.close();
      for (const key of ["jobs", "experiments", "experiment", "overview", "dataset", "datasets"]) {
        queryClient.invalidateQueries({ queryKey: [key] }); // new results are on disk now
      }
    });
    return () => source.close();
  }, [id, queryClient]);

  return lines;
}

function LogView({ job }: { job: Job }) {
  const lines = useJobLog(job);
  const box = useRef<HTMLPreElement>(null);
  useEffect(() => {
    const el = box.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [lines]);
  return (
    <pre ref={box} tabIndex={0} aria-label={`Log of ${job.experiment}`} className="h-72 overflow-auto rounded-md border bg-wash p-3 font-mono text-xs leading-relaxed whitespace-pre-wrap">
      {lines.length ? lines.join("\n") : job.status === "queued" ? "Waiting for the job ahead of it." : "No output yet."}
    </pre>
  );
}

function elapsed(job: Job) {
  if (!job.started) return "—";
  return duration((job.finished ?? Date.now() / 1000) - job.started);
}

export function Runner() {
  const queryClient = useQueryClient();
  const [params] = useSearchParams();
  const experiments = useQuery({ queryKey: ["experiments"], queryFn: api.experiments });
  const health = useQuery({ queryKey: ["health"], queryFn: api.health });
  const jobs = useQuery({
    queryKey: ["jobs"],
    queryFn: api.jobs,
    refetchInterval: (query) => (query.state.data?.some((j) => j.status === "queued" || j.status === "running") ? 2000 : false),
  });

  const [picked, setPicked] = useState<string[]>(() => {
    const asked = params.get("experiment");
    return asked ? [asked] : [];
  });
  const [profile, setProfile] = useState<"quick" | "full">("quick");
  const [workers, setWorkers] = useState("2");
  const [confirming, setConfirming] = useState(false);
  const [openJob, setOpenJob] = useState<string | null>(null);

  const start = useMutation({
    mutationFn: async () => {
      const made: Job[] = [];
      for (const experiment of experiments.data!.filter((e) => picked.includes(e.id))) {
        made.push(await api.startJob(experiment.id, profile, Number(workers)));
      }
      return made;
    },
    onSuccess: (made) => {
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
      if (made[0]) setOpenJob(made[0].id);
      setConfirming(false);
    },
  });
  const cancel = useMutation({
    mutationFn: api.cancelJob,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["jobs"] }),
  });

  if (experiments.isError) return <ErrorState error={experiments.error} onRetry={() => experiments.refetch()} />;
  if (!experiments.data) return <LoadingBlock />;

  const all = experiments.data;
  const everything = picked.length === all.length;
  const shownJob = jobs.data?.find((j) => j.id === openJob) ?? jobs.data?.[0] ?? null;
  const cpus = health.data?.cpus ?? 2;
  const workerOptions = [1, 2, 4, 8].filter((n) => n <= cpus);

  return (
    <div className="space-y-4">
      <PageHeader title="Experiment Runner" description="Run the experiment scripts from here, one at a time, and follow their output. Finished results appear in the rest of the interface straight away." />

      <div className="grid grid-cols-1 items-start gap-4 xl:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
        <Card className="gap-3 p-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h2 className="text-sm font-semibold">Choose experiments</h2>
            <Button variant="ghost" size="sm" onClick={() => setPicked(everything ? [] : all.map((e) => e.id))}>{everything ? "Clear all" : "Select all"}</Button>
          </div>
          <ul className="divide-y rounded-md border">
            {all.map((e) => (
              <li key={e.id} className="flex items-center gap-2.5 px-3 py-1.5">
                <Checkbox
                  id={`run-${e.id}`}
                  checked={picked.includes(e.id)}
                  onCheckedChange={(on) => setPicked((p) => (on ? [...p, e.id] : p.filter((x) => x !== e.id)))}
                />
                <Label htmlFor={`run-${e.id}`} className="block min-w-0 flex-1 truncate text-[13px] font-normal" title={e.title}>
                  <span className="tnum font-medium">{e.id.toUpperCase()}</span> {e.title}
                </Label>
                <span className="tnum shrink-0 text-xs text-subtle">{e.profile ? `${e.profile}, ${e.generated?.slice(5, 10)}` : "not run"}</span>
              </li>
            ))}
          </ul>
          <div className="flex flex-wrap items-end gap-4">
            <div className="space-y-1">
              <span className="text-xs font-medium text-muted-foreground">Profile</span>
              <div>
                <Segmented label="Profile" value={profile} onChange={setProfile} options={[{ value: "quick", label: "Quick: minutes" }, { value: "full", label: "Full: hours" }]} />
              </div>
            </div>
            <div className="w-28 space-y-1">
              <Label htmlFor="workers" className="text-xs font-medium text-muted-foreground">Worker processes</Label>
              <Select value={workers} onValueChange={setWorkers}>
                <SelectTrigger id="workers" size="sm" className="w-full"><SelectValue /></SelectTrigger>
                <SelectContent>{(workerOptions.length ? workerOptions : [1]).map((n) => <SelectItem key={n} value={String(n)}>{n}</SelectItem>)}</SelectContent>
              </Select>
            </div>
            <Button className="ml-auto" disabled={picked.length === 0} onClick={() => setConfirming(true)}>
              <Play /> Run {picked.length || ""}
            </Button>
          </div>
        </Card>

        <Card className="gap-3 p-4">
          <h2 className="text-sm font-semibold">Queue</h2>
          {jobs.isError && <ErrorState error={jobs.error} onRetry={() => jobs.refetch()} />}
          {jobs.data?.length ? (
            <>
              <div className="max-h-52 overflow-auto rounded-md border">
                <table className="w-full text-[13px]">
                  <caption className="sr-only">Experiment jobs, newest first</caption>
                  <thead className="sticky top-0 bg-wash text-left text-xs text-muted-foreground">
                    <tr>
                      <th scope="col" className="px-3 py-2 font-medium">Experiment</th>
                      <th scope="col" className="px-3 py-2 font-medium">Status</th>
                      <th scope="col" className="px-3 py-2 text-right font-medium">Time</th>
                      <th scope="col" className="px-3 py-2 font-medium">Started</th>
                      <th scope="col" className="px-3 py-2"><span className="sr-only">Actions</span></th>
                    </tr>
                  </thead>
                  <tbody>
                    {jobs.data.map((j) => (
                      <tr key={j.id} data-state={shownJob?.id === j.id ? "selected" : undefined} className="border-t data-[state=selected]:bg-wash">
                        <td className="px-3 py-1.5 whitespace-nowrap">
                          <button type="button" className="font-medium underline-offset-2 hover:underline" aria-current={shownJob?.id === j.id} onClick={() => setOpenJob(j.id)}>
                            {j.experiment.toUpperCase()} · {j.profile}
                          </button>
                        </td>
                        <td className="px-3 py-1.5"><StatusPill status={j.status} /></td>
                        <td className="tnum px-3 py-1.5 text-right whitespace-nowrap">{elapsed(j)}</td>
                        <td className="tnum px-3 py-1.5 whitespace-nowrap text-muted-foreground">{j.started ? clock(j.started) : "—"}</td>
                        <td className="px-3 py-1 text-right">
                          {(j.status === "queued" || j.status === "running") && (
                            <Button variant="ghost" size="sm" onClick={() => cancel.mutate(j.id)}><Square /> Cancel</Button>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {shownJob && <LogView key={shownJob.id} job={shownJob} />}
            </>
          ) : (
            <EmptyState icon={<Terminal className="size-6" />} title="No jobs yet">Choose experiments on the left and press Run. Output appears here as it happens.</EmptyState>
          )}
        </Card>
      </div>

      <Dialog open={confirming} onOpenChange={setConfirming}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Run {picked.length} experiment{picked.length === 1 ? "" : "s"} on the {profile} profile?</DialogTitle>
            <DialogDescription>
              This overwrites the current results in results/ and docs/figures/ and rewrites docs/RESULTS.md. Those files are tracked by git: commit first if you want to keep what is there.
            </DialogDescription>
          </DialogHeader>
          {profile === "full" && (
            <Notice tone="warning" title="The full profile takes hours">Five seeds and 20,000 points. Keep this machine awake and the server running: stopping `python -m funcprox ui` stops the experiment. Closing the browser tab does not.</Notice>
          )}
          {start.error && <Notice tone="critical">{(start.error as Error).message}</Notice>}
          <DialogFooter>
            <Button variant="outline" onClick={() => setConfirming(false)}>Keep current results</Button>
            <Button onClick={() => start.mutate()} disabled={start.isPending}>Overwrite and run</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

import { useQuery } from "@tanstack/react-query";
import { ArrowRight, CircleDot, Equal, Sigma } from "lucide-react";
import { useEffect, useRef } from "react";
import { Link } from "react-router";
import { RunsChart } from "@/components/charts/RunsChart";
import { sequentialRamp } from "@/components/charts/core";
import { ErrorState, LoadingBlock, Notice, PageHeader, Section, Stat, useTargets } from "@/components/common";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { useTheme } from "@/hooks/useTheme";
import { type Grid, type Overview as OverviewData, type TimelineItem, type Verdict, api } from "@/lib/api";
import { EXPERIMENT_CHARTS } from "@/lib/experimentCharts";
import { daysUntil, factor, integer, percent, sci, shortDate } from "@/lib/format";

function Preview({ grid }: { grid: Grid }) {
  const { theme } = useTheme();
  const canvas = useRef<HTMLCanvasElement>(null);
  const exact = grid.exact as number[]; // the targets of this page are defined everywhere

  useEffect(() => {
    if (grid.kind !== "2d") return;
    const ctx = canvas.current?.getContext("2d");
    if (!ctx) return;
    const nx = grid.x.length;
    const ny = grid.y.length;
    const lo = Math.min(...exact);
    const hi = Math.max(...exact);
    const colour = sequentialRamp(theme);
    const image = ctx.createImageData(nx, ny);
    for (let i = 0; i < nx; i++) {
      for (let j = 0; j < ny; j++) {
        const p = ((ny - 1 - j) * nx + i) * 4;
        const [r, g, b] = colour((exact[i * ny + j] - lo) / (hi - lo || 1));
        image.data.set([r, g, b, 255], p);
      }
    }
    ctx.putImageData(image, 0, 0);
  }, [grid, exact, theme]);

  if (grid.kind === "2d") {
    return <canvas ref={canvas} width={grid.x.length} height={grid.y.length} aria-hidden="true" className="h-16 w-full rounded-sm" />;
  }
  const lo = Math.min(...exact);
  const hi = Math.max(...exact);
  const x0 = grid.x[0];
  const x1 = grid.x[grid.x.length - 1];
  const place = (x: number) => ((x - x0) / (x1 - x0)) * 200;
  const d = grid.x
    .map((x, i) => `${i ? "L" : "M"}${place(x).toFixed(1)},${(58 - ((exact[i] - lo) / (hi - lo || 1)) * 52).toFixed(1)}`)
    .join("");
  return (
    <svg viewBox="0 0 200 64" preserveAspectRatio="none" aria-hidden="true" className="h-16 w-full">
      {grid.gap && <rect x={place(grid.gap[0][0])} y={0} width={place(grid.gap[0][1]) - place(grid.gap[0][0])} height={64} fill="var(--wash)" />}
      <path d={d} fill="none" stroke="var(--series-1)" strokeWidth="1.75" vectorEffect="non-scaling-stroke" strokeLinejoin="round" />
    </svg>
  );
}

const VERDICT = {
  classical: { label: "Classical wins", icon: Sigma },
  network: { label: "Network wins", icon: CircleDot },
  tie: { label: "Within seed spread", icon: Equal },
};

function VerdictTable({ verdicts }: { verdicts: Verdict[] }) {
  return (
    <div className="overflow-x-auto rounded-lg border bg-card">
      <table className="w-full text-[13px]">
        <caption className="sr-only">Best network against best classical approximator for each target of experiment E6</caption>
        <thead className="bg-wash text-left text-xs text-muted-foreground">
          <tr>
            <th scope="col" className="px-3 py-2 font-medium">Target</th>
            <th scope="col" className="px-3 py-2 text-right font-medium">Best network</th>
            <th scope="col" className="px-3 py-2 text-right font-medium">Best classical</th>
            <th scope="col" className="px-3 py-2 font-medium">Verdict</th>
            <th scope="col" className="px-3 py-2 text-right font-medium">Gap</th>
          </tr>
        </thead>
        <tbody>
          {verdicts.map((v) => {
            const Icon = VERDICT[v.winner].icon;
            return (
              <tr key={v.target} className="border-t">
                <td className="px-3 py-2 whitespace-nowrap"><span className="font-medium">{v.target}</span> <span className="text-muted-foreground">{v.formula}</span></td>
                <td className="tnum px-3 py-2 text-right whitespace-nowrap" title={v.network.model}>{sci(v.network.rmse)} <span className="text-xs text-subtle">at {integer(v.network.n_params)}</span></td>
                <td className="tnum px-3 py-2 text-right whitespace-nowrap" title={v.classical.model}>{sci(v.classical.rmse)} <span className="text-xs text-subtle">at {integer(v.classical.n_params)}</span></td>
                <td className="px-3 py-2 whitespace-nowrap">
                  <span className="inline-flex items-center gap-1.5"><Icon className="size-3.5 text-subtle" aria-hidden="true" />{VERDICT[v.winner].label}</span>
                </td>
                <td className="tnum px-3 py-2 text-right font-medium whitespace-nowrap">{factor(v.ratio)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function Timeline({ items }: { items: TimelineItem[] }) {
  if (!items.length) return <p className="text-xs text-subtle">No plan found in docs/PROJECT_PLAN.md.</p>;
  const time = (iso: string) => {
    const [y, m, d] = iso.split("-").map(Number);
    return Date.UTC(y, m - 1, d);
  };
  const first = time(items[0].start);
  const last = Math.max(...items.map((t) => time(t.end)));
  const pos = (iso: string) => ((time(iso) - first) / (last - first || 1)) * 100;
  const today = new Date();
  const todayPos = ((Date.UTC(today.getFullYear(), today.getMonth(), today.getDate()) - first) / (last - first || 1)) * 100;
  const planned = items.filter((t) => !t.done);

  return (
    <ol className="space-y-1.5">
      {planned.map((t) => {
        const left = pos(t.start);
        const width = Math.max(pos(t.end) - left, 0.8);
        const dates = t.start === t.end ? shortDate(t.start) : `${shortDate(t.start)} to ${shortDate(t.end)}`;
        return (
          <li key={t.label} className="text-xs">
            <div className="flex justify-between gap-3">
              <span>{t.label}</span>
              <span className="tnum shrink-0 text-subtle">{dates}</span>
            </div>
            <div className="relative mt-0.5 h-2.5 rounded-sm bg-wash" aria-hidden="true">
              {todayPos >= 0 && todayPos <= 100 && <span className="absolute inset-y-0 w-px bg-foreground" style={{ left: `${todayPos}%` }} />}
              {t.milestone
                ? <span className="absolute top-1/2 size-2 -translate-x-1/2 -translate-y-1/2 rotate-45 bg-foreground" style={{ left: `${left}%` }} />
                : <span className="absolute inset-y-0.5 rounded-sm bg-primary" style={{ left: `${left}%`, width: `${width}%` }} />}
            </div>
          </li>
        );
      })}
    </ol>
  );
}

function Highlights({ data }: { data: OverviewData }) {
  const h = data.highlights;
  const ratios = Object.values(h.lm_vs_adam ?? {});
  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-5">
      {h.engine && <Stat label="Engine derivatives" value={sci(h.engine.worst_error)} detail={`Worst finite-difference error across ${h.engine.layers} layer types`} />}
      {ratios.length > 0 && (
        <Stat label="LM over Adam, same network" value={`${factor(Math.min(...ratios))} to ${factor(Math.max(...ratios))}`} detail="Median error ratio, Runge and Franke, 6 seeds" />
      )}
      {h.kan && <Stat label="KAN on exp(sin πx + y²)" value={sci(h.kan.rmse)} detail={`${h.kan.n_params} parameters; the best MLP reaches ${sci(h.kan.best_mlp)}`} />}
      {h.spectral && (
        <Stat
          label="Learning sin 16πx"
          value={h.spectral.fourier_learned_at != null ? `${integer(h.spectral.fourier_learned_at)} steps` : "Not learned"}
          detail={`With Fourier features. A tanh MLP ${h.spectral.tanh_learned_at == null ? `has not learned it after ${integer(h.spectral.steps)}` : `needs ${integer(h.spectral.tanh_learned_at)}`}`}
        />
      )}
      {h.conformal && <Stat label="Conformal coverage" value={percent(h.conformal.coverage)} detail={`Promised ${percent(h.conformal.nominal, 0)}, kept on unseen data`} />}
    </div>
  );
}

export function Overview() {
  const overview = useQuery({ queryKey: ["overview"], queryFn: api.overview });
  const experiments = useQuery({ queryKey: ["experiments"], queryFn: api.experiments });
  const e6 = useQuery({ queryKey: ["dataset", "e6", "runs"], queryFn: () => api.dataset("e6", "runs") });
  const targets = useTargets();

  if (overview.isError) return <ErrorState error={overview.error} onRetry={() => overview.refetch()} />;
  if (!overview.data) return <LoadingBlock className="h-[600px]" />;
  const data = overview.data;
  const quick = data.profiles.quick ?? 0;
  const days = data.deadline ? daysUntil(data.deadline) : null;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Neural network as a device for function approximation"
        description="A from-scratch NumPy engine learns unknown functions from samples. Every result here is scored against the exact function, so real approximation quality is not confused with fitting the noise."
        actions={(
          <>
            <Button asChild><Link to="/finder">Find a curve <ArrowRight /></Link></Button>
            <Button variant="outline" asChild><Link to="/lab">Open the Training Lab</Link></Button>
          </>
        )}
      />

      {quick > 0 && (
        <Notice
          tone="warning"
          title={`${quick} of ${Object.values(data.profiles).reduce((a, b) => a + b, 0)} experiments have quick-profile results only`}
          action={<Button variant="outline" size="sm" asChild><Link to="/runner">Open the runner</Link></Button>}
        >
          Few seeds and at most 2,000 points. The full profile must confirm these numbers before they go into the report.
        </Notice>
      )}

      <Highlights data={data} />

      <div className="grid grid-cols-1 gap-6 min-[1400px]:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        <Section title="Is a network the right tool? It depends on the target." description="Experiment E6: best test RMSE and the size that reached it, over the whole size sweep. The project does not claim gaps below ×1.5.">
          <VerdictTable verdicts={data.verdicts} />
        </Section>
        <Section
          title={days != null ? (days >= 0 ? `${days} days to Phase I submission` : `Phase I submission was ${-days} days ago`) : "Plan"}
          description="Remaining plan from PROJECT_PLAN.md. The vertical line marks today."
        >
          <Card className="p-3"><Timeline items={data.timeline} /></Card>
        </Section>
      </div>

      {e6.data && <RunsChart spec={EXPERIMENT_CHARTS.e6[0]} data={e6.data} />}

      <Section title="Target functions" description="Each one tests one property of neural approximation. Open one in the lab to train on it.">
        <ul className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4 2xl:grid-cols-6">
          {targets.data?.map((t) => (
            <li key={t.key}>
              <Link to={`/lab?target=${t.key}`} className="group flex h-full flex-col gap-1.5 rounded-lg border bg-card p-3 transition-colors hover:border-axis">
                <Preview grid={t.preview} />
                <div className="flex items-baseline justify-between gap-2">
                  <span className="text-sm font-medium">{t.key} · {t.name}</span>
                  <ArrowRight className="size-3.5 shrink-0 text-subtle opacity-0 transition-opacity group-hover:opacity-100" aria-hidden="true" />
                </div>
                <p className="text-xs text-muted-foreground">{t.formula}</p>
                <p className="text-xs text-subtle">{t.probes}</p>
              </Link>
            </li>
          ))}
        </ul>
      </Section>

      <Section title="Experiments" description={`${integer(data.run_count)} recorded runs${data.generated ? `, latest generated ${data.generated}` : ""}.`}>
        <div className="overflow-x-auto rounded-lg border bg-card">
          <table className="w-full text-[13px]">
            <caption className="sr-only">All experiments and the question each one answers</caption>
            <thead className="bg-wash text-left text-xs text-muted-foreground">
              <tr>
                <th scope="col" className="px-3 py-2 font-medium">Experiment</th>
                <th scope="col" className="px-3 py-2 font-medium">Question</th>
                <th scope="col" className="px-3 py-2 font-medium">Generated</th>
              </tr>
            </thead>
            <tbody>
              {experiments.data?.map((e) => (
                <tr key={e.id} className="border-t">
                  <td className="px-3 py-1.5 whitespace-nowrap">
                    <Link to={`/experiments/${e.id}`} className="font-medium underline-offset-2 hover:underline">{e.id.toUpperCase()} · {e.title}</Link>
                  </td>
                  <td className="min-w-64 px-3 py-1.5 text-muted-foreground">{e.question}</td>
                  <td className="tnum px-3 py-1.5 whitespace-nowrap text-muted-foreground">{e.generated ?? "Not run yet"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>
    </div>
  );
}

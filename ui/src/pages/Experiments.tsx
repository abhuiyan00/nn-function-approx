import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router";
import { ErrorState, LoadingBlock, PageHeader, ProfileBadge, Section } from "@/components/common";
import type { ExperimentSummary } from "@/lib/api";
import { api } from "@/lib/api";

function ExperimentCard({ experiment: e }: { experiment: ExperimentSummary }) {
  return (
    <li>
      <Link to={`/experiments/${e.id}`} className="flex h-full flex-col gap-2 rounded-lg border bg-card p-4 transition-colors hover:border-axis">
        <div className="flex items-start justify-between gap-2">
          <h3 className="text-sm font-semibold"><span className="tnum text-subtle">{e.id.toUpperCase()}</span> {e.title}</h3>
          <ProfileBadge profile={e.profile} />
        </div>
        <p className="text-[13px]">{e.question}</p>
        <p className="line-clamp-3 text-xs text-muted-foreground"><span className="font-medium">Hypothesis: </span>{e.hypothesis}</p>
        <div className="mt-auto flex justify-between gap-2 pt-1 text-xs text-subtle">
          <span>{e.owner ? `Owner: Member ${e.owner}` : ""}</span>
          <span className="tnum">{e.generated ?? "Not run yet"}</span>
        </div>
      </Link>
    </li>
  );
}

export function Experiments() {
  const experiments = useQuery({ queryKey: ["experiments"], queryFn: api.experiments });
  if (experiments.isError) return <ErrorState error={experiments.error} onRetry={() => experiments.refetch()} />;
  if (!experiments.data) return <LoadingBlock />;
  const group = (g: ExperimentSummary["group"]) => experiments.data.filter((e) => e.group === g);

  return (
    <div className="space-y-6">
      <PageHeader title="Experiments" description="What each study asked, what it expected, and what it found. Text comes from the project documents, numbers from the result files." />
      <Section title="Sensitivity study" description="The assignment: how quality depends on the network and on the training.">
        <ul className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">{group("sensitivity").map((e) => <ExperimentCard key={e.id} experiment={e} />)}</ul>
      </Section>
      <Section title="Unique studies" description="Beyond the assignment, built on recent research and implemented in the same engine.">
        <ul className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4">{group("unique").map((e) => <ExperimentCard key={e.id} experiment={e} />)}</ul>
      </Section>
    </div>
  );
}

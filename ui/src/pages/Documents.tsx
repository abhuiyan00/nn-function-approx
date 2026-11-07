import { useQuery } from "@tanstack/react-query";
import { NavLink, useParams } from "react-router";
import { EmptyState, ErrorState, LoadingBlock } from "@/components/common";
import { Markdown } from "@/components/Markdown";
import { Card } from "@/components/ui/card";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";

export function Documents() {
  const { slug } = useParams();
  const documents = useQuery({ queryKey: ["documents"], queryFn: api.documents });
  const active = slug ?? documents.data?.find((d) => d.slug === "findings")?.slug ?? documents.data?.[0]?.slug;
  const doc = useQuery({ queryKey: ["document", active], queryFn: () => api.document(active!), enabled: !!active });

  if (documents.isError) return <ErrorState error={documents.error} onRetry={() => documents.refetch()} />;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 items-start gap-4 lg:grid-cols-[230px_minmax(0,1fr)]">
        <nav aria-label="Documents" className="lg:sticky lg:top-0">
          <ul className="flex gap-1 overflow-x-auto pb-1 lg:flex-col lg:overflow-visible lg:pb-0">
            {documents.data?.map((d) => (
              <li key={d.slug} className="shrink-0">
                <NavLink
                  to={`/documents/${d.slug}`}
                  aria-current={d.slug === active ? "page" : undefined}
                  className={cn(
                    "block max-w-60 truncate rounded-md px-2.5 py-1.5 text-[13px] text-muted-foreground transition-colors hover:bg-wash hover:text-foreground lg:max-w-none",
                    d.slug === active && "bg-wash font-medium text-foreground",
                  )}
                  title={d.path}
                >
                  {d.slug === "readme" ? "Project README" : `${d.slug[0].toUpperCase()}${d.slug.slice(1).replace(/_/g, " ")}`}
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>
        <Card className="min-w-0 p-5 lg:p-8">
          {doc.isError ? (
            <EmptyState title="This document does not exist">It may have been renamed. Pick one from the list.</EmptyState>
          ) : doc.data ? (
            <article className="mx-auto max-w-[82ch]">
              <p className="mb-3 font-mono text-xs text-subtle">{doc.data.path}</p>
              <Markdown>{doc.data.markdown}</Markdown>
            </article>
          ) : (
            <LoadingBlock />
          )}
        </Card>
      </div>
    </div>
  );
}

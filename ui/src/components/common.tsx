import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, CheckCircle2, CircleDashed, CircleSlash, Info, Loader2, RotateCw, XCircle } from "lucide-react";
import type { ReactNode } from "react";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { ApiError, type JobStatus, type RunStatus, api } from "@/lib/api";
import { cn } from "@/lib/utils";

/** Single choice from a few visible options, as a radio group with arrow-key support. */
export function Segmented<T extends string>({ value, onChange, options, label, className }: {
  value: T | undefined;
  onChange: (value: T) => void;
  options: { value: T; label: ReactNode }[];
  label: string;
  className?: string;
}) {
  return (
    <div
      role="radiogroup"
      aria-label={label}
      className={cn("inline-flex max-w-full flex-wrap gap-0.5 rounded-md bg-wash p-0.5", className)}
      onKeyDown={(event) => {
        const delta = event.key === "ArrowRight" || event.key === "ArrowDown" ? 1 : event.key === "ArrowLeft" || event.key === "ArrowUp" ? -1 : 0;
        if (!delta) return;
        event.preventDefault();
        const at = options.findIndex((o) => o.value === value);
        const next = options[(at + delta + options.length) % options.length];
        onChange(next.value);
        const group = event.currentTarget;
        requestAnimationFrame(() => group.querySelector<HTMLElement>('[aria-checked="true"]')?.focus());
      }}
    >
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          role="radio"
          aria-checked={option.value === value}
          tabIndex={option.value === value ? 0 : -1}
          data-slot="tabs-trigger"
          onClick={() => onChange(option.value)}
          className={cn(
            "rounded-sm px-2.5 py-1 text-xs font-medium whitespace-nowrap text-muted-foreground transition-colors hover:text-foreground",
            option.value === value && "bg-surface text-foreground shadow-xs",
          )}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

export function PageHeader({ title, description, actions }: { title: string; description?: ReactNode; actions?: ReactNode }) {
  return (
    <header className="flex flex-wrap items-start justify-between gap-x-6 gap-y-3">
      <div className="min-w-0 max-w-3xl">
        <h1 className="text-xl font-semibold tracking-tight text-balance">{title}</h1>
        {description && <p className="mt-1 text-sm text-muted-foreground">{description}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </header>
  );
}

export function Section({ title, description, actions, children, className }: {
  title: string;
  description?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={cn("space-y-3", className)}>
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div className="min-w-0">
          <h2 className="text-sm font-semibold">{title}</h2>
          {description && <p className="text-xs text-muted-foreground">{description}</p>}
        </div>
        {actions}
      </div>
      {children}
    </section>
  );
}

export function Stat({ label, value, detail, className }: { label: string; value: ReactNode; detail?: ReactNode; className?: string }) {
  return (
    <div className={cn("min-w-0 rounded-lg border bg-card p-3", className)}>
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="mt-0.5 truncate text-lg font-semibold tracking-tight">{value}</div>
      {detail && <div className="mt-0.5 text-xs text-subtle">{detail}</div>}
    </div>
  );
}

const STATUS_STYLE: Record<RunStatus | JobStatus, { label: string; icon: ReactNode; className: string }> = {
  queued: { label: "Queued", icon: <CircleDashed className="size-3.5" />, className: "bg-wash text-muted-foreground" },
  running: { label: "Running", icon: <Loader2 className="size-3.5 animate-spin" />, className: "bg-info-wash text-info-ink" },
  done: { label: "Finished", icon: <CheckCircle2 className="size-3.5" />, className: "bg-good-wash text-good" },
  stopped: { label: "Stopped", icon: <CircleSlash className="size-3.5" />, className: "bg-wash text-muted-foreground" },
  cancelled: { label: "Cancelled", icon: <CircleSlash className="size-3.5" />, className: "bg-wash text-muted-foreground" },
  failed: { label: "Failed", icon: <XCircle className="size-3.5" />, className: "bg-critical-wash text-critical-ink" },
};

export function StatusPill({ status }: { status: RunStatus | JobStatus }) {
  const s = STATUS_STYLE[status];
  return (
    <span className={cn("inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium whitespace-nowrap", s.className)}>
      {s.icon}
      {s.label}
    </span>
  );
}

export function Notice({ tone = "info", title, children, action, className }: {
  tone?: "info" | "warning" | "critical";
  title?: string;
  children?: ReactNode;
  action?: ReactNode;
  className?: string;
}) {
  const Icon = tone === "info" ? Info : tone === "warning" ? AlertTriangle : XCircle;
  return (
    <div
      role={tone === "critical" ? "alert" : "status"}
      className={cn(
        "flex flex-wrap items-start gap-x-3 gap-y-2 rounded-lg px-3 py-2.5 text-sm",
        tone === "info" && "bg-info-wash text-info-ink",
        tone === "warning" && "bg-warning-wash text-warning-ink",
        tone === "critical" && "bg-critical-wash text-critical-ink",
        className,
      )}
    >
      <Icon className="mt-0.5 size-4 shrink-0" aria-hidden="true" />
      <div className="min-w-0 flex-1 basis-60">
        {title && <div className="font-medium">{title}</div>}
        {children && <div className={title ? "text-[13px]" : undefined}>{children}</div>}
      </div>
      {action}
    </div>
  );
}

export function ErrorState({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  const message = error instanceof ApiError || error instanceof Error ? error.message : "Something went wrong.";
  return (
    <Notice
      tone="critical"
      title="Could not load this"
      action={onRetry && (
        <Button variant="outline" size="sm" onClick={onRetry}>
          <RotateCw /> Try again
        </Button>
      )}
    >
      {message}
    </Notice>
  );
}

export function EmptyState({ icon, title, children, action }: { icon?: ReactNode; title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center gap-2 rounded-lg border border-dashed px-6 py-10 text-center">
      {icon && <div className="text-subtle">{icon}</div>}
      <p className="font-medium">{title}</p>
      {children && <p className="max-w-md text-sm text-muted-foreground">{children}</p>}
      {action && <div className="mt-2">{action}</div>}
    </div>
  );
}

export function LoadingBlock({ className }: { className?: string }) {
  return <Skeleton className={cn("h-64 w-full", className)} role="status" aria-label="Loading" />;
}

export function useTargets() {
  return useQuery({ queryKey: ["targets"], queryFn: api.targets, staleTime: Infinity });
}

export function ProfileBadge({ profile }: { profile: string | null | undefined }) {
  if (!profile) return <span className="text-xs text-subtle">No results</span>;
  const full = profile === "full";
  return (
    <span className={cn("inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium whitespace-nowrap", full ? "bg-good-wash text-good" : "bg-warning-wash text-warning-ink")}>
      {full ? <CheckCircle2 className="size-3.5" aria-hidden="true" /> : <AlertTriangle className="size-3.5" aria-hidden="true" />}
      {full ? "Full profile" : "Quick profile"}
    </span>
  );
}

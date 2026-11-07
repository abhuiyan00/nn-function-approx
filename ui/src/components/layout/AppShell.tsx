import { useQuery } from "@tanstack/react-query";
import { BookOpenText, Boxes, ChartScatter, FlaskConical, LayoutDashboard, ListChecks, Menu, Moon, Sun, Terminal } from "lucide-react";
import { type ReactNode, useEffect, useState } from "react";
import { NavLink, useLocation } from "react-router";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetDescription, SheetTitle } from "@/components/ui/sheet";
import { useTheme } from "@/hooks/useTheme";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";

const NAV = [
  { to: "/", label: "Overview", icon: LayoutDashboard, end: true },
  { to: "/lab", label: "Training Lab", icon: FlaskConical },
  { to: "/experiments", label: "Experiments", icon: ListChecks },
  { to: "/explorer", label: "Run Explorer", icon: ChartScatter },
  { to: "/models", label: "Saved Models", icon: Boxes },
  { to: "/runner", label: "Experiment Runner", icon: Terminal },
  { to: "/documents", label: "Documents", icon: BookOpenText },
];

function Logo() {
  return (
    <svg viewBox="0 0 32 32" className="size-7 shrink-0" aria-hidden="true">
      <rect width="32" height="32" rx="7" fill="var(--primary)" />
      <path d="M5 21c3 0 4-11 7-11s4 12 7 12 3-8 8-8" fill="none" stroke="var(--primary-foreground)" strokeWidth="2.6" strokeLinecap="round" />
    </svg>
  );
}

function ThemeToggle() {
  const { theme, toggle } = useTheme();
  const next = theme === "dark" ? "light" : "dark";
  return (
    <Button variant="ghost" size="icon-sm" onClick={toggle} aria-label={`Switch to ${next} theme`} title={`Switch to ${next} theme`}>
      {theme === "dark" ? <Sun /> : <Moon />}
    </Button>
  );
}

function SidebarContent({ onNavigate }: { onNavigate?: () => void }) {
  const health = useQuery({ queryKey: ["health"], queryFn: api.health, refetchInterval: 15_000, retry: false, staleTime: 0 });
  return (
    <div className="flex h-full flex-col gap-4 p-3">
      <div className="flex items-center gap-2.5 px-1.5 pt-1">
        <Logo />
        <div className="min-w-0">
          <div className="text-sm leading-tight font-semibold">fapprox</div>
          <div className="truncate text-xs text-subtle">Function approximation</div>
        </div>
      </div>
      <nav aria-label="Main" className="flex-1">
        <ul className="space-y-0.5">
          {NAV.map(({ to, label, icon: Icon, end }) => (
            <li key={to}>
              <NavLink
                to={to}
                end={end}
                onClick={onNavigate}
                className={({ isActive }) => cn(
                  "flex items-center gap-2.5 rounded-md px-2.5 py-1.5 text-sm text-muted-foreground transition-colors hover:bg-wash hover:text-foreground pointer-coarse:py-3",
                  isActive && "bg-wash font-medium text-foreground",
                )}
              >
                <Icon className="size-4 shrink-0" aria-hidden="true" />
                {label}
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>
      <div className="space-y-1 border-t px-1.5 pt-3 text-xs text-subtle">
        <div className="flex items-center gap-1.5" role="status">
          <span aria-hidden="true" className={cn("size-2 rounded-full", health.isError ? "bg-destructive" : "bg-[#0ca30c]")} />
          {health.isError ? "Server not responding" : health.data ? `Engine ${health.data.version} connected` : "Connecting"}
        </div>
        {health.data && <div className="tnum">Python {health.data.python} · NumPy {health.data.numpy}</div>}
        <a href="/api/docs" target="_blank" rel="noreferrer" className="inline-block underline underline-offset-2 hover:text-foreground">API reference</a>
      </div>
    </div>
  );
}

export function AppShell({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const { pathname } = useLocation();

  useEffect(() => {
    const title = NAV.find((n) => (n.end ? pathname === n.to : pathname.startsWith(n.to)))?.label;
    document.title = title ? `${title} · fapprox` : "fapprox";
    document.getElementById("main")?.scrollTo(0, 0);
  }, [pathname]);

  return (
    <div className="flex h-dvh overflow-hidden">
      <a href="#main" className="sr-only z-50 rounded-md bg-primary px-3 py-2 text-primary-foreground focus:not-sr-only focus:absolute focus:top-2 focus:left-2">
        Skip to content
      </a>
      <aside className="hidden w-56 shrink-0 border-r bg-surface lg:block">
        <SidebarContent />
      </aside>
      <Sheet open={open} onOpenChange={setOpen}>
        <SheetContent side="left" className="w-64 bg-surface p-0">
          <SheetTitle className="sr-only">Navigation</SheetTitle>
          <SheetDescription className="sr-only">Pages of the fapprox interface</SheetDescription>
          <SidebarContent onNavigate={() => setOpen(false)} />
        </SheetContent>
      </Sheet>
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex h-12 shrink-0 items-center justify-between border-b bg-surface px-3 lg:h-11 lg:px-5">
          <div className="flex items-center gap-2 lg:hidden">
            <Button variant="ghost" size="icon-sm" aria-label="Open navigation" onClick={() => setOpen(true)}>
              <Menu />
            </Button>
            <Logo />
            <span className="text-sm font-semibold">fapprox</span>
          </div>
          <div className="hidden text-xs text-subtle lg:block">Soft Computing · Project I · Topic 11 · WUST 2026/27</div>
          <ThemeToggle />
        </div>
        <main id="main" tabIndex={-1} className="min-w-0 flex-1 overflow-y-auto outline-none">
          <div className="mx-auto max-w-[1440px] p-4 lg:p-6">{children}</div>
        </main>
      </div>
    </div>
  );
}

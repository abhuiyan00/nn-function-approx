import { Component, type ErrorInfo, type ReactNode, Suspense, lazy } from "react";
import { Link, Route, Routes, useLocation } from "react-router";
import { EmptyState, ErrorState, LoadingBlock } from "./components/common";
import { AppShell } from "./components/layout/AppShell";
import { Button } from "./components/ui/button";

const named = <K extends string>(load: () => Promise<Record<K, () => ReactNode>>, key: K) =>
  lazy(async () => ({ default: (await load())[key] }));

const Overview = named(() => import("./pages/Overview"), "Overview");
const Finder = named(() => import("./pages/Finder"), "Finder");
const Lab = named(() => import("./pages/lab/LabPage"), "LabPage");
const Experiments = named(() => import("./pages/Experiments"), "Experiments");
const ExperimentDetail = named(() => import("./pages/ExperimentDetail"), "ExperimentDetail");
const Explorer = named(() => import("./pages/Explorer"), "Explorer");
const Models = named(() => import("./pages/Models"), "Models");
const Runner = named(() => import("./pages/Runner"), "Runner");
const Documents = named(() => import("./pages/Documents"), "Documents");

class ErrorBoundary extends Component<{ children: ReactNode }, { error: unknown }> {
  state = { error: null as unknown };

  static getDerivedStateFromError(error: unknown) {
    return { error };
  }

  componentDidCatch(error: unknown, info: ErrorInfo) {
    console.error(error, info.componentStack);
  }

  render() {
    if (this.state.error) return <ErrorState error={this.state.error} onRetry={() => this.setState({ error: null })} />;
    return this.props.children;
  }
}

function NotFound() {
  return (
    <EmptyState title="This page does not exist" action={<Button asChild><Link to="/">Go to the overview</Link></Button>}>
      The address may be from an older version of the interface.
    </EmptyState>
  );
}

export function App() {
  const { pathname } = useLocation();
  return (
    <AppShell>
      <ErrorBoundary key={pathname}>
        <Suspense fallback={<LoadingBlock className="h-96" />}>
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/finder" element={<Finder />} />
            <Route path="/lab" element={<Lab />} />
            <Route path="/experiments" element={<Experiments />} />
            <Route path="/experiments/:id" element={<ExperimentDetail />} />
            <Route path="/explorer" element={<Explorer />} />
            <Route path="/models" element={<Models />} />
            <Route path="/runner" element={<Runner />} />
            <Route path="/documents" element={<Documents />} />
            <Route path="/documents/:slug" element={<Documents />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </Suspense>
      </ErrorBoundary>
    </AppShell>
  );
}

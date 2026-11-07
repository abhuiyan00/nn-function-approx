import { useMutation, useQuery } from "@tanstack/react-query";
import { Boxes, TriangleAlert } from "lucide-react";
import { useId, useState } from "react";
import { Link, useSearchParams } from "react-router";
import { EmptyState, ErrorState, LoadingBlock, Notice, PageHeader, Stat } from "@/components/common";
import { FitView } from "@/components/FitView";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { type ModelDetail, api } from "@/lib/api";
import { bytes, clock, integer, sci } from "@/lib/format";

/** "0.5, 1" in 1-D; one "x y" per line in 2-D. Null when it does not fit the model. */
function parseInputs(text: string, dim: number): number[][] | null {
  const numbers = text.trim().split(/[\s,;]+/).filter(Boolean).map(Number);
  if (!numbers.length || numbers.some((n) => !Number.isFinite(n)) || numbers.length % dim) return null;
  const rows: number[][] = [];
  for (let i = 0; i < numbers.length; i += dim) rows.push(numbers.slice(i, i + dim));
  return rows;
}

function PredictForm({ model }: { model: ModelDetail }) {
  const id = useId();
  const dim = model.dim ?? 1;
  const [text, setText] = useState(dim === 1 ? "0, 0.25, 0.5" : "0 0\n0.5 0.5");
  const inputs = parseInputs(text, dim);
  const predict = useMutation({ mutationFn: (rows: number[][]) => api.predict(model.name, rows) });
  const result = predict.data;
  const queried = predict.variables;

  return (
    <Card className="gap-3 p-4">
      <h3 className="text-sm font-semibold">Ask the model</h3>
      <form className="space-y-2" onSubmit={(e) => { e.preventDefault(); if (inputs) predict.mutate(inputs); }}>
        <Label htmlFor={id} className="text-xs font-medium text-muted-foreground">
          {dim === 1 ? "Input values, separated by commas or spaces" : "One pair of input values per line"}
        </Label>
        <Textarea id={id} value={text} onChange={(e) => setText(e.target.value)} rows={3} className="tnum font-mono text-xs" aria-invalid={!inputs} aria-describedby={inputs ? undefined : `${id}-error`} />
        {!inputs && <p id={`${id}-error`} className="text-xs text-critical-ink">{dim === 1 ? "Enter numbers only." : "Enter pairs of numbers."}</p>}
        <Button type="submit" size="sm" disabled={!inputs || predict.isPending}>Predict</Button>
      </form>
      {predict.error && <Notice tone="critical">{(predict.error as Error).message}</Notice>}
      {result && queried && (
        <div className="overflow-x-auto rounded-md border">
          <table className="w-full text-xs">
            <caption className="sr-only">Predictions for the given inputs</caption>
            <thead className="bg-wash text-left text-muted-foreground">
              <tr>
                <th scope="col" className="px-3 py-2 font-medium">Input</th>
                <th scope="col" className="px-3 py-2 text-right font-medium">Network</th>
                <th scope="col" className="px-3 py-2 text-right font-medium">Exact f</th>
                <th scope="col" className="px-3 py-2 text-right font-medium">Absolute error</th>
              </tr>
            </thead>
            <tbody>
              {result.y_hat.map((y, i) => (
                <tr key={i} className="border-t">
                  <td className="tnum px-3 py-1.5 whitespace-nowrap">
                    {queried[i].join(", ")}
                    {result.outside_box[i] && (
                      <span className="ml-2 inline-flex items-center gap-1 text-warning-ink"><TriangleAlert className="size-3" aria-hidden="true" />Outside the training box</span>
                    )}
                  </td>
                  <td className="tnum px-3 py-1.5 text-right font-medium">{sci(y)}</td>
                  <td className="tnum px-3 py-1.5 text-right">{sci(result.exact?.[i])}</td>
                  <td className="tnum px-3 py-1.5 text-right">{sci(result.abs_error?.[i])}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}

function ModelView({ name }: { name: string }) {
  const detail = useQuery({ queryKey: ["model", name], queryFn: () => api.model(name) });
  if (detail.isError) return <ErrorState error={detail.error} onRetry={() => detail.refetch()} />;
  if (!detail.data) return <LoadingBlock />;
  const m = detail.data;
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-2 lg:grid-cols-4">
        <Stat label="RMSE vs exact f" value={sci(m.rmse)} detail="When it was saved" />
        <Stat label="Max error" value={sci(m.max_err)} />
        <Stat label="R²" value={m.r2 == null ? "—" : m.r2.toFixed(5)} />
        <Stat label="Parameters" value={integer(m.n_params)} />
      </div>
      <div className="grid grid-cols-1 items-start gap-4 xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        {m.grid ? (
          <Card className="gap-2 p-4">
            <h3 className="text-sm font-semibold">{m.target} · {m.formula}</h3>
            <FitView grid={m.grid} pred={m.pred} residual={m.residual} />
          </Card>
        ) : (
          <Notice tone="warning" title="No target recorded">This file does not say which function it was trained on, so it cannot be compared with one.</Notice>
        )}
        <PredictForm key={m.name} model={m} />
      </div>
    </div>
  );
}

export function Models() {
  const [params, setParams] = useSearchParams();
  const models = useQuery({ queryKey: ["models"], queryFn: api.models });
  if (models.isError) return <ErrorState error={models.error} onRetry={() => models.refetch()} />;
  if (!models.data) return <LoadingBlock />;
  const usable = models.data.filter((m) => !m.error);
  const selected = params.get("model") ?? usable[0]?.name ?? null;

  return (
    <div className="space-y-4">
      <PageHeader title="Saved Models" description="Networks saved to the runs folder, from this interface or the command line. Reload one to query it and to check it against the exact function." />
      {models.data.length === 0 ? (
        <EmptyState
          icon={<Boxes className="size-6" />}
          title="No saved models yet"
          action={<Button asChild><Link to="/lab">Open the Training Lab</Link></Button>}
        >
          Train a network and press the save icon in the runs table. Files saved with the command line appear here too.
        </EmptyState>
      ) : (
        <>
          <div className="overflow-x-auto rounded-lg border bg-card">
            <table className="w-full text-[13px]">
              <caption className="sr-only">Saved model files</caption>
              <thead className="bg-wash text-left text-xs text-muted-foreground">
                <tr>
                  <th scope="col" className="px-3 py-2 font-medium">Name</th>
                  <th scope="col" className="px-3 py-2 font-medium">Network</th>
                  <th scope="col" className="px-3 py-2 font-medium">Target</th>
                  <th scope="col" className="px-3 py-2 text-right font-medium">RMSE</th>
                  <th scope="col" className="px-3 py-2 text-right font-medium">Params</th>
                  <th scope="col" className="px-3 py-2 text-right font-medium">Size</th>
                  <th scope="col" className="px-3 py-2 font-medium">Saved</th>
                </tr>
              </thead>
              <tbody>
                {models.data.map((m) => (
                  <tr key={m.name} data-state={m.name === selected ? "selected" : undefined} className="border-t data-[state=selected]:bg-wash">
                    <td className="px-3 py-1.5">
                      {m.error ? (
                        <span title={m.error}>{m.name}</span>
                      ) : (
                        <button type="button" className="font-medium underline-offset-2 hover:underline" aria-current={m.name === selected} onClick={() => setParams({ model: m.name }, { replace: true })}>{m.name}</button>
                      )}
                    </td>
                    <td className="px-3 py-1.5 text-muted-foreground">{m.error ? "Unreadable file" : m.label ?? m.arch}</td>
                    <td className="px-3 py-1.5">{m.target ?? "—"}</td>
                    <td className="tnum px-3 py-1.5 text-right font-medium">{sci(m.rmse)}</td>
                    <td className="tnum px-3 py-1.5 text-right">{integer(m.n_params)}</td>
                    <td className="tnum px-3 py-1.5 text-right whitespace-nowrap">{bytes(m.size)}</td>
                    <td className="tnum px-3 py-1.5 whitespace-nowrap text-muted-foreground">{clock(m.modified)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {selected && <ModelView key={selected} name={selected} />}
        </>
      )}
    </div>
  );
}

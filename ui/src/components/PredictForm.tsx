import { useMutation } from "@tanstack/react-query";
import { TriangleAlert } from "lucide-react";
import { useId, useState } from "react";
import { Notice } from "@/components/common";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import type { Prediction } from "@/lib/api";
import { sci } from "@/lib/format";

/** "0.5, 1" in 1-D; one "x y" per line in 2-D. Null when it does not fit the model. */
function parseInputs(text: string, dim: number): number[][] | null {
  const numbers = text.trim().split(/[\s,;]+/).filter(Boolean).map(Number);
  if (!numbers.length || numbers.some((n) => !Number.isFinite(n)) || numbers.length % dim) return null;
  const rows: number[][] = [];
  for (let i = 0; i < numbers.length; i += dim) rows.push(numbers.slice(i, i + dim));
  return rows;
}

export function PredictForm({ title, dim, onPredict }: { title: string; dim: number; onPredict: (rows: number[][]) => Promise<Prediction> }) {
  const id = useId();
  const [text, setText] = useState(dim === 1 ? "0, 0.25, 0.5" : "0 0\n0.5 0.5");
  const inputs = parseInputs(text, dim);
  const predict = useMutation({ mutationFn: onPredict });
  const result = predict.data;
  const queried = predict.variables;

  return (
    <Card className="gap-3 p-4">
      <h3 className="text-sm font-semibold">{title}</h3>
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
                {result.exact && <th scope="col" className="px-3 py-2 text-right font-medium">Exact f</th>}
                {result.exact && <th scope="col" className="px-3 py-2 text-right font-medium">Error</th>}
              </tr>
            </thead>
            <tbody>
              {result.y_hat.map((y, i) => (
                <tr key={i} className="border-t">
                  <td className="tnum px-3 py-1.5 whitespace-nowrap">
                    {queried[i].join(", ")}{" "}
                    {result.outside_box[i] && (
                      <span className="flex items-center gap-1 text-warning-ink"><TriangleAlert className="size-3" aria-hidden="true" />Outside the training box</span>
                    )}
                    {result.in_gap[i] && <span className="block text-muted-foreground">In the missing middle</span>}
                  </td>
                  <td className="tnum px-3 py-1.5 text-right font-medium whitespace-nowrap">{sci(y)}</td>
                  {result.exact && <td className="tnum px-3 py-1.5 text-right whitespace-nowrap">{sci(result.exact[i])}</td>}
                  {result.exact && <td className="tnum px-3 py-1.5 text-right whitespace-nowrap">{sci(result.abs_error?.[i])}</td>}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}

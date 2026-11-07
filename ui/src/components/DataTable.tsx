import { ArrowDown, ArrowUp, Download } from "lucide-react";
import { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import type { DatasetColumn } from "@/lib/api";
import { sci } from "@/lib/format";
import { type Row, downloadText, toCsv } from "@/lib/runs";

const PAGE = 25;

function compare(a: Row[string], b: Row[string]) {
  if (a == null) return 1;
  if (b == null) return -1;
  return typeof a === "number" && typeof b === "number" ? a - b : String(a).localeCompare(String(b), "en", { numeric: true });
}

/** Sortable result table. Columns that never change are hidden until asked for. */
export function DataTable({ columns, rows, filename }: { columns: DatasetColumn[]; rows: Row[]; filename: string }) {
  const [sort, setSort] = useState<{ column: string; desc: boolean } | null>(null);
  const [page, setPage] = useState(0);
  const [showConstant, setShowConstant] = useState(false);

  const visible = useMemo(() => columns.filter((c) => showConstant || !c.constant), [columns, showConstant]);
  const hidden = columns.length - visible.length;
  const sorted = useMemo(() => {
    if (!sort) return rows;
    const out = [...rows].sort((a, b) => compare(a[sort.column], b[sort.column]));
    if (sort.desc) out.reverse();
    return out;
  }, [rows, sort]);
  const pages = Math.max(1, Math.ceil(sorted.length / PAGE));
  const at = Math.min(page, pages - 1);
  const slice = sorted.slice(at * PAGE, (at + 1) * PAGE);

  const toggle = (column: string) => setSort((s) => (s?.column === column ? (s.desc ? null : { column, desc: true }) : { column, desc: false }));

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="tnum text-xs text-muted-foreground">
          {rows.length.toLocaleString("en-US")} rows · {visible.length} columns
        </div>
        <div className="flex flex-wrap items-center gap-3">
          {(hidden > 0 || showConstant) && (
            <div className="flex items-center gap-2">
              <Switch id={`${filename}-constant`} checked={showConstant} onCheckedChange={setShowConstant} />
              <Label htmlFor={`${filename}-constant`} className="text-xs font-normal text-muted-foreground">
                Show columns that never change{hidden > 0 ? ` (${hidden})` : ""}
              </Label>
            </div>
          )}
          <Button variant="outline" size="sm" onClick={() => downloadText(`${filename}.csv`, toCsv(visible.map((c) => c.name), sorted))}>
            <Download /> CSV
          </Button>
        </div>
      </div>
      <div className="overflow-x-auto rounded-md border bg-card">
        <table className="w-full text-xs">
          <caption className="sr-only">{filename}: result rows</caption>
          <thead className="bg-wash text-left text-muted-foreground">
            <tr>
              {visible.map((c) => {
                const state = sort?.column === c.name ? (sort.desc ? "descending" : "ascending") : "none";
                return (
                  <th key={c.name} scope="col" aria-sort={state} className="p-0 font-medium whitespace-nowrap">
                    <button type="button" onClick={() => toggle(c.name)} className={`flex w-full items-center gap-1 px-3 py-2 hover:text-foreground ${c.type === "number" ? "justify-end" : ""}`}>
                      {c.name}
                      {state === "ascending" && <ArrowUp className="size-3" aria-hidden="true" />}
                      {state === "descending" && <ArrowDown className="size-3" aria-hidden="true" />}
                    </button>
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {slice.map((row, i) => (
              <tr key={i} className="border-t hover:bg-wash/60">
                {visible.map((c) => (
                  <td key={c.name} className={`px-3 py-1.5 whitespace-nowrap ${c.type === "number" ? "tnum text-right" : ""}`}>
                    {row[c.name] == null ? <span className="text-subtle">—</span> : typeof row[c.name] === "number" ? sci(row[c.name] as number) : String(row[c.name])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {pages > 1 && (
        <nav aria-label="Table pages" className="flex items-center justify-end gap-2 text-xs text-muted-foreground">
          <span className="tnum">Page {at + 1} of {pages}</span>
          <Button variant="outline" size="sm" disabled={at === 0} onClick={() => setPage(at - 1)}>Previous</Button>
          <Button variant="outline" size="sm" disabled={at >= pages - 1} onClick={() => setPage(at + 1)}>Next</Button>
        </nav>
      )}
    </div>
  );
}

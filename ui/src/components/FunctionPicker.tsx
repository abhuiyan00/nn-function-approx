import { Eraser, Upload } from "lucide-react";
import { type PointerEvent, useEffect, useId, useRef, useState } from "react";
import { Segmented } from "@/components/common";
import { Field, NumberField, SelectField } from "@/components/fields";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import type { RunSpec, Target } from "@/lib/api";
import { AXES, PAD_COLUMNS, type Range, columnsToPoints, defaultGap, eraseSpan, examplePoints, formulaDim, padCol, padX, paintColumns, parsePoints, pointsToColumns, pointsToText, resizeRanges } from "@/lib/functions";

type Mode = "builtin" | "formula" | "points" | "draw";
type Patch = Partial<Pick<RunSpec, "target" | "formula" | "domain" | "points" | "gap">>;

const FORMULAS: { label: string; formula: string; domain: Range[] }[] = [
  { label: "Damped wave", formula: "sin(3*x) * exp(-x^2/4)", domain: [[-4, 4]] },
  { label: "Cubic", formula: "x^3 - x", domain: [[-1.5, 1.5]] },
  { label: "Undefined in the middle", formula: "sqrt(x^2 - 0.25)", domain: [[-1, 1]] },
  { label: "Surface", formula: "sin(3*x) * cos(2*y)", domain: [[-1, 1], [-1, 1]] },
  { label: "3 inputs", formula: "sin(pi*x) * y + z^2", domain: [[-1, 1], [-1, 1], [-1, 1]] },
];

function FormulaInput({ spec, error, onChange }: { spec: RunSpec; error: string | null; onChange: (patch: Patch) => void }) {
  const id = useId();
  const dim = formulaDim(spec.formula ?? "");
  const domain = resizeRanges(spec.domain, dim);
  const setFormula = (formula: string) => {
    const next = formulaDim(formula);
    onChange({ formula, domain: resizeRanges(spec.domain, next), gap: spec.gap?.length === next ? spec.gap : null });
  };
  const setRange = (axis: number, side: number, value: number) =>
    onChange({ domain: domain.map((r, i) => (i === axis ? (side ? [r[0], value] : [value, r[1]]) as Range : r)) });
  return (
    <div className="space-y-2.5">
      <Field label={`f(${AXES.slice(0, dim).join(", ")}) =`} htmlFor={id} error={error ?? undefined} help="Numbers, x, y, z, pi, e, + - * / ^ and sin cos tan exp log sqrt abs sign step min max and more.">
        <Input id={id} value={spec.formula ?? ""} className="h-8 font-mono text-xs" spellCheck={false} autoComplete="off" aria-invalid={!!error} aria-describedby={error ? `${id}-error` : undefined} onChange={(e) => setFormula(e.target.value)} />
      </Field>
      <div className="flex flex-wrap gap-1">
        {FORMULAS.map((f) => (
          <Button key={f.label} type="button" variant="outline" size="xs" onClick={() => onChange({ formula: f.formula, domain: f.domain, gap: null })}>{f.label}</Button>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-2">
        {domain.map((range, axis) => [
          <NumberField key={`${axis}a`} label={`${AXES[axis]} from`} value={range[0]} onChange={(v) => setRange(axis, 0, v)} />,
          <NumberField key={`${axis}b`} label={`${AXES[axis]} to`} value={range[1]} onChange={(v) => setRange(axis, 1, v)} />,
        ])}
      </div>
    </div>
  );
}

function PointsInput({ spec, error, onChange }: { spec: RunSpec; error: string | null; onChange: (patch: Patch) => void }) {
  const id = useId();
  const [text, setText] = useState(() => pointsToText(spec.points ?? []));
  const parsed = parsePoints(text);
  const change = (value: string) => {
    setText(value);
    onChange({ points: parsePoints(value).points });
  };
  const message = parsed.error ?? error ?? undefined;
  return (
    <div className="space-y-2">
      <Field label="One x, y pair per line" htmlFor={id} error={message} help={`${parsed.points.length} points. A CSV file with a header works too.`}>
        <Textarea id={id} value={text} rows={7} spellCheck={false} className="tnum font-mono text-xs" aria-invalid={!!message} aria-describedby={message ? `${id}-error` : undefined} onChange={(e) => change(e.target.value)} />
      </Field>
      <Button type="button" variant="outline" size="xs" asChild>
        <label className="cursor-pointer">
          <Upload /> Open a file
          <input type="file" accept=".csv,.txt" className="sr-only" onChange={async (e) => { const file = e.target.files?.[0]; if (file) change(await file.text()); }} />
        </label>
      </Button>
    </div>
  );
}

/** Column heights over x in [-1, 1]. Leaving a part blank is itself a missing middle. */
function DrawPad({ spec, error, onChange }: { spec: RunSpec; error: string | null; onChange: (patch: Patch) => void }) {
  const [columns, setColumns] = useState(() => pointsToColumns(spec.points));
  const latest = useRef(columns); // the pointer can lift before the last stroke has rendered
  const pen = useRef<{ col: number; y: number } | null>(null);
  const last = PAD_COLUMNS - 1;
  const paint = (from: { col: number; y: number }, to: { col: number; y: number }) => {
    latest.current = paintColumns(latest.current, from, to);
    setColumns(latest.current);
  };

  const mark = (event: PointerEvent<SVGSVGElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    const col = Math.round(((event.clientX - box.left) / box.width) * last);
    return { col: Math.min(last, Math.max(0, col)), y: Math.min(1, Math.max(-1, 1 - (2 * (event.clientY - box.top)) / box.height)) };
  };
  const path = columns.reduce((d, y, col) => {
    if (y == null) return d;
    return `${d}${col && columns[col - 1] != null ? "L" : "M"}${col},${((1 - y) * 50).toFixed(1)}`;
  }, "");
  const count = columns.filter((y) => y != null).length;

  return (
    <div className="space-y-2">
      <svg
        viewBox={`0 0 ${last} 100`}
        preserveAspectRatio="none"
        role="img"
        aria-label="Drawing area: x and y both run from -1 to 1. Without a pointer, use Points instead."
        data-testid="draw-pad"
        className="h-44 w-full cursor-crosshair touch-none rounded-md border bg-surface"
        onPointerDown={(event) => {
          event.currentTarget.setPointerCapture(event.pointerId);
          pen.current = mark(event);
          paint(pen.current, pen.current);
        }}
        onPointerMove={(event) => {
          if (!pen.current) return;
          const to = mark(event);
          paint(pen.current, to);
          pen.current = to;
        }}
        onPointerUp={() => {
          pen.current = null;
          onChange({ points: columnsToPoints(latest.current) });
        }}
      >
        <line x1={0} x2={last} y1={50} y2={50} stroke="var(--border)" vectorEffect="non-scaling-stroke" />
        <line x1={last / 2} x2={last / 2} y1={0} y2={100} stroke="var(--border)" vectorEffect="non-scaling-stroke" />
        <path d={path} fill="none" stroke="var(--series-1)" strokeWidth={2.5} strokeLinecap="round" strokeLinejoin="round" vectorEffect="non-scaling-stroke" />
      </svg>
      <div className="flex items-center justify-between gap-2">
        <p className={`text-xs ${error && count ? "text-critical-ink" : "text-subtle"}`}>{count ? error ?? `${count} points. Draw again to change them; a blank part stays missing.` : "Draw a curve with the mouse or a finger."}</p>
        <Button type="button" variant="outline" size="xs" onClick={() => { latest.current = pointsToColumns(null); setColumns(latest.current); onChange({ points: [] }); }}><Eraser /> Clear</Button>
      </div>
    </div>
  );
}

/** How to say what the function is. */
export function FunctionPicker({ spec, targets, error, onChange }: {
  spec: RunSpec;
  targets: Target[];
  error: string | null;
  onChange: (patch: Patch) => void;
}) {
  const [mode, setMode] = useState<Mode>(spec.formula != null ? "formula" : spec.points != null ? "points" : "builtin");
  const target = targets.find((t) => t.key === spec.target);

  useEffect(() => { // the settings can change from outside: a preset, or reusing an earlier run
    setMode((m) => {
      if (spec.formula != null) return "formula";
      if (spec.points != null) return m === "draw" ? "draw" : "points";
      return "builtin";
    });
  }, [spec.formula, spec.points]);

  const change = (next: Mode) => {
    setMode(next);
    if (next === "builtin") onChange({ target: target?.key ?? targets[0].key, formula: null, domain: null, points: null, gap: null });
    if (next === "formula") onChange({ formula: FORMULAS[0].formula, domain: FORMULAS[0].domain, points: null, gap: null });
    if (next === "points") onChange({ formula: null, domain: null, points: examplePoints(), gap: null });
    if (next === "draw") onChange({ formula: null, domain: null, points: [], gap: null });
  };

  return (
    <div className="space-y-2.5">
      <Segmented
        label="How to give the function"
        value={mode}
        onChange={change}
        options={[
          { value: "builtin", label: "Examples" },
          { value: "formula", label: "Formula" },
          { value: "points", label: "Points" },
          { value: "draw", label: "Draw" },
        ]}
      />
      {mode === "builtin" && (
        <SelectField
          label="Function to approximate"
          value={target?.key ?? targets[0].key}
          onChange={(key) => onChange({ target: key, gap: null })}
          options={targets.map((t) => ({ value: t.key, label: `${t.key} · ${t.preview.gap || t.formula.length > 26 ? t.name : t.formula}` }))}
          help={target && `Tests: ${target.probes}.`}
        />
      )}
      {mode === "formula" && <FormulaInput spec={spec} error={error} onChange={onChange} />}
      {mode === "points" && <PointsInput key={mode} spec={spec} error={error} onChange={onChange} />}
      {mode === "draw" && <DrawPad key={mode} spec={spec} error={error} onChange={onChange} />}
      {mode === "builtin" && error && <p className="text-xs text-critical-ink">{error}</p>}
    </div>
  );
}

function GapRange({ axis, range, value, onChange }: { axis: string; range: Range; value: Range; onChange: (value: Range) => void }) {
  const step = (range[1] - range[0]) / 200;
  const tidy = (v: number) => Number(v.toPrecision(4));
  return (
    <div className="space-y-0.5">
      <div className="grid grid-cols-[auto_minmax(0,1fr)_3.5rem] items-center gap-x-2 text-xs text-muted-foreground">
        <span aria-hidden="true">{axis} from</span>
        <input type="range" aria-label={`Missing middle starts at this ${axis}`} className="w-full accent-[var(--primary)]" min={range[0]} max={range[1]} step={step} value={value[0]} onChange={(e) => onChange([tidy(Math.min(Number(e.target.value), value[1] - step)), value[1]])} />
        <span className="tnum text-right">{value[0]}</span>
        <span aria-hidden="true">{axis} to</span>
        <input type="range" aria-label={`Missing middle ends at this ${axis}`} className="w-full accent-[var(--primary)]" min={range[0]} max={range[1]} step={step} value={value[1]} onChange={(e) => onChange([value[0], tidy(Math.max(Number(e.target.value), value[0] + step))])} />
        <span className="tnum text-right">{value[1]}</span>
      </div>
    </div>
  );
}

/** ``box`` and ``gap`` as the server resolved them, so a built-in target shows its own hole. */
export function MissingMiddle({ box, gap, onChange }: { box: Range[]; gap: Range[] | null | undefined; onChange: (gap: Range[]) => void }) {
  const id = useId();
  const on = !!gap?.length && gap.length === box.length;
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-3">
        <Label htmlFor={id} className="text-xs font-medium text-muted-foreground">Hide a part of the range</Label>
        <Switch id={id} checked={on} onCheckedChange={(checked) => onChange(checked ? defaultGap(box) : [])} />
      </div>
      {on ? gap.map((value, i) => (
        <GapRange key={i} axis={AXES[i]} range={box[i]} value={value} onChange={(next) => onChange(gap.map((g, j) => (j === i ? next : g)))} />
      )) : <p className="text-xs text-subtle">Turn on to train without any data there. The network has to bridge the hole, and the score still covers it.</p>}
    </div>
  );
}

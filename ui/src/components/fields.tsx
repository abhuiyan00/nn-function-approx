import { type ReactNode, useEffect, useId, useState } from "react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { parseIntList } from "@/lib/runs";

export function Field({ label, help, error, children, htmlFor }: { label: string; help?: ReactNode; error?: string; children: ReactNode; htmlFor: string }) {
  return (
    <div className="min-w-0 space-y-1">
      <Label htmlFor={htmlFor} className="text-xs font-medium text-muted-foreground">{label}</Label>
      {children}
      {error ? <p id={`${htmlFor}-error`} className="text-xs text-critical-ink">{error}</p> : help && <p className="text-xs text-subtle">{help}</p>}
    </div>
  );
}

/** Keeps the text as typed, so "1e-" or an empty box can exist on the way to a valid number. */
export function NumberField({ label, value, onChange, min, max, help, integerOnly }: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  help?: ReactNode;
  integerOnly?: boolean;
}) {
  const id = useId();
  const [text, setText] = useState(String(value));
  const parsed = Number(text);
  const ok = text.trim() !== "" && Number.isFinite(parsed) && (!integerOnly || Number.isInteger(parsed)) && (min == null || parsed >= min) && (max == null || parsed <= max);

  useEffect(() => {
    if (Number(text) !== value) setText(String(value));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);

  const range = min != null && max != null ? `${integerOnly ? "A whole number" : "A number"} from ${min} to ${max}.` : "Enter a number.";
  return (
    <Field label={label} help={help} error={ok ? undefined : range} htmlFor={id}>
      <Input
        id={id}
        type="text"
        inputMode={integerOnly ? "numeric" : "decimal"}
        value={text}
        aria-invalid={!ok}
        aria-describedby={ok ? undefined : `${id}-error`}
        className="tnum h-8"
        onChange={(event) => {
          setText(event.target.value);
          const v = Number(event.target.value);
          if (event.target.value.trim() !== "" && Number.isFinite(v) && (!integerOnly || Number.isInteger(v)) && (min == null || v >= min) && (max == null || v <= max)) onChange(v);
        }}
      />
    </Field>
  );
}

export function ListField({ label, value, onChange, help, error, allowEmpty }: {
  label: string;
  value: number[];
  onChange: (value: number[]) => void;
  help?: ReactNode;
  error?: string;
  allowEmpty?: boolean;
}) {
  const id = useId();
  const [text, setText] = useState(value.join(","));
  const parsed = parseIntList(text);
  const ok = parsed != null || (allowEmpty && text.trim() === "");

  useEffect(() => {
    if ((parsed ?? []).join() !== value.join()) setText(value.join(","));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);

  return (
    <Field label={label} help={help} error={error ?? (ok ? undefined : "Use positive whole numbers, such as 16,16 or 3x64.")} htmlFor={id}>
      <Input
        id={id}
        value={text}
        aria-invalid={!ok || !!error}
        aria-describedby={!ok || error ? `${id}-error` : undefined}
        className="tnum h-8"
        onChange={(event) => {
          setText(event.target.value);
          const v = parseIntList(event.target.value);
          if (v) onChange(v);
          else if (allowEmpty && event.target.value.trim() === "") onChange([]);
        }}
      />
    </Field>
  );
}

export function SelectField<T extends string>({ label, value, onChange, options, help }: {
  label: string;
  value: T;
  onChange: (value: T) => void;
  options: { value: T; label: string }[];
  help?: ReactNode;
}) {
  const id = useId();
  return (
    <Field label={label} help={help} htmlFor={id}>
      <Select value={value} onValueChange={(v) => onChange(v as T)}>
        <SelectTrigger id={id} size="sm" className="w-full"><SelectValue /></SelectTrigger>
        <SelectContent>
          {options.map((o) => <SelectItem key={o.value} value={o.value}>{o.label}</SelectItem>)}
        </SelectContent>
      </Select>
    </Field>
  );
}

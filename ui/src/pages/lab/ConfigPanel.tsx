import { Copy, Play, Square } from "lucide-react";
import type { ReactNode } from "react";
import { toast } from "sonner";
import { Notice, Segmented } from "@/components/common";
import { ListField, NumberField, SelectField } from "@/components/fields";
import { FunctionPicker, MissingMiddle } from "@/components/FunctionPicker";
import { Button } from "@/components/ui/button";
import type { Activation, FunctionPreview, Optimizer, RunSpec, Target } from "@/lib/api";
import { integer } from "@/lib/format";
import { boxOf, functionDim } from "@/lib/functions";
import { type SpecErrors, cliCommand, defaultSpec, paramCount } from "@/lib/runs";

type ModelKind = "mlp" | "fourier" | "siren" | "kan";

const kindOf = (s: RunSpec): ModelKind => (s.arch === "kan" ? "kan" : s.act === "sine" ? "siren" : s.fourier_m ? "fourier" : "mlp");

function Group({ title, children }: { title: string; children: ReactNode }) {
  return (
    <fieldset className="space-y-2.5 border-t pt-3 first:border-t-0 first:pt-0">
      <legend className="float-left mb-2 w-full text-xs font-semibold">{title}</legend>
      <div className="clear-both space-y-2.5">{children}</div>
    </fieldset>
  );
}

const ACTIVATIONS: { value: Activation; label: string }[] = [
  { value: "tanh", label: "tanh" },
  { value: "sigmoid", label: "sigmoid" },
  { value: "silu", label: "SiLU" },
  { value: "relu", label: "ReLU" },
];
const OPTIMIZERS: { value: Optimizer; label: string }[] = [
  { value: "adam", label: "Adam" },
  { value: "sgd", label: "SGD" },
  { value: "momentum", label: "Momentum" },
  { value: "nesterov", label: "Nesterov" },
  { value: "lm", label: "Levenberg-Marquardt" },
];

export function ConfigPanel({ spec, onChange, targets, preview, functionError, checking, errors, serverError, running, onStart, onStop }: {
  spec: RunSpec;
  onChange: (spec: RunSpec) => void;
  targets: Target[];
  preview: FunctionPreview | undefined;
  functionError: string | null;
  checking: boolean;
  errors: SpecErrors;
  serverError: string | null;
  running: boolean;
  onStart: () => void;
  onStop: () => void;
}) {
  const dim = functionDim(spec, targets);
  const set = (patch: Partial<RunSpec>) => onChange({ ...spec, ...patch });
  const kind = kindOf(spec);
  const lm = spec.optimizer === "lm";
  const invalid = Object.keys(errors).length > 0 || !!functionError || checking;

  const setKind = (next: ModelKind) => {
    const hidden = spec.hidden;
    const base = { grid_updates: [] as number[] };
    if (next === "kan") return set({ arch: "kan", widths: [dim, 5, 1], grid: 10, grid_updates: [25, 75], fourier_m: 0, act: "tanh", bias_init: null });
    if (next === "siren") return set({ ...base, arch: "mlp", widths: null, act: "sine", fourier_m: 0, bias_init: null, hidden });
    if (next === "fourier") return set({ ...base, arch: "mlp", widths: null, act: spec.act === "sine" ? "tanh" : spec.act, fourier_m: 32, fourier_scale: 4, bias_init: null, hidden });
    return set({ ...base, arch: "mlp", widths: null, act: spec.act === "sine" ? "tanh" : spec.act, fourier_m: 0, hidden });
  };

  const setFunction = (patch: Partial<RunSpec>) => {
    const next = { ...spec, ...patch };
    const inputs = functionDim(next, targets);
    set({ ...patch, widths: spec.arch === "kan" && spec.widths ? [inputs, ...spec.widths.slice(1)] : spec.widths });
  };

  const setOptimizer = (optimizer: Optimizer) =>
    set(optimizer === "lm"
      ? { optimizer, lr: defaultSpec.lr, batch: defaultSpec.batch, l2: 0 } // LM is full batch and has no learning rate
      : { optimizer });

  const copyCommand = async () => {
    try {
      await navigator.clipboard.writeText(cliCommand(spec));
      toast.success("Command copied");
    } catch {
      toast.error("Could not reach the clipboard");
    }
  };

  return (
    <form
      aria-label="Training configuration"
      className="flex flex-col rounded-lg border bg-card"
      onSubmit={(event) => {
        event.preventDefault();
        if (!invalid) onStart();
      }}
    >
      <div className="space-y-3 p-3">
        <Group title="Target function">
          <FunctionPicker spec={spec} targets={targets} error={functionError} onChange={setFunction} />
          {preview && <MissingMiddle box={boxOf(preview.grid)} gap={preview.grid.gap} onChange={(gap) => set({ gap })} />}
        </Group>

        <Group title="Network">
          <Segmented
            label="Network type"
            value={kind}
            onChange={setKind}
            options={[
              { value: "mlp", label: "MLP" },
              { value: "fourier", label: "Fourier" },
              { value: "siren", label: "SIREN" },
              { value: "kan", label: "KAN" },
            ]}
          />
          {kind !== "kan" ? (
            <ListField label="Hidden layers" value={spec.hidden} onChange={(hidden) => set({ hidden })} error={errors.hidden} help="Units per layer: 16,16 or 3x64." />
          ) : (
            <>
              <ListField label="Layer widths" value={spec.widths ?? [dim, 5, 1]} onChange={(widths) => set({ widths })} error={errors.widths} help={`From the ${dim} input${dim > 1 ? "s" : ""} to 1 output.`} />
              <div className="grid grid-cols-2 gap-2">
                <NumberField label="Grid size G" value={spec.grid} min={2} max={64} integerOnly onChange={(grid) => set({ grid })} />
                <ListField label="Grid updates at" value={spec.grid_updates} allowEmpty onChange={(grid_updates) => set({ grid_updates })} />
              </div>
            </>
          )}
          {(kind === "mlp" || kind === "fourier") && (
            <SelectField label="Activation" value={spec.act} onChange={(act) => set({ act, bias_init: act === "relu" ? spec.bias_init : null })} options={ACTIVATIONS} />
          )}
          {spec.act === "relu" && kind === "mlp" && (
            <SelectField
              label="Hinge placement"
              value={spec.bias_init ?? "zero"}
              onChange={(v) => set({ bias_init: v === "zero" ? null : v })}
              options={[{ value: "zero", label: "All at x = 0" }, { value: "kink", label: "Random across the domain" }]}
            />
          )}
          {kind === "fourier" && (
            <div className="grid grid-cols-2 gap-2">
              <NumberField label="Features m" value={spec.fourier_m} min={1} max={512} integerOnly onChange={(fourier_m) => set({ fourier_m })} />
              <NumberField label="Frequency scale" value={spec.fourier_scale} min={0.01} max={128} onChange={(fourier_scale) => set({ fourier_scale })} />
            </div>
          )}
          {kind === "siren" && <NumberField label="Frequency ω₀" value={spec.w0} min={0.1} max={300} onChange={(w0) => set({ w0 })} help="A sensitive setting: try 10 and 30." />}
        </Group>

        <Group title="Training">
          <SelectField label="Optimiser" value={spec.optimizer} onChange={setOptimizer} options={OPTIMIZERS} />
          {!lm && (
            <div className="grid grid-cols-2 gap-2">
              <NumberField label="Learning rate" value={spec.lr} min={1e-7} max={10} onChange={(lr) => set({ lr })} />
              <NumberField label="Batch (0 = full)" value={spec.batch} min={0} max={20000} integerOnly onChange={(batch) => set({ batch })} />
            </div>
          )}
          <div className="grid grid-cols-2 gap-2">
            <NumberField label={lm ? "Max iterations" : "Epochs"} value={spec.epochs} min={1} max={50000} integerOnly onChange={(epochs) => set({ epochs })} />
            <NumberField label="Patience (0 = off)" value={spec.patience} min={0} max={50000} integerOnly onChange={(patience) => set({ patience })} />
          </div>
          {!lm && <NumberField label="L2 weight decay" value={spec.l2} min={0} max={10} onChange={(l2) => set({ l2 })} />}
        </Group>

        <Group title="Data">
          <div className="grid grid-cols-2 gap-2">
            {spec.points == null && (
              <>
                <NumberField label="Training points" value={spec.n_train} min={10} max={20000} integerOnly onChange={(n_train) => set({ n_train })} />
                <NumberField label="Validation points" value={spec.n_val} min={5} max={5000} integerOnly onChange={(n_val) => set({ n_val })} />
                <NumberField label="Noise σ" value={spec.sigma} min={0} max={5} onChange={(sigma) => set({ sigma })} />
              </>
            )}
            <NumberField label="Model seed" value={spec.seed} min={0} max={2147483647} integerOnly onChange={(seed) => set({ seed })} />
          </div>
          {spec.points != null && <p className="text-xs text-subtle">A fifth of your points is held out for validation and scoring.</p>}
        </Group>
      </div>

      <div className="sticky bottom-0 space-y-2 rounded-b-lg border-t bg-card p-3">
        {serverError && <Notice tone="critical">{serverError}</Notice>}
        <div className="flex items-center justify-between text-xs">
          <span className="text-muted-foreground">Parameters</span>
          <span className="tnum font-medium" data-testid="param-count">{integer(paramCount(spec, dim))}</span>
        </div>
        <div className="flex gap-2">
          {running ? (
            <Button type="button" variant="secondary" className="flex-1" onClick={onStop}><Square /> Stop</Button>
          ) : (
            <Button type="submit" className="flex-1" disabled={invalid}><Play /> Train</Button>
          )}
          <Button type="button" variant="outline" size="icon" onClick={copyCommand} aria-label="Copy as a command line" title={cliCommand(spec)}><Copy /></Button>
        </div>
      </div>
    </form>
  );
}

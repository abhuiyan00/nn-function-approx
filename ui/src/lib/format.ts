const DASH = "—";

/** Compact number in the project's style: plain in the readable range, otherwise scientific. */
export function sci(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return DASH;
  if (value === 0) return "0";
  const abs = Math.abs(value);
  if (abs >= 1e-2 && abs < 1e3) return String(Number(value.toPrecision(3)));
  return value.toExponential(2);
}

export function factor(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return DASH;
  if (value >= 1e4) return `×${value.toExponential(1)}`;
  return `×${value >= 10 ? Math.round(value) : value.toFixed(1)}`;
}

export function duration(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return DASH;
  if (seconds < 10) return `${seconds.toFixed(2)} s`;
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min ${Math.round(seconds % 60)} s`;
  return `${Math.floor(seconds / 3600)} h ${Math.floor((seconds % 3600) / 60)} min`;
}

export function integer(value: number | null | undefined): string {
  return value == null || !Number.isFinite(value) ? DASH : Math.round(value).toLocaleString("en-US");
}

export function percent(value: number | null | undefined, digits = 1): string {
  return value == null || !Number.isFinite(value) ? DASH : `${(value * 100).toFixed(digits)} %`;
}

/** Whole calendar days from ``now`` to an ISO date; negative once it has passed. */
export function daysUntil(isoDate: string, now: Date = new Date()): number {
  const [y, m, d] = isoDate.split("-").map(Number);
  const target = Date.UTC(y, m - 1, d);
  const today = Date.UTC(now.getFullYear(), now.getMonth(), now.getDate());
  return Math.round((target - today) / 86_400_000);
}

export function shortDate(isoDate: string): string {
  const [y, m, d] = isoDate.split("-").map(Number);
  return new Date(y, m - 1, d).toLocaleDateString("en-GB", { day: "numeric", month: "short" });
}

export function bytes(n: number): string {
  return n < 1024 ? `${n} B` : n < 1048576 ? `${(n / 1024).toFixed(1)} KB` : `${(n / 1048576).toFixed(1)} MB`;
}

export function clock(epochSeconds: number): string {
  return new Date(epochSeconds * 1000).toLocaleString("en-GB", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
}

import { useEffect, useRef } from "react";
import { useTheme } from "@/hooks/useTheme";
import type { NetView } from "@/lib/api";
import { sci } from "@/lib/format";
import { LAYER_GAP, layerLayout, pingPong, project } from "@/lib/network3d";

const PASS_MS = 1600; // one forward pass on screen
const YAW = 0.55;
const PITCH = 0.22;
const DISTANCE = 9;
const EASE = 0.14; // share of the remaining way the camera and the weights cover per frame
const clamp = (v: number, lo = 0, hi = 1) => Math.min(hi, Math.max(lo, v));

/** The network as a turnable 3-D scene: every pass sends one input through it, and the weights follow the training. */
export function NetworkView({ net }: { net: NetView }) {
  const { theme } = useTheme();
  const canvas = useRef<HTMLCanvasElement>(null);
  const hud = useRef<HTMLSpanElement>(null);
  const latest = useRef(net);
  const cam = useRef({ yaw: YAW, pitch: PITCH, toYaw: YAW, toPitch: PITCH, x: 0, y: 0, drag: false });

  useEffect(() => {
    latest.current = net;
  }, [net]);

  useEffect(() => {
    const el = canvas.current;
    const ctx = el?.getContext("2d");
    if (!el || !ctx) return;
    const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let ink: { pos: string; neg: string; line: string; text: string; surface: string; font: string } | null = null;
    let frame = 0;
    let seen = true;
    let drawn: NetView | null = null;
    let told = -1;
    let layout: Float32Array[] = [];
    let weights: Float32Array[] = [];
    let peak: number[] = [];
    const io = new IntersectionObserver(([entry]) => { seen = entry.isIntersecting; });
    io.observe(el);

    const draw = (now: number) => {
      frame = requestAnimationFrame(draw);
      const view = latest.current;
      if (!seen) return;
      const dpr = window.devicePixelRatio || 1;
      const W = el.clientWidth;
      const H = el.clientHeight;
      if (el.width !== Math.round(W * dpr) || el.height !== Math.round(H * dpr)) {
        el.width = Math.round(W * dpr);
        el.height = Math.round(H * dpr);
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      if (!ink) { // read here, not in the effect: the theme reaches the document after this component's effects
        const css = getComputedStyle(document.documentElement);
        const color = (name: string) => css.getPropertyValue(name).trim();
        ink = { pos: color("--series-1"), neg: color("--series-2"), line: color("--axis"), text: color("--subtle"),
                surface: color("--surface"), font: `11px ${getComputedStyle(el).fontFamily}` };
      }

      if (view !== drawn) {
        if (drawn?.shown.join() !== view.shown.join()) {
          layout = layerLayout(view.shown);
          weights = view.w.map((link) => Float32Array.from(link, (v) => v ?? 0));
        }
        peak = view.act.map((layer) => layer.reduce<number>((m, v) => Math.max(m, Math.abs(v ?? 0)), 1e-9));
        drawn = view;
        told = -1;
      }
      view.w.forEach((link, j) => { // the weights glide to the newest snapshot instead of jumping
        const w = weights[j];
        for (let i = 0; i < w.length; i++) w[i] += ((link[i] ?? 0) - w[i]) * (still ? 1 : EASE);
      });

      const c = cam.current;
      c.yaw += (c.toYaw - c.yaw) * EASE;
      c.pitch += (c.toPitch - c.pitch) * EASE;
      const yaw = c.yaw + (still ? 0 : 0.2 * Math.sin(now / 3600));
      const links = view.w.length;
      const phase = (now % PASS_MS) / PASS_MS;
      const probe = still ? view.y.length >> 1 : pingPong(Math.floor(now / PASS_MS), view.y.length);
      const front = still ? Infinity : phase * (links + 0.6); // how far this pass has come, in layers
      const envelope = still ? 1 : Math.min(1, phase / 0.06, (1 - phase) / 0.1);

      if (probe !== told && hud.current) {
        told = probe;
        const dim = view.sizes[0];
        const x = view.x.slice(probe * dim, (probe + 1) * dim).map(sci).join(", ");
        hud.current.textContent = `${dim > 1 ? `(${x})` : `x = ${x}`}  →  ŷ = ${sci(view.y[probe])}`;
      }

      const unit = Math.min(W / (LAYER_GAP * links + 2.4), (H - 28) / 2.4);
      const cx = W / 2;
      const cy = (H - 22) / 2;
      const seenAt = layout.map((layer) => {
        const out = new Float32Array(layer.length);
        for (let i = 0; i < layer.length; i += 3) {
          const [x, y, scale] = project(layer[i], layer[i + 1], layer[i + 2], yaw, c.pitch, DISTANCE);
          out[i] = cx + x * unit;
          out[i + 1] = cy - y * unit;
          out[i + 2] = scale;
        }
        return out;
      });

      ctx.lineCap = "round";
      for (let j = 0; j < links; j++) {
        const a = seenAt[j];
        const b = seenAt[j + 1];
        const w = weights[j];
        const nA = view.shown[j];
        const nB = view.shown[j + 1];
        let top = 1e-9;
        for (let i = 0; i < w.length; i++) top = Math.max(top, Math.abs(w[i]));
        const floor = w.length > 120 ? 0.12 : 0; // a dense link shows only its stronger weights
        for (let i = 0; i < nA; i++) {
          for (let o = 0; o < nB; o++) {
            const v = w[i * nB + o];
            const size = Math.abs(v) / top;
            if (size <= floor) continue;
            ctx.globalAlpha = 0.05 + 0.45 * size;
            ctx.strokeStyle = v >= 0 ? ink.pos : ink.neg;
            ctx.lineWidth = (0.4 + 1.6 * size) * a[i * 3 + 2];
            ctx.beginPath();
            ctx.moveTo(a[i * 3], a[i * 3 + 1]);
            ctx.lineTo(b[o * 3], b[o * 3 + 1]);
            ctx.stroke();
          }
        }

        const along = front - j;
        if (along < 0 || along > 1) continue;
        // what unit i sends to unit o: its value times the weight; a KAN link carries its own strength
        const values = view.act[j];
        const sent = (i: number, o: number) => (view.kan ? 1 : values[probe * nA + i] ?? 0) * w[i * nB + o];
        let most = 1e-9;
        for (let i = 0; i < nA; i++) for (let o = 0; o < nB; o++) most = Math.max(most, Math.abs(sent(i, o)));
        const eased = along * along * (3 - 2 * along);
        for (let i = 0; i < nA; i++) {
          for (let o = 0; o < nB; o++) {
            const s = sent(i, o);
            const size = Math.abs(s) / most;
            if (size < 0.3) continue;
            ctx.globalAlpha = size * envelope;
            ctx.fillStyle = s >= 0 ? ink.pos : ink.neg;
            ctx.beginPath();
            ctx.arc(a[i * 3] + (b[o * 3] - a[i * 3]) * eased, a[i * 3 + 1] + (b[o * 3 + 1] - a[i * 3 + 1]) * eased,
                    (1.2 + 2.2 * size) * a[i * 3 + 2], 0, 2 * Math.PI);
            ctx.fill();
          }
        }
      }

      const units: [number, number][] = [];
      view.shown.forEach((n, j) => { for (let i = 0; i < n; i++) units.push([j, i]); });
      units.sort((u, v) => seenAt[u[0]][u[1] * 3 + 2] - seenAt[v[0]][v[1] * 3 + 2]); // far units first
      const radius = clamp(unit * 0.11, 3, 7);
      for (const [j, i] of units) {
        const p = seenAt[j];
        const value = view.act[j][probe * view.shown[j] + i] ?? 0;
        const glow = (Math.abs(value) / peak[j]) * clamp((front - j) / 0.3 + 0.5) * envelope;
        const circle = () => {
          ctx.beginPath();
          ctx.arc(p[i * 3], p[i * 3 + 1], radius * p[i * 3 + 2], 0, 2 * Math.PI);
        };
        ctx.globalAlpha = 1;
        ctx.fillStyle = ink.surface; // a solid base hides the links behind the unit
        circle();
        ctx.fill();
        ctx.globalAlpha = glow;
        ctx.fillStyle = ctx.shadowColor = value >= 0 ? ink.pos : ink.neg;
        ctx.shadowBlur = 14 * glow;
        circle();
        ctx.fill();
        ctx.shadowBlur = 0;
        ctx.globalAlpha = 1;
        ctx.strokeStyle = ink.line;
        ctx.lineWidth = 1;
        circle();
        ctx.stroke();
      }

      ctx.fillStyle = ink.text;
      ctx.font = ink.font;
      ctx.textAlign = "center";
      view.names.forEach((name, j) => {
        const [x] = project((j - links / 2) * LAYER_GAP, 0, 0, yaw, c.pitch, DISTANCE);
        const thinned = view.shown[j] < view.sizes[j] && unit * LAYER_GAP > 110 ? ` (${view.shown[j]} drawn)` : "";
        ctx.fillText(`${name} · ${view.sizes[j]}${thinned}`, cx + x * unit, H - 8);
      });
    };

    frame = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(frame);
      io.disconnect();
    };
  }, [theme]);

  const turn = (yaw: number, pitch: number) => {
    const c = cam.current;
    c.toYaw += yaw;
    c.toPitch = clamp(c.toPitch + pitch, -1.1, 1.1);
  };
  const KEYS: Record<string, [number, number]> = { ArrowLeft: [-0.2, 0], ArrowRight: [0.2, 0], ArrowUp: [0, -0.15], ArrowDown: [0, 0.15] };

  return (
    <div className="space-y-1.5">
      <div className="relative">
        <canvas
          ref={canvas}
          data-testid="network-view"
          role="img"
          tabIndex={0}
          aria-label={`The network in three dimensions: layers of ${net.sizes.join(", ")} units. Arrow keys turn it, Home resets the view.`}
          className="viewport block h-80 w-full cursor-grab touch-pan-y rounded-md active:cursor-grabbing"
          onPointerDown={(e) => {
            Object.assign(cam.current, { drag: true, x: e.clientX, y: e.clientY });
            e.currentTarget.setPointerCapture(e.pointerId);
          }}
          onPointerMove={(e) => {
            const c = cam.current;
            if (!c.drag) return;
            turn((e.clientX - c.x) * 0.008, (e.clientY - c.y) * 0.008);
            c.x = e.clientX;
            c.y = e.clientY;
          }}
          onPointerUp={() => { cam.current.drag = false; }}
          onPointerCancel={() => { cam.current.drag = false; }}
          onDoubleClick={() => Object.assign(cam.current, { toYaw: YAW, toPitch: PITCH })}
          onKeyDown={(e) => {
            if (e.key === "Home") Object.assign(cam.current, { toYaw: YAW, toPitch: PITCH });
            else if (e.key in KEYS) turn(...KEYS[e.key]);
            else return;
            e.preventDefault();
          }}
        />
        <span ref={hud} aria-hidden="true" className="tnum pointer-events-none absolute top-2 left-2.5 rounded bg-surface/80 px-1.5 py-0.5 font-mono text-xs whitespace-pre text-muted-foreground backdrop-blur-sm" />
      </div>
      <p className="text-xs text-muted-foreground">
        Each pass sends one input through the network{net.kan ? "" : ": a dot carries a unit's value times the weight"}. Blue is positive, orange negative; a brighter unit and a thicker link mean a larger value. Drag to turn, double-click to reset.
      </p>
    </div>
  );
}

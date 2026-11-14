/** Geometry of the network view: where the units sit in space and how the camera sees them. */

export const LAYER_GAP = 2.2;

/** Units of every layer as flat [x, y, z, ...]: layers along x, each one a near-square panel at most 2 tall. */
export function layerLayout(shown: number[]): Float32Array[] {
  return shown.map((n, j) => {
    const rows = Math.ceil(Math.sqrt(n));
    const cols = Math.ceil(n / rows);
    const pitch = Math.min(0.5, 2 / rows);
    const out = new Float32Array(n * 3);
    for (let k = 0; k < n; k++) {
      out[k * 3] = (j - (shown.length - 1) / 2) * LAYER_GAP;
      out[k * 3 + 1] = ((rows - 1) / 2 - (k % rows)) * pitch;
      out[k * 3 + 2] = (Math.floor(k / rows) - (cols - 1) / 2) * pitch;
    }
    return out;
  });
}

/** Turn a point by yaw (around the vertical axis) and pitch, then apply perspective: [x, y, scale]. */
export function project(x: number, y: number, z: number, yaw: number, pitch: number, distance: number): [number, number, number] {
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  const x1 = x * cy + z * sy;
  const z1 = z * cy - x * sy;
  const scale = distance / (distance + y * sp + z1 * cp);
  return [x1 * scale, (y * cp - z1 * sp) * scale, scale];
}

/** Step 0, 1, ... n-1, n-2, ... 0, 1, ...: the probe input sweeps the range and comes back. */
export function pingPong(step: number, n: number): number {
  if (n < 2) return 0;
  const m = step % (2 * (n - 1));
  return m < n ? m : 2 * (n - 1) - m;
}

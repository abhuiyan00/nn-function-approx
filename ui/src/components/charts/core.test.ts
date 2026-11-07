import { describe, expect, it } from "vitest";
import { inkOn, rampStops, tickLabel, ticksOf } from "./core";

describe("ticksOf", () => {
  it("uses 1-2-5 steps when a log axis spans too few decades", () => {
    expect(ticksOf("log", [44, 337], 400)).toEqual([50, 100, 200]);
  });
  it("uses decades on a long log axis", () => {
    expect(ticksOf("log", [1e-6, 1], 600)).toEqual([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]);
  });
  it("thins decades to fit a short axis", () => {
    expect(ticksOf("log", [1e-15, 1e-1], 200).length).toBeLessThanOrEqual(4);
  });
});

describe("tickLabel", () => {
  it.each([[50, "50"], [200, "200"], [1000, "1,000"], [1e-5, "1e-5"], [0.1, "0.1"], [0.002, "0.002"]])("labels %s as %s", (v, text) => {
    expect(tickLabel("log", v)).toBe(text);
  });
});

describe("inkOn", () => {
  const luminance = (rgb: number[]) => {
    const [r, g, b] = rgb.map((v) => (v / 255 <= 0.03928 ? v / 255 / 12.92 : ((v / 255 + 0.055) / 1.055) ** 2.4));
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  };
  it.each(["light", "dark"] as const)("keeps 4.5:1 on every step of the %s ramp", (theme) => {
    for (const hex of rampStops("sequential", theme)) {
      const rgb = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16)) as [number, number, number];
      const fill = luminance(rgb);
      const text = inkOn(rgb) === "#ffffff" ? 1 : 0;
      const ratio = (Math.max(fill, text) + 0.05) / (Math.min(fill, text) + 0.05);
      expect(ratio).toBeGreaterThanOrEqual(4.5);
    }
  });
});

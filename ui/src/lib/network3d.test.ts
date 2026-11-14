import { describe, expect, it } from "vitest";
import { LAYER_GAP, layerLayout, pingPong, project } from "./network3d";

describe("layerLayout", () => {
  it("centres the layers on x and every panel on y and z", () => {
    const layout = layerLayout([2, 24, 1]);
    expect(layout.map((l) => l.length)).toEqual([6, 72, 3]);
    [-LAYER_GAP, 0, LAYER_GAP].forEach((x, j) => expect(layout[j][0]).toBeCloseTo(x));
    for (const layer of layout) {
      for (const axis of [1, 2]) {
        const values = Array.from(layer).filter((_, i) => i % 3 === axis);
        expect(Math.min(...values) + Math.max(...values)).toBeCloseTo(0);
        expect(Math.max(...values)).toBeLessThanOrEqual(1);
      }
    }
  });

  it("stacks a few units vertically, so they do not hide each other from the front", () => {
    const [two] = layerLayout([2]);
    expect(two[2]).toBe(two[5]);
    expect(two[1]).toBeGreaterThan(two[4]);
  });
});

describe("project", () => {
  it("leaves a point of the screen plane alone when the camera looks straight on", () => {
    expect(project(1, 2, 0, 0, 0, 9)).toEqual([1, 2, 1]);
  });

  it("draws nearer points larger and turns x into depth after a quarter turn", () => {
    expect(project(0, 0, -1, 0, 0, 9)[2]).toBeGreaterThan(1);
    const [x, , scale] = project(1, 0, 0, Math.PI / 2, 0, 9);
    expect(x).toBeCloseTo(0);
    expect(scale).toBeGreaterThan(1);
  });
});

describe("pingPong", () => {
  it("sweeps up and back without repeating the ends", () => {
    expect([0, 1, 2, 3, 4, 5, 6, 7].map((s) => pingPong(s, 4))).toEqual([0, 1, 2, 3, 2, 1, 0, 1]);
    expect(pingPong(5, 1)).toBe(0);
  });
});

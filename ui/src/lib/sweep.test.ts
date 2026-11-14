import { describe, expect, it } from "vitest";
import type { SweepRow } from "./api";
import { formulaDim } from "./functions";
import { parseValues, settingsFor, showValue, sweepVerdict } from "./sweep";

const rows = (seeds: number) => [{ seeds } as SweepRow];

describe("settingsFor", () => {
  it("offers only what the run can use", () => {
    const mlp = settingsFor({ arch: "mlp", optimizer: "adam", points: null });
    expect(mlp).toContain("lr");
    expect(mlp).toContain("width");
    expect(mlp).not.toContain("grid");
    const lm = settingsFor({ arch: "mlp", optimizer: "lm", points: null });
    expect(lm).not.toContain("lr"); // Levenberg-Marquardt has no learning rate
    expect(lm).not.toContain("batch");
    expect(settingsFor({ arch: "kan", optimizer: "lm", points: null })).toEqual(expect.arrayContaining(["grid"]));
    expect(settingsFor({ arch: "kan", optimizer: "lm", points: null })).not.toContain("width");
    expect(settingsFor({ arch: "mlp", optimizer: "adam", points: [[0, 1]] })).not.toContain("n_train");
  });
});

describe("parseValues", () => {
  it("reads lists in the usual notations", () => {
    expect(parseValues("1e-4, 3e-4;1e-3  0.01")).toEqual([1e-4, 3e-4, 1e-3, 0.01]);
    expect(parseValues(showValue(0.0001) + ", " + showValue(64))).toEqual([0.0001, 64]);
  });

  it("refuses one value, repeats, words and long lists", () => {
    expect(parseValues("3")).toBeNull();
    expect(parseValues("2, 2")).toBeNull();
    expect(parseValues("1, two")).toBeNull();
    expect(parseValues(Array.from({ length: 13 }, (_, i) => i).join(","))).toBeNull();
  });
});

describe("sweepVerdict", () => {
  it("calls an effect real only when it exceeds the seed spread by half", () => {
    expect(sweepVerdict({ param: "width", effect: 40, spread: 2, best: 32, rows: rows(3) })).toMatch(/changes by ×40\..*by ×2\.0\. The units per hidden layer matters more.*best at 32/);
    expect(sweepVerdict({ param: "l2", effect: 2.5, spread: 2, best: 0, rows: rows(3) })).toMatch(/within the luck of the starting weights/);
  });

  it("does not judge a study of one seed, or one without results", () => {
    expect(sweepVerdict({ param: "lr", effect: 40, spread: 1, best: 1e-3, rows: rows(1) })).toMatch(/One seed gives no spread/);
    expect(sweepVerdict({ param: "lr", effect: null, spread: null, best: null, rows: [] })).toBeNull();
  });
});

describe("formulaDim", () => {
  it("counts inputs up to the last of x, y, z that is named", () => {
    expect([formulaDim("sin(x)"), formulaDim("x*y"), formulaDim("x + z"), formulaDim("exp(x)")]).toEqual([1, 2, 3, 1]);
  });
});

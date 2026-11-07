import { describe, expect, it } from "vitest";
import type { SearchState } from "./api";
import { boxOf, columnsToPoints, curvesCsv, defaultGap, describeSearch, examplePoints, fitRegions, formulaDim, functionDim, functionFields, isCustom, paintColumns, parsePoints, pointsToColumns, pointsToText, resizeRanges, undefinedRegions } from "./functions";
import { defaultSpec } from "./runs";

describe("the draw pad", () => {
  it("fills every column of a fast stroke, in either direction", () => {
    const blank = Array<number | null>(6).fill(null);
    expect(paintColumns(blank, { col: 1, y: 0 }, { col: 4, y: 3 })).toEqual([null, 0, 1, 2, 3, null]);
    expect(paintColumns(blank, { col: 4, y: 3 }, { col: 1, y: 0 })).toEqual([null, 0, 1, 2, 3, null]);
    expect(paintColumns(blank, { col: 2, y: 5 }, { col: 2, y: 7 })).toEqual([null, null, 7, null, null, null]);
  });
  it("leaves what was drawn before", () => {
    expect(paintColumns([9, null, null], { col: 2, y: 1 }, { col: 2, y: 1 })).toEqual([9, null, 1]);
  });
  it("gives points only where the pen went, across -1 to 1", () => {
    expect(columnsToPoints([0.5, null, -0.5])).toEqual([[-1, 0.5], [1, -0.5]]);
  });
  it("can be restored from its points", () => {
    expect(pointsToColumns([[-1, 0.5], [1, -0.5], [7, 1]], 3)).toEqual([0.5, null, -0.5]);
    expect(pointsToColumns(null, 2)).toEqual([null, null]);
  });
});

describe("examplePoints", () => {
  it("is enough data for the server, and valid text", () => {
    const points = examplePoints();
    expect(points.length).toBeGreaterThanOrEqual(8);
    expect(parsePoints(pointsToText(points)).points).toEqual(points);
  });
});

describe("fitRegions", () => {
  const base = { kind: "1d" as const, x: [-3, -1, 0, 1, 3], exact: [1, 1, 1, 1, 1], box: [-3, 3] as [number, number] };
  it("shades the missing middle", () => {
    expect(fitRegions({ ...base, gap: [[-1, 1]] })).toEqual([{ x0: -1, x1: 1, label: "Missing middle" }]);
  });
  it("shades where the function is undefined", () => {
    expect(fitRegions({ ...base, exact: [1, null, null, 1, 1] })).toEqual([{ x0: -1, x1: 0, label: "Undefined" }]);
  });
  it("shades the extrapolation zone on both sides", () => {
    expect(fitRegions({ ...base, box: [-1, 1] }).map((r) => r.label)).toEqual(["No training data", "No training data"]);
  });
});

describe("describeSearch", () => {
  const net = { key: "MLP", kind: "network", rmse: 0.1, rmse_gap: 0.3, rmse_seen: 0.01 };
  const poly = { key: "Chebyshev LS", kind: "classical", rmse: 0.001, rmse_gap: 0.002, rmse_seen: 0.0005 };
  const state = (s: object) => ({ known: true, rows: [poly, net], winner: "MLP", best_network: "MLP", blind_pick: "MLP", ...s }) as unknown as SearchState;
  it("compares the hole with the rest", () => {
    expect(describeSearch(state({}))[0]).toMatch(/misses the exact function by 0\.01.* by 0\.3 inside the missing middle/);
  });
  it("says when a classical fit is closer", () => {
    expect(describeSearch(state({ winner: "Chebyshev LS" })).join()).toMatch(/Chebyshev LS gets closer than any network here, by ×100/);
  });
  it("warns when validation would pick differently", () => {
    expect(describeSearch(state({ blind_pick: "Chebyshev LS" })).join()).toMatch(/would pick Chebyshev LS/);
  });
  it("does not warn when there is no hole to be wrong about", () => {
    const noHole = state({ blind_pick: "Chebyshev LS", rows: [{ ...net, rmse_gap: null }] });
    expect(describeSearch(noHole).join()).not.toMatch(/would pick/);
  });
  it("speaks of your points when there is no exact function", () => {
    expect(describeSearch(state({ known: false }))[0]).toMatch(/misses your points/);
  });
  it("says that a stretch without points cannot be scored", () => {
    expect(describeSearch(state({ known: false })).join()).toMatch(/nothing can be scored/);
    expect(describeSearch(state({})).join()).not.toMatch(/nothing can be scored/);
  });
});

describe("parsePoints", () => {
  it("reads commas, semicolons, tabs and spaces", () => {
    expect(parsePoints("0, 1\n1;2\n2\t3\n3 4").points).toEqual([[0, 1], [1, 2], [2, 3], [3, 4]]);
  });
  it("skips a header and blank lines", () => {
    const parsed = parsePoints("x,y\n\n0,1\n  \n1,2\n");
    expect(parsed).toEqual({ points: [[0, 1], [1, 2]], error: null });
  });
  it("accepts scientific notation and negatives", () => {
    expect(parsePoints("-1e-3, 2.5E2").points).toEqual([[-0.001, 250]]);
  });
  it("names the first bad line", () => {
    expect(parsePoints("0,1\nabc,2").error).toMatch(/Line 2/);
    expect(parsePoints("0,1\n1").error).toMatch(/Line 2/);
    expect(parsePoints("0,1\n1,2,3").error).toMatch(/Line 2/);
  });
  it("treats only a first line as a possible header", () => {
    expect(parsePoints("0,1\nx,y").error).toMatch(/Line 2/);
  });
  it("gives nothing for empty text", () => {
    expect(parsePoints("  \n").points).toEqual([]);
  });
  it("round-trips through text", () => {
    const points: [number, number][] = [[0, 1], [-0.5, 2.25]];
    expect(parsePoints(pointsToText(points)).points).toEqual(points);
  });
});

describe("formulaDim", () => {
  it.each([["sin(x)", 1], ["x*y", 2], ["abs(y)", 2], ["sinh(x)", 1], ["exp(x) + 2", 1], ["3", 1]])("%s has %i input(s)", (text, dim) => {
    expect(formulaDim(text)).toBe(dim);
  });
});

describe("functionDim", () => {
  const targets = [{ key: "f5", dim: 2 }, { key: "f1", dim: 1 }];
  it("comes from the formula, the points or the target", () => {
    expect(functionDim({ ...defaultSpec, formula: "x*y" }, targets)).toBe(2);
    expect(functionDim({ ...defaultSpec, points: [[0, 1]] }, targets)).toBe(1);
    expect(functionDim({ ...defaultSpec, target: "f5" }, targets)).toBe(2);
    expect(functionDim({ ...defaultSpec, target: "gone" }, targets)).toBe(1);
  });
});

describe("defaultGap", () => {
  it("hides the middle 30 % of every range", () => {
    expect(defaultGap([[-1, 1]])).toEqual([[-0.3, 0.3]]);
    expect(defaultGap([[0, 4], [10, 20]])).toEqual([[1.4, 2.6], [13.5, 16.5]]);
  });
});

describe("resizeRanges", () => {
  it("adds and drops ranges without touching the rest", () => {
    expect(resizeRanges([[0, 5]], 2)).toEqual([[0, 5], [-1, 1]]);
    expect(resizeRanges([[0, 5], [1, 2]], 1)).toEqual([[0, 5]]);
    expect(resizeRanges(null, 2)).toEqual([[-1, 1], [-1, 1]]);
  });
});

describe("boxOf", () => {
  it("gives one range per input", () => {
    expect(boxOf({ kind: "1d", x: [], exact: [], box: [-2, 2] })).toEqual([[-2, 2]]);
    expect(boxOf({ kind: "2d", x: [], y: [], exact: [], box: [[0, 1], [2, 3]] })).toEqual([[0, 1], [2, 3]]);
  });
});

describe("isCustom and functionFields", () => {
  it("tell a typed function from a built-in one", () => {
    expect(isCustom(defaultSpec)).toBe(false);
    expect(isCustom({ ...defaultSpec, formula: "x" })).toBe(true);
    expect(isCustom({ ...defaultSpec, points: [] })).toBe(true);
  });
  it("send only the function, and only what is set", () => {
    expect(functionFields({ ...defaultSpec, target: "m1" })).toEqual({ target: "m1" });
    expect(functionFields({ ...defaultSpec, formula: "x", domain: [[0, 1]], gap: [] })).toEqual({ target: "f1", formula: "x", domain: [[0, 1]], gap: [] });
  });
});

describe("undefinedRegions", () => {
  it("turns runs of missing values into spans", () => {
    expect(undefinedRegions([0, 1, 2, 3, 4, 5], [1, null, null, 2, 3, null])).toEqual([{ x0: 1, x1: 2 }, { x0: 5, x1: 5 }]);
  });
  it("has nothing to mark when all or none are missing", () => {
    expect(undefinedRegions([0, 1], [1, 2])).toEqual([]);
    expect(undefinedRegions([0, 1], [null, null])).toEqual([]);
  });
});

describe("curvesCsv", () => {
  const state = {
    known: true,
    grid: { kind: "1d", x: [0, 1], exact: [1, 2], box: [0, 1] },
    rows: [
      { key: "A", pred: [1.5, 2.5] },
      { key: 'B, "quoted"', pred: [3, null] },
      { key: "No curve yet" },
    ],
  } as unknown as SearchState;
  it("has one column per curve and quotes what needs it", () => {
    expect(curvesCsv(state)).toBe('x,exact,A,"B, ""quoted"""\n0,1,1.5,3\n1,2,2.5,\n');
  });
  it("has no exact column for measured data", () => {
    expect(curvesCsv({ ...state, known: false }).split("\n")[0]).toBe('x,A,"B, ""quoted"""');
  });
  it("lists both inputs of a surface", () => {
    const surface = { known: true, grid: { kind: "2d", x: [0, 1], y: [5], exact: [7, 8], box: [[0, 1], [5, 6]] }, rows: [{ key: "A", pred: [1, 2] }] } as unknown as SearchState;
    expect(curvesCsv(surface)).toBe("x,y,exact,A\n0,5,7,1\n1,5,8,2\n");
  });
});

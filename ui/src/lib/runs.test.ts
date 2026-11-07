import { describe, expect, it } from "vitest";
import { aggregate, cliCommand, defaultSpec, paramCount, parseIntList, toCsv, validateSpec } from "./runs";

describe("parseIntList", () => {
  it("accepts commas, spaces and an × shape", () => {
    expect(parseIntList("16,16")).toEqual([16, 16]);
    expect(parseIntList(" 8 16 ")).toEqual([8, 16]);
    expect(parseIntList("3x64")).toEqual([64, 64, 64]);
    expect(parseIntList("2×16")).toEqual([16, 16]);
  });
  it("rejects anything else", () => {
    expect(parseIntList("")).toBeNull();
    expect(parseIntList("a,b")).toBeNull();
    expect(parseIntList("1.5")).toBeNull();
    expect(parseIntList("0,4")).toBeNull();
  });
});

describe("paramCount mirrors funcprox.model", () => {
  it.each([
    [{ hidden: [32] }, 1, 97],
    [{ hidden: [8, 8] }, 1, 97],
    [{ hidden: [64, 64], fourier_m: 32 }, 1, 8385],
    [{ arch: "kan" as const, widths: [2, 5, 1], grid: 5 }, 2, 141],
    [{ arch: "kan" as const, widths: [2, 5, 1], grid: 3 }, 2, 111],
  ])("%j in %id has %i parameters", (spec, dim, n) => {
    expect(paramCount({ ...defaultSpec, ...spec }, dim)).toBe(n);
  });
});

describe("validateSpec", () => {
  it("accepts the default", () => expect(validateSpec(defaultSpec, 1)).toEqual({}));
  it("flags a network too large for Levenberg-Marquardt", () => {
    const e = validateSpec({ ...defaultSpec, optimizer: "lm", hidden: [64, 64] }, 1);
    expect(e.hidden).toMatch(/Levenberg-Marquardt/);
  });
  it("flags KAN widths that do not fit the target", () => {
    expect(validateSpec({ ...defaultSpec, arch: "kan", widths: [1, 5, 1] }, 2).widths).toMatch(/start with 2/);
    expect(validateSpec({ ...defaultSpec, arch: "kan", widths: [2, 5, 2] }, 2).widths).toMatch(/end with 1/);
  });
});

describe("cliCommand", () => {
  it("lists only what differs from the CLI defaults", () => {
    expect(cliCommand({ ...defaultSpec, target: "f4", hidden: [16, 16], optimizer: "lm", epochs: 300, patience: 30 }))
      .toBe("python -m funcprox train --target f4 --hidden 16,16 --optimizer lm --patience 30");
  });
  it("describes a KAN with its grid updates", () => {
    expect(cliCommand({ ...defaultSpec, target: "f7", arch: "kan", widths: [2, 1, 1], grid: 10, grid_updates: [25, 75], optimizer: "lm" }))
      .toBe("python -m funcprox train --target f7 --arch kan --widths 2,1,1 --grid 10 --grid-updates 25,75 --optimizer lm");
  });
});

describe("aggregate", () => {
  const rows = [
    { n: 10, rmse: 1, m: "a" },
    { n: 10, rmse: 3, m: "a" },
    { n: 10, rmse: 2, m: "a" },
    { n: 5, rmse: 4, m: "a" },
    { n: 5, rmse: 0, m: "b" },
    { n: null, rmse: 1, m: "b" },
    { n: 5, rmse: 7, m: "b" },
  ];
  it("gives median and seed range per x, sorted by x", () => {
    const [a] = aggregate(rows, "n", "rmse", "m");
    expect(a.key).toBe("a");
    expect(a.points).toEqual([
      { x: 5, y: 4, lo: 4, hi: 4, n: 1 },
      { x: 10, y: 2, lo: 1, hi: 3, n: 3 },
    ]);
  });
  it("drops values a log axis cannot draw", () => {
    const [, b] = aggregate(rows, "n", "rmse", "m", { positiveY: true });
    expect(b.points).toEqual([{ x: 5, y: 7, lo: 7, hi: 7, n: 1 }]);
  });
  it("keeps series order stable across filters", () => {
    expect(aggregate(rows, "n", "rmse", "m", { order: ["b", "a"] }).map((s) => s.key)).toEqual(["b", "a"]);
  });
});

describe("toCsv", () => {
  it("quotes what needs quoting", () => {
    expect(toCsv(["a", "b"], [{ a: "x,y", b: 'q"r' }, { a: null, b: 1 }])).toBe('a,b\n"x,y","q""r"\n,1\n');
  });
});

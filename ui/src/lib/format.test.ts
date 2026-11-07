import { describe, expect, it } from "vitest";
import { daysUntil, duration, factor, integer, sci } from "./format";

describe("sci", () => {
  it.each([
    [0, "0"],
    [0.0123, "0.0123"],
    [0.65, "0.65"],
    [123.4, "123"],
    [1.54e-3, "1.54e-3"],
    [9.12e-16, "9.12e-16"],
    [1234, "1.23e+3"],
    [-2.5e-4, "-2.50e-4"],
  ])("formats %s as %s", (value, text) => expect(sci(value)).toBe(text));

  it("shows a dash for a missing value", () => {
    expect(sci(null)).toBe("—");
    expect(sci(undefined)).toBe("—");
    expect(sci(Number.NaN)).toBe("—");
  });
});

describe("factor", () => {
  it.each([
    [81.2, "×81"],
    [1.94, "×1.9"],
    [2.2e10, "×2.2e+10"],
  ])("formats %s as %s", (value, text) => expect(factor(value)).toBe(text));
});

describe("duration", () => {
  it.each([
    [0.42, "0.42 s"],
    [12.34, "12.3 s"],
    [65, "1 min 5 s"],
    [3700, "1 h 1 min"],
  ])("formats %s seconds as %s", (value, text) => expect(duration(value)).toBe(text));
});

describe("daysUntil", () => {
  it("counts whole calendar days", () => {
    expect(daysUntil("2026-12-14", new Date("2026-11-07T23:30:00"))).toBe(37);
    expect(daysUntil("2026-11-07", new Date("2026-11-07T01:00:00"))).toBe(0);
    expect(daysUntil("2026-11-01", new Date("2026-11-07T12:00:00"))).toBe(-6);
  });
});

describe("integer", () => {
  it("groups thousands", () => expect(integer(16897)).toBe("16,897"));
});

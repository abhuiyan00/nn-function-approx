import { describe, expect, it } from "vitest";
import { rewriteHref, rewriteImage } from "./markdown";

describe("rewriteHref", () => {
  it.each([
    ["docs/FINDINGS.md", "/documents/findings"],
    ["FINDINGS.md", "/documents/findings"],
    ["PROJECT_PLAN.md", "/documents/project_plan"],
    ["CONTRIBUTING.md", "/documents/contributing"],
    ["docs/figures/e6_baselines_scaling.png", "/figures/e6_baselines_scaling.png"],
    ["figures/u1.png", "/figures/u1.png"],
    ["docs/screenshots/lab.png", "/screenshots/lab.png"],
    ["https://example.org/x", "https://example.org/x"],
  ])("maps %s to %s", (href, out) => expect(rewriteHref(href)).toBe(out));
  it("refuses script links", () => expect(rewriteHref("javascript:alert(1)")).toBeUndefined());
});

describe("rewriteImage", () => {
  it("serves repository figures from the API", () => {
    expect(rewriteImage("docs/figures/a.png")).toBe("/figures/a.png");
  });
});

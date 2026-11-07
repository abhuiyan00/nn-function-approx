import AxeBuilder from "@axe-core/playwright";
import { type Page, expect, test } from "@playwright/test";

const PAGES = ["/", "/lab", "/experiments", "/experiments/e6", "/experiments/e3", "/explorer", "/models", "/runner", "/documents/findings"];
const EXPERIMENTS = ["e0", "e1", "e2", "e3", "e4", "e5", "e6", "u1", "u2", "u3", "u4"];

/** Fails a test on any console error or uncaught exception in the page. */
function watchErrors(page: Page) {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(`exception: ${e.message}`));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(`console: ${m.text()}`);
  });
  return errors;
}

async function runPreset(page: Page, id: string) {
  await page.goto("/lab");
  await page.getByTestId(`preset-${id}`).click();
  await expect(page.getByTestId("run-status")).toContainText(/Finished|Stopped/, { timeout: 80_000 });
}

test("overview shows numbers computed from the result files", async ({ page }) => {
  const errors = watchErrors(page);
  await page.goto("/");
  await expect(page.getByRole("heading", { level: 1 })).toContainText("function approximation");
  await expect(page.getByText("×54 to ×81")).toBeVisible();
  await expect(page.getByText("90.2 %")).toBeVisible();
  await expect(page.getByRole("cell", { name: "Classical wins" }).first()).toBeVisible();
  await expect(page.getByRole("cell", { name: "Within seed spread" })).toBeVisible();
  await expect(page.getByText(/experiments have quick-profile results only/)).toBeVisible();
  await expect(page.locator("svg[role=img]").first()).toBeVisible();
  expect(errors).toEqual([]);
});

test("a target card opens the lab on that target", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("link", { name: /f3 · multiscale/ }).click();
  await expect(page).toHaveURL(/\/lab\?target=f3/);
  await expect(page.getByRole("heading", { name: /f3 · sin\(πx\)/ })).toBeVisible();
});

test("the lab recovers from a stale stored configuration and a bad address", async ({ page }) => {
  await page.goto("/lab");
  await page.evaluate(() => localStorage.setItem("fapprox-lab-config", JSON.stringify({ target: "f99", removed_setting: 1, hidden: [4] })));
  await page.goto("/lab");
  await expect(page.getByRole("button", { name: "Train" })).toBeVisible();
  await page.getByRole("button", { name: "Train" }).click();
  await expect(page.getByTestId("run-title")).toContainText("f1 · MLP 1×4 tanh");
  await page.getByRole("button", { name: "Stop" }).click().catch(() => {});
  await page.evaluate(() => localStorage.clear());
  await page.goto("/lab?target=nonsense");
  await expect(page.getByRole("heading", { name: /f1 · sin\(2πx\)/ })).toBeVisible();
});

test("training streams live and ends with metrics and baselines", async ({ page }) => {
  const errors = watchErrors(page);
  await runPreset(page, "lm-runge");
  await expect(page.getByTestId("run-title")).toHaveText(/f4 · MLP 2×16 tanh · LM/);
  const rmse = Number(await page.getByTestId("metric-rmse").innerText());
  expect(rmse).toBeLessThan(1e-3); // U1: LM gets far below this on Runge
  await expect(page.getByText(/beats this network by|This network beats/)).toBeVisible();
  await page.getByRole("radio", { name: "Damping μ" }).click();
  await expect(page.getByRole("img", { name: /damping/i })).toBeVisible();
  await page.getByRole("radio", { name: "Spectrum" }).click();
  await expect(page.getByRole("img", { name: /Amplitude spectrum/ })).toBeVisible();
  expect(errors).toEqual([]);
});

test("a two-dimensional target draws three surfaces", async ({ page }) => {
  const errors = watchErrors(page);
  await runPreset(page, "franke");
  await expect(page.locator("canvas[role=img]")).toHaveCount(3);
  const alpha = (c: HTMLCanvasElement) => c.getContext("2d")!.getImageData(0, 0, 1, 1).data[3];
  for (const i of [0, 1, 2]) expect(await page.locator("canvas[role=img]").nth(i).evaluate(alpha)).toBe(255);
  await expect(page.getByRole("radio", { name: "Spectrum" })).toHaveCount(0);
  expect(errors).toEqual([]);
});

test("the KAN preset reads the formula back", async ({ page }) => {
  const errors = watchErrors(page);
  await runPreset(page, "kan-f7");
  await expect(page.getByRole("radio", { name: "Edges" })).toBeChecked();
  await expect(page.getByText(/R² = /)).toHaveCount(3);
  expect(errors).toEqual([]);
});

test("spectral preset opens the spectrum and tracks each frequency", async ({ page }) => {
  await page.goto("/lab");
  await page.getByTestId("preset-spectral-fourier").click();
  await expect(page.getByRole("radio", { name: "Spectrum" })).toBeChecked();
  await expect(page.getByRole("img", { name: /per frequency component/ })).toBeVisible({ timeout: 30_000 });
  await page.getByRole("button", { name: "Stop" }).click();
  await expect(page.getByTestId("run-status")).toContainText("Stopped", { timeout: 30_000 });
});

test("the form refuses a network too large for Levenberg-Marquardt", async ({ page }) => {
  await page.goto("/lab");
  await page.evaluate(() => localStorage.clear());
  await page.reload();
  await page.getByRole("combobox", { name: "Optimiser" }).click();
  await page.getByRole("option", { name: "Levenberg-Marquardt" }).click();
  await page.getByLabel("Hidden layers").fill("64,64");
  await expect(page.getByText(/exceed its limit of 1,500/)).toBeVisible();
  await expect(page.getByRole("button", { name: "Train" })).toBeDisabled();
  await page.getByLabel("Hidden layers").fill("16,16");
  await expect(page.getByTestId("param-count")).toHaveText("321");
  await expect(page.getByRole("button", { name: "Train" })).toBeEnabled();
});

test("runs can be compared side by side", async ({ page }) => {
  await runPreset(page, "adam-runge");
  await page.getByTestId("preset-lm-runge").click();
  await expect(page.getByTestId("run-title")).toContainText("LM");
  await expect(page.getByTestId("run-status")).toContainText("Finished", { timeout: 60_000 });
  const boxes = page.getByRole("checkbox", { name: /Compare/ });
  await boxes.nth(0).click();
  await boxes.nth(1).click();
  await page.getByRole("radio", { name: /Compare/ }).click();
  await expect(page.getByRole("img", { name: /Validation curves/ })).toBeVisible();
});

test("a saved model can be reloaded and queried", async ({ page }) => {
  const errors = watchErrors(page);
  await runPreset(page, "lm-runge");
  await page.getByRole("button", { name: /^Save MLP/ }).first().click();
  await page.getByLabel("Name").fill("e2e-runge");
  await page.getByRole("button", { name: "Save model" }).click();
  await expect(page.getByText("Saved as e2e-runge").first()).toBeVisible();

  await page.goto("/models?model=e2e-runge");
  await expect(page.getByRole("button", { name: "e2e-runge" })).toBeVisible();
  await page.getByLabel(/Input values/).fill("0, 3");
  await page.getByRole("button", { name: "Predict" }).click();
  await expect(page.getByText("Outside the training box")).toBeVisible();
  await expect(page.getByRole("row", { name: /^0 / })).toContainText("1");
  expect(errors).toEqual([]);
});

for (const id of EXPERIMENTS) {
  test(`experiment ${id} draws its results`, async ({ page }) => {
    const errors = watchErrors(page);
    await page.goto(`/experiments/${id}`);
    await expect(page.getByRole("heading", { level: 1 })).toContainText(id.toUpperCase());
    await expect(page.getByRole("heading", { name: "What we found" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Interactive results" })).toBeVisible();
    await expect(page.locator("svg[role=img], table[class*=min-w]").first()).toBeVisible();
    await expect(page.getByText("Raw data")).toBeVisible();
    expect(errors).toEqual([]);
  });
}

test("charts answer keyboard users and offer a table", async ({ page }) => {
  await page.goto("/experiments/e6");
  const chart = page.getByRole("img", { name: /classical approximators/ });
  await chart.focus();
  await page.keyboard.press("ArrowRight");
  await expect(page.getByText(/Parameters or coefficients: /)).toBeVisible();
  await page.getByRole("button", { name: "Show as table" }).click();
  await expect(page.getByRole("columnheader", { name: "Median" })).toBeVisible();
});

test("explorer keeps its choices in the address", async ({ page }) => {
  const errors = watchErrors(page);
  await page.goto("/explorer?data=u2/runs");
  await expect(page.getByRole("combobox", { name: "Only target" })).toHaveText("f7"); // medians across targets mean nothing
  await page.getByRole("combobox", { name: "Vertical axis" }).click();
  await page.getByRole("option", { name: "max_err" }).click();
  await expect(page).toHaveURL(/y=max_err/);
  await page.getByRole("combobox", { name: "Only target" }).click();
  await page.getByRole("option", { name: "f5" }).click();
  await expect(page).toHaveURL(/f\.target=f5/);
  await expect(page.getByText(/rows ·/)).toContainText(/^\d+ rows/);
  const url = page.url();
  await page.goto(url);
  await expect(page.getByRole("heading", { name: "max err against n params" })).toBeVisible();
  expect(errors).toEqual([]);
});

test("the runner warns before overwriting results and shows the log", async ({ page }) => {
  const errors = watchErrors(page);
  await page.goto("/runner?experiment=e1");
  await expect(page.getByRole("checkbox", { name: /E1/ })).toBeChecked();
  await page.getByRole("button", { name: /^Run 1/ }).click();
  await expect(page.getByRole("dialog")).toContainText("overwrites the current results");
  await page.getByRole("button", { name: "Overwrite and run" }).click();
  await expect(page.getByText("stand-in for the experiment σ")).toBeVisible();
  await expect(page.getByRole("cell", { name: "Finished" })).toBeVisible({ timeout: 20_000 });
  expect(errors).toEqual([]);
});

test("documents render, including diagrams and internal links", async ({ page }) => {
  const errors = watchErrors(page);
  await page.goto("/documents/project_plan");
  await expect(page.getByRole("heading", { level: 1 })).toContainText("Project plan");
  await expect(page.locator("article svg").first()).toBeVisible({ timeout: 20_000 });
  await page.getByRole("link", { name: "REPORT_OUTLINE.md" }).click();
  await expect(page).toHaveURL(/\/documents\/report_outline/);
  await page.goto("/documents/readme");
  await expect(page.locator("article svg").first()).toBeVisible({ timeout: 20_000 });
  await expect(page.locator("article img").first()).toBeVisible();
  const screenshot = page.locator('article img[src^="/screenshots/"]').first();
  await screenshot.scrollIntoViewIfNeeded();
  await expect.poll(() => screenshot.evaluate((img: HTMLImageElement) => img.naturalWidth)).toBeGreaterThan(0);
  await page.goto("/documents/nope");
  await expect(page.getByText("This document does not exist")).toBeVisible();
  expect(errors.filter((e) => !e.includes("404"))).toEqual([]);
});

test("an unknown address shows a way back", async ({ page }) => {
  await page.goto("/no-such-page");
  await expect(page.getByText("This page does not exist")).toBeVisible();
});

test("the theme toggle sticks across a reload", async ({ page }) => {
  await page.goto("/");
  await page.evaluate(() => localStorage.clear());
  await page.reload();
  const html = page.locator("html");
  const before = await html.getAttribute("data-theme");
  await page.getByRole("button", { name: /Switch to/ }).click();
  await expect(html).not.toHaveAttribute("data-theme", before ?? "");
  const after = await html.getAttribute("data-theme");
  await page.reload();
  await expect(html).toHaveAttribute("data-theme", after ?? "");
});

test.describe("on a phone", () => {
  test.use({ viewport: { width: 390, height: 844 } });

  for (const path of PAGES) {
    test(`${path} has no sideways page scroll`, async ({ page }) => {
      await page.goto(path);
      await page.waitForLoadState("networkidle");
      const overflow = await page.evaluate(() => {
        const main = document.getElementById("main")!;
        return { page: document.documentElement.scrollWidth - innerWidth, main: main.scrollWidth - main.clientWidth };
      });
      expect(overflow.page).toBeLessThanOrEqual(0);
      expect(overflow.main).toBeLessThanOrEqual(0);
    });
  }

  test("navigation moves into a drawer", async ({ page }) => {
    await page.goto("/");
    await page.getByRole("button", { name: "Open navigation" }).click();
    await page.getByRole("link", { name: "Training Lab" }).click();
    await expect(page).toHaveURL(/\/lab/);
    await expect(page.getByRole("dialog")).toHaveCount(0);
  });
});

for (const theme of ["light", "dark"]) {
  test.describe(`accessibility, ${theme} theme`, () => {
    for (const path of PAGES) {
      test(`${path} has no serious axe violations`, async ({ page }) => {
        await page.addInitScript((t) => localStorage.setItem("fapprox-theme", t), theme);
        await page.goto(path);
        await page.waitForLoadState("networkidle");
        const { violations } = await new AxeBuilder({ page }).analyze();
        const serious = violations.filter((v) => v.impact === "serious" || v.impact === "critical");
        expect(serious.map((v) => `${v.id}: ${v.nodes.slice(0, 3).map((n) => n.target.join(" ")).join(" | ")}`)).toEqual([]);
      });
    }
  });
}

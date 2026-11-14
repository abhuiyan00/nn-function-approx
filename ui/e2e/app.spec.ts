import AxeBuilder from "@axe-core/playwright";
import { type Page, expect, test } from "@playwright/test";

const PAGES = ["/", "/finder", "/lab", "/experiments", "/experiments/e6", "/experiments/e3", "/explorer", "/models", "/runner", "/documents/findings"];
const EXPERIMENTS = ["e0", "e1", "e2", "e3", "e4", "e5", "e6", "e7", "u1", "u2", "u3", "u4"];

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

async function openFinder(page: Page) {
  await page.goto("/finder");
  await page.evaluate(() => localStorage.clear());
  await page.reload();
  await expect(page.getByTestId("preview-title")).toBeVisible();
}

const findButton = (page: Page) => page.getByRole("button", { name: "Find the closest curve" });

test("the curve finder ranks every model on a function with a missing middle", async ({ page }) => {
  test.setTimeout(180_000); // six real trainings, on whatever machine runs the tests
  const errors = watchErrors(page);
  await openFinder(page);
  await expect(page.getByTestId("preview-title")).toContainText("m1");
  await expect(page.getByRole("switch", { name: "Hide a part of the range" })).toBeChecked();
  await findButton(page).click();
  await expect(page.getByTestId("search-status")).toContainText("Finished", { timeout: 120_000 });
  await expect(page.getByTestId("ranking-row")).toHaveCount(9); // six networks, polynomial, spline, RBF network
  await expect(page.getByRole("columnheader", { name: "In the hole" })).toBeVisible();
  await expect(page.getByRole("img", { name: /curves found/ })).toBeVisible();
  await expect(page.locator("text=Missing middle").first()).toBeVisible();

  const curves = await page.getByRole("checkbox", { name: /on the chart/ }).count();
  expect(curves).toBe(9);
  await page.getByRole("checkbox", { name: /Show Cubic spline LS/ }).check();
  await expect(page.getByRole("img", { name: /curves found/ }).locator("..").locator("..").getByText("Cubic spline LS")).toBeVisible();

  await page.getByLabel(/Input values/).fill("0, 0.5");
  await page.getByRole("button", { name: "Predict" }).click();
  await expect(page.getByRole("row", { name: /^0 In the missing middle/ })).toBeVisible();

  await page.getByRole("link", { name: /Open MLP 2×16 tanh · LM in the Training Lab/ }).click();
  await expect(page).toHaveURL(/\/lab\?run=/);
  await expect(page.getByTestId("metric-gap")).toBeVisible();
  expect(errors).toEqual([]);
});

test("a typed formula is checked while typing and explains mistakes", async ({ page }) => {
  await openFinder(page);
  await page.getByRole("radio", { name: "Formula" }).click();
  const formula = page.getByLabel("f(x) =");
  await formula.fill("2x");
  await expect(page.getByText(/Write multiplication as 2\*x/)).toBeVisible();
  await expect(findButton(page)).toBeDisabled();
  await formula.fill("open('secret')");
  await expect(page.getByText(/Unknown function 'open'/)).toBeVisible();
  await formula.fill("x^3 - x");
  await expect(page.getByTestId("preview-title")).toHaveText("x^3 - x");
  await expect(findButton(page)).toBeEnabled();

  await page.getByRole("switch", { name: "Hide a part of the range" }).click();
  await expect(page.getByRole("slider", { name: /starts at this x/ })).toBeVisible();
  await expect(page.locator("svg text", { hasText: "Missing middle" })).toBeVisible();

  await formula.fill("x*y");
  await expect(page.getByLabel("f(x, y) =")).toBeVisible();
  await expect(page.getByLabel("y from")).toBeVisible();
  await expect(page.locator("canvas[role=img]:not([data-testid=network-view])").first()).toBeVisible();
});

test("a function undefined in the middle shows the hole by itself", async ({ page }) => {
  await openFinder(page);
  await page.getByRole("radio", { name: "Formula" }).click();
  await page.getByRole("button", { name: "Undefined in the middle" }).click();
  await expect(page.locator("svg text", { hasText: "Undefined" })).toBeVisible();
  await expect(findButton(page)).toBeEnabled();
});

test("pasted points are checked line by line", async ({ page }) => {
  await openFinder(page);
  await page.getByRole("radio", { name: "Points" }).click();
  const box = page.getByLabel("One x, y pair per line");
  await expect(box).not.toBeEmpty();
  await box.fill("0, 1\nnot numbers");
  await expect(page.getByText(/Line 2 is not a pair of numbers/)).toBeVisible();
  await expect(findButton(page)).toBeDisabled();
  await box.fill(Array.from({ length: 20 }, (_, i) => `${i / 10}, ${Math.sin(i / 3)}`).join("\n"));
  await expect(page.getByText("20 points.")).toBeVisible();
  await expect(findButton(page)).toBeEnabled();
  await expect(page.getByRole("img", { name: /Network against the exact function/ })).toBeVisible();
});

test("a drawn curve becomes the training data", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await openFinder(page);
  await page.getByRole("radio", { name: "Draw" }).click();
  await expect(page.getByText("Draw a curve with the mouse or a finger.")).toBeVisible();
  await expect(findButton(page)).toBeDisabled();
  const pad = await page.getByTestId("draw-pad").boundingBox();
  await page.mouse.move(pad!.x + 4, pad!.y + pad!.height / 2);
  await page.mouse.down();
  for (let i = 1; i <= 8; i++) await page.mouse.move(pad!.x + (i / 8) * (pad!.width - 8), pad!.y + pad!.height * (0.5 + 0.3 * Math.sin(i)));
  await page.mouse.up();
  await expect(page.getByText(/\d+ points\. Draw again/)).toBeVisible();
  await expect(findButton(page)).toBeEnabled();
  await page.getByRole("button", { name: "Clear" }).click();
  await expect(findButton(page)).toBeDisabled();
  expect(errors).toEqual([]);
});

test("searching again keeps both the old results and the new function in view", async ({ page }) => {
  await openFinder(page);
  await findButton(page).click();
  await expect(page.getByTestId("search-results")).toBeVisible();
  await page.getByRole("button", { name: "Stop" }).click();
  await expect(page.getByTestId("search-status")).toContainText("Stopped", { timeout: 40_000 });
  await page.getByRole("switch", { name: "Hide a part of the range" }).click();
  await expect(page.getByText(/Changed since the search below/)).toBeVisible();
  await expect(page.getByTestId("search-results")).toBeVisible();
});

test("an address of a finished server's search shows what happened", async ({ page }) => {
  await page.goto("/finder?search=nope");
  await expect(page.getByText("This search is no longer available")).toBeVisible();
});

test("the lab trains on a typed formula with a missing middle", async ({ page }) => {
  await page.goto("/lab");
  await page.evaluate(() => localStorage.clear());
  await page.reload();
  await page.getByRole("radio", { name: "Formula" }).click();
  await page.getByLabel("f(x) =").fill("x^3 - x");
  await page.getByRole("switch", { name: "Hide a part of the range" }).click();
  await page.getByLabel("Epochs").fill("30");
  await page.getByRole("button", { name: "Train" }).click();
  await expect(page.getByTestId("run-title")).toContainText("custom · MLP");
  await expect(page.getByText("f = x^3 - x")).toBeVisible();
  await expect(page.getByTestId("run-status")).toContainText("Finished", { timeout: 60_000 });
  await expect(page.getByTestId("metric-gap")).toBeVisible();
  await expect(page.getByRole("columnheader", { name: "In the missing middle" })).toBeVisible();
});

test("a target card opens the lab on that target", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("link", { name: /f3 · multiscale/ }).click();
  await expect(page).toHaveURL(/\/lab\?target=f3/);
  await expect(page.getByRole("heading", { name: /f3 · sin\(πx\)/ })).toBeVisible();
});

test("the lab recovers from a stale stored configuration and a bad address", async ({ page }) => {
  await page.goto("/lab");
  await page.evaluate(() => localStorage.setItem("funcprox-lab-config", JSON.stringify({ target: "f99", removed_setting: 1, hidden: [4] })));
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
  const view = page.getByTestId("network-view");
  await expect(view).toHaveAccessibleName(/layers of 1, 16, 16, 1 units/);
  // the scene is drawn: some pixel of the canvas is not empty
  await expect.poll(() => view.evaluate((c: HTMLCanvasElement) => c.getContext("2d")!.getImageData(0, 0, c.width, c.height).data.some((v) => v > 0))).toBe(true);
  await page.getByRole("radio", { name: "Damping μ" }).click();
  await expect(page.getByRole("img", { name: /damping/i })).toBeVisible();
  await page.getByRole("radio", { name: "Spectrum" }).click();
  await expect(page.getByRole("img", { name: /Amplitude spectrum/ })).toBeVisible();
  expect(errors).toEqual([]);
});

test("a finished run can be taken apart: units, slope, loss landscape, replay and export", async ({ page }) => {
  const errors = watchErrors(page);
  await runPreset(page, "lm-runge");
  await expect(page.getByTestId("metric-band")).toHaveText(/^± \d/);
  // only <main> scrolls: a document taller than the window lets the whole shell slide away
  expect(await page.evaluate(() => document.documentElement.scrollHeight <= window.innerHeight)).toBe(true);
  await page.getByRole("radio", { name: "Units" }).click();
  await expect(page.getByRole("img", { name: /Share of each hidden unit/ })).toBeVisible();
  await expect(page.getByText("All units + bias = the network")).toBeVisible();
  await page.getByRole("radio", { name: "Slope" }).click();
  await expect(page.getByRole("img", { name: /Slope of the network/ })).toBeVisible();
  expect(Number(await page.getByTestId("slope-rmse").innerText())).toBeLessThan(0.05);
  await page.getByRole("radio", { name: "Loss landscape" }).click();
  await expect(page.getByRole("img", { name: /heat map over direction 1 and direction 2/ })).toBeVisible();
  await expect(page.locator("polyline.draw-in")).toHaveCount(1);

  await page.getByRole("button", { name: "Replay the training" }).click();
  const slider = page.getByRole("slider", { name: "Point in training" });
  await slider.fill("0");
  await expect(page.getByTestId("replay-epoch")).toHaveText("Iteration 0");
  await page.getByRole("button", { name: "Play the replay" }).click();
  await expect(page.getByTestId("replay-epoch")).not.toHaveText("Iteration 0");
  await page.getByRole("button", { name: "Close" }).click();
  await expect(slider).toHaveCount(0);

  const download = page.waitForEvent("download");
  await page.getByTestId("export-device").click();
  const file = await download;
  expect(file.suggestedFilename()).toMatch(/^funcprox_\w+\.py$/);
  await expect(page.getByRole("button", { name: "Print a report" })).toBeVisible();
  expect(errors).toEqual([]);
});

test("a formula of three inputs trains and is shown as a slice", async ({ page }) => {
  const errors = watchErrors(page);
  await page.goto("/lab");
  await page.evaluate(() => localStorage.clear());
  await page.reload();
  await page.getByRole("radio", { name: "Formula" }).click();
  await page.getByRole("button", { name: "3 inputs" }).click();
  await expect(page.getByLabel("f(x, y, z) =")).toHaveValue("sin(pi*x) * y + z^2");
  await expect(page.getByLabel("z from")).toHaveValue("-1");
  await page.getByLabel("Epochs").fill("20");
  await page.getByRole("button", { name: "Train" }).click();
  await expect(page.getByTestId("run-status")).toContainText("Finished", { timeout: 60_000 });
  await expect(page.getByTestId("slice-note")).toContainText("z = 0");
  await expect(page.getByTestId("network-view")).toHaveAccessibleName(/layers of 3, 32, 32, 1 units/);
  await expect(page.getByRole("cell", { name: "RBF network LS" })).toBeVisible();
  await expect(page.getByRole("cell", { name: "Chebyshev LS" })).toHaveCount(0);
  expect(errors).toEqual([]);
});

test("a two-dimensional target draws three surfaces", async ({ page }) => {
  const errors = watchErrors(page);
  await runPreset(page, "franke");
  await expect(page.locator("canvas[role=img]:not([data-testid=network-view])")).toHaveCount(3);
  const alpha = (c: HTMLCanvasElement) => c.getContext("2d")!.getImageData(0, 0, 1, 1).data[3];
  for (const i of [0, 1, 2]) expect(await page.locator("canvas[role=img]:not([data-testid=network-view])").nth(i).evaluate(alpha)).toBe(255);
  await expect(page.getByRole("radio", { name: "Spectrum" })).toHaveCount(0);
  expect(errors).toEqual([]);
});

test("the KAN preset reads the formula back", async ({ page }) => {
  const errors = watchErrors(page);
  await runPreset(page, "kan-f7");
  await expect(page.getByRole("radio", { name: "Edges" })).toBeChecked();
  await expect(page.getByText(/R² [=>] /)).toHaveCount(3);
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

test("explorer offers the filters that matter on a file with many columns", async ({ page }) => {
  await page.goto("/explorer?data=e7/runs");
  await expect(page.getByRole("combobox", { name: "Only target" })).toHaveText("m1");
  await expect(page.getByRole("combobox", { name: "Only sigma" })).toBeVisible();
  await expect(page.getByRole("combobox", { name: "Horizontal axis" })).toHaveText("gap_pct");
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
        await page.addInitScript((t) => localStorage.setItem("funcprox-theme", t), theme);
        await page.goto(path);
        await page.waitForLoadState("networkidle");
        const { violations } = await new AxeBuilder({ page }).analyze();
        const serious = violations.filter((v) => v.impact === "serious" || v.impact === "critical");
        expect(serious.map((v) => `${v.id}: ${v.nodes.slice(0, 3).map((n) => n.target.join(" ")).join(" | ")}`)).toEqual([]);
      });
    }
  });
}

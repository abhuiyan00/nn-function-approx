// Screenshots and GIFs of the interface for the README, taken from the real engine.
//
//   npm run build && npm run capture                     every picture, about eight minutes
//   npm run capture -- --only lab                        only the scenes whose name contains "lab"
//
// The GIFs are put together by Pillow, which comes with matplotlib.
import { execFileSync, spawn } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { chromium } from "@playwright/test";

const PORT = 8796;
const BASE = `http://127.0.0.1:${PORT}`;
const UI = path.resolve(import.meta.dirname, "..");
const OUT = path.resolve(UI, "../docs/screenshots");
const only = process.argv.includes("--only") ? process.argv[process.argv.indexOf("--only") + 1] : "";
const GIF = `
import glob, sys
from PIL import Image
folder, out, width, ms = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
frames = [Image.open(f).convert("RGB") for f in sorted(glob.glob(folder + "/*.png"))]
frames = [f.resize((width, round(f.height * width / f.width)), Image.LANCZOS) for f in frames]
frames = [f.convert("P", palette=Image.ADAPTIVE, colors=96) for f in frames]
frames[0].save(out, save_all=True, append_images=frames[1:], duration=ms, loop=0, optimize=True)
`;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const server = spawn("python", ["e2e/serve.py", String(PORT)], { cwd: UI, stdio: "ignore" });
for (let i = 0; ; i++) {
  try {
    if ((await fetch(`${BASE}/api/health`)).ok) break;
  } catch {
    if (i > 60) throw new Error("the server did not start");
  }
  await sleep(500);
}

const browser = await chromium.launch();
const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
const page = await context.newPage();

const png = (name) => path.join(OUT, `${name}.png`);
const card = (locator) => locator.locator("xpath=ancestor::*[@data-slot='card'][1]");
const analysis = () => card(page.getByRole("radiogroup", { name: "Analysis" }));
const fitCard = () => card(page.getByRole("heading", { name: "Fit against the exact function" }));
const finished = (timeout = 180_000) => page.getByTestId("run-status").getByText(/Finished|Stopped/).waitFor({ timeout });
const settle = () => sleep(800); // entrance animations and chart measuring

/** Do something that starts a run, and wait until the page shows that new run. */
async function start(action) {
  const before = new URL(page.url()).searchParams.get("run");
  await action();
  await page.waitForFunction((prev) => new URL(location.href).searchParams.get("run") !== prev, before);
  await page.getByTestId("network-view").waitFor();
}
const preset = (id) => start(() => page.getByTestId(`preset-${id}`).click());

async function tall(height, take) { // a page scrolls inside the shell, so a long picture needs a tall window
  await page.setViewportSize({ width: 1440, height });
  await page.evaluate(() => document.getElementById("main").scrollTo(0, 0));
  await settle();
  await take();
  await page.setViewportSize({ width: 1440, height: 900 });
}

/** ``frames`` pictures of ``target``, one every ``every`` ms, as one GIF. */
async function gif(name, target, { frames, every, width }) {
  const folder = mkdtempSync(path.join(tmpdir(), "funcprox-gif-"));
  for (let i = 0; i < frames; i++) {
    const started = Date.now();
    await target.screenshot({ path: path.join(folder, `${String(i).padStart(3, "0")}.png`) });
    await sleep(Math.max(0, every - (Date.now() - started)));
  }
  execFileSync("python", ["-c", GIF, folder, path.join(OUT, `${name}.gif`), String(width), String(every)]);
  rmSync(folder, { recursive: true });
}

async function lab(config) {
  await page.goto(`${BASE}/lab`);
  await page.evaluate((c) => {
    localStorage.clear();
    if (c) localStorage.setItem("funcprox-lab-config", JSON.stringify(c));
  }, config);
  await page.goto(`${BASE}/lab`);
  await page.getByRole("button", { name: "Train" }).waitFor();
}

const scenes = {
  async overview() {
    await page.goto(BASE);
    await page.getByRole("heading", { level: 1 }).waitFor();
    await page.locator("svg[role=img]").first().waitFor();
    await settle();
    await page.screenshot({ path: png("overview") });
  },

  async finder() {
    await page.goto(`${BASE}/finder`);
    await page.evaluate(() => localStorage.clear());
    await page.reload();
    await page.getByTestId("preview-title").waitFor();
    await settle();
    await page.screenshot({ path: png("curve-finder-start") });

    await page.getByRole("radio", { name: "Formula" }).click();
    await page.getByLabel("f(x) =").fill("sin(3*x) * exp(-x^2)");
    await page.getByRole("switch", { name: "Hide a part of the range" }).click();
    await settle();
    await page.screenshot({ path: png("curve-finder-formula") });

    await page.getByRole("radio", { name: "Draw" }).click();
    const pad = await page.getByTestId("draw-pad").boundingBox();
    const at = (t) => [pad.x + 6 + t * (pad.width - 12), pad.y + pad.height * (0.5 - 0.32 * Math.sin(5 * t) * (1 - 0.5 * t))];
    await page.mouse.move(...at(0));
    await page.mouse.down();
    for (let t = 0; t <= 1; t += 0.01) await page.mouse.move(...at(t));
    await page.mouse.up();
    await page.getByRole("radio", { name: "Eraser" }).click(); // the erased stretch stays dashed: scored, never trained on
    await page.mouse.move(pad.x + pad.width * 0.42, pad.y + 12);
    await page.mouse.down();
    await page.mouse.move(pad.x + pad.width * 0.6, pad.y + 12, { steps: 10 });
    await page.mouse.up();
    await settle();
    await page.screenshot({ path: png("curve-finder-draw") });

    await page.evaluate(() => localStorage.clear());
    await page.reload();
    await page.getByTestId("preview-title").waitFor();
    await page.getByRole("button", { name: "Find the closest curve" }).click();
    await page.getByTestId("search-progress").waitFor();
    await gif("curve-finder-live", page, { frames: 30, every: 450, width: 900 });
    await page.getByTestId("search-status").getByText("Finished").waitFor({ timeout: 300_000 });
    await settle();
    await page.screenshot({ path: png("curve-finder-results") });
    await tall(1500, () => page.screenshot({ path: png("curve-finder-ranking") }));
  },

  async lab() {
    await lab();
    await preset("adam-runge");
    await gif("network-live", card(page.getByTestId("network-view")), { frames: 40, every: 110, width: 760 });
    await finished();

    await preset("spectral-tanh");
    await tall(1420, () => gif("training-lab-live", page, { frames: 26, every: 450, width: 900 }));
    await page.getByRole("button", { name: "Stop" }).click();
    await finished();

    await preset("lm-runge");
    await finished();
    await sleep(1800);
    await tall(2200, () => page.screenshot({ path: png("training-lab") }));
    for (const [panel, name] of [["Units", "lab-units"], ["Slope", "lab-slope"], ["Loss landscape", "lab-landscape"], ["Baselines", "lab-baselines"]]) {
      await page.getByRole("radio", { name: panel }).click();
      if (panel === "Loss landscape") await page.locator("polyline.draw-in").waitFor();
      await sleep(1700);
      await analysis().screenshot({ path: png(name) });
    }
    await page.getByRole("button", { name: "Replay the training" }).click();
    await gif("replay", fitCard(), { frames: 32, every: 150, width: 700 });
  },

  async hole() {
    // noisy samples, so the band is wide enough to see
    await lab({ target: "m4", hidden: [16, 16], optimizer: "lm", epochs: 300, patience: 30, sigma: 0.1 });
    await start(() => page.getByRole("button", { name: "Train" }).click());
    await finished();
    await sleep(1200);
    await fitCard().screenshot({ path: png("lab-trust-band") });
  },

  async kan() {
    await lab();
    await preset("kan-f7");
    await finished();
    await sleep(1200);
    await analysis().screenshot({ path: png("lab-kan-edges") });
  },

  async three() {
    await lab();
    await page.getByRole("radio", { name: "Formula" }).click();
    await page.getByRole("button", { name: "3 inputs" }).click();
    await page.getByLabel("Epochs").fill("200");
    await start(() => page.getByRole("button", { name: "Train" }).click());
    await finished();
    await sleep(1500);
    await tall(1750, () => page.screenshot({ path: png("lab-three-inputs") }));
  },

  async dark() {
    await lab();
    await page.evaluate(() => localStorage.setItem("funcprox-theme", "dark"));
    await page.reload();
    await preset("franke");
    await finished();
    await sleep(1800);
    await tall(1500, () => page.screenshot({ path: png("training-lab-dark") }));
    await page.evaluate(() => localStorage.removeItem("funcprox-theme"));
  },

  async sweep() {
    await lab({ target: "f4", hidden: [16, 16], optimizer: "adam", epochs: 300, patience: 0, n_train: 500, n_val: 100 });
    await page.goto(`${BASE}/sweep`);
    await page.getByRole("button", { name: "Run the study" }).click();
    await page.getByTestId("sweep-progress").waitFor();
    await gif("sensitivity-live", page, { frames: 26, every: 450, width: 900 });
    await page.getByTestId("sweep-progress").getByText(/^(\d+) of \1 /).waitFor({ timeout: 300_000 });
    await settle();
    await tall(1150, () => page.screenshot({ path: png("sensitivity") }));
  },

  async pages() {
    const json = { method: "POST", headers: { "Content-Type": "application/json", Origin: BASE } };
    const body = { target: "m2", hidden: [16, 16], optimizer: "lm", epochs: 120, patience: 0, n_train: 500, n_val: 100 };
    const run = await (await fetch(`${BASE}/api/lab/runs`, { ...json, body: JSON.stringify(body) })).json();
    while (!(await (await fetch(`${BASE}/api/lab/runs/${run.id}`)).json()).final) await sleep(300);
    await fetch(`${BASE}/api/lab/runs/${run.id}/save`, { ...json, body: JSON.stringify({ name: "runge-hidden-peak" }) });
    for (const [address, name, ready] of [
      ["/experiments", "experiments", () => page.getByRole("heading", { level: 1 }).waitFor()],
      ["/experiments/e6", "experiment-detail", () => page.locator("svg[role=img]").first().waitFor()],
      ["/explorer", "run-explorer", () => page.locator("svg[role=img]").first().waitFor()],
      ["/models?model=runge-hidden-peak", "saved-models", () => page.getByRole("button", { name: "Predict" }).waitFor()],
      ["/runner", "runner", () => page.getByRole("heading", { level: 1 }).waitFor()],
      ["/documents/findings", "documents", () => page.getByRole("heading", { level: 1 }).first().waitFor()],
    ]) {
      await page.goto(BASE + address);
      await ready();
      await sleep(1500);
      await page.screenshot({ path: png(name) });
    }
  },
};

try {
  for (const [name, scene] of Object.entries(scenes)) {
    if (only && !name.includes(only)) continue;
    const started = Date.now();
    await scene();
    console.log(`${name}: ${((Date.now() - started) / 1000).toFixed(0)} s`);
  }
} finally {
  await browser.close();
  server.kill();
}

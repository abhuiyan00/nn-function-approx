import { defineConfig, devices } from "@playwright/test";

const PORT = 8799;

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false, // the lab trains one run at a time on the server
  workers: 1,
  timeout: 90_000,
  reporter: [["list"]],
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    trace: "retain-on-failure",
    ...devices["Desktop Chrome"],
    viewport: { width: 1440, height: 900 },
  },
  webServer: {
    command: `python e2e/serve.py ${PORT}`,
    url: `http://127.0.0.1:${PORT}/api/health`,
    reuseExistingServer: false,
    timeout: 30_000,
  },
});

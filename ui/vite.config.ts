import path from "node:path";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

const API = "http://127.0.0.1:8765";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": path.resolve(import.meta.dirname, "src") } },
  build: { outDir: "../src/funcprox_ui/static", emptyOutDir: true, chunkSizeWarningLimit: 1500 },
  server: {
    // changeOrigin stays off: the API compares Host with Origin before accepting writes
    proxy: { "/api": { target: API }, "/figures": { target: API }, "/screenshots": { target: API } },
  },
  test: { include: ["src/**/*.test.ts"] },
});

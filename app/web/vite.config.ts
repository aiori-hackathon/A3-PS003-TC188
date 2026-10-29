import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// The Python API (app/backend/server.py) listens on 127.0.0.1:8000 and serves every route under /api.
const API = process.env.AIORI_API ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { port: 5173, proxy: { "/api": { target: API, changeOrigin: true } } },
  preview: { port: 4173, proxy: { "/api": { target: API, changeOrigin: true } } },
  build: { outDir: "dist", sourcemap: false, chunkSizeWarningLimit: 900 },
  test: { environment: "node", include: ["src/**/*.test.ts"] },
});

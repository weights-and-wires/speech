import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

const target = process.env.API_TARGET ?? "http://127.0.0.1:8000";

// Local development only. With VITE_DEV_SSO=1 the dev-server proxy signs every
// /api and /book request in as the user "dev", the way Caddy does after
// Authelia. Start the backend with SSO_PROXY_SECRET=dev-secret to match.
// Production serves the built bundle, never this server, so this never ships.
const devSso =
  process.env.VITE_DEV_SSO === "1"
    ? { headers: { "X-SSO-Proxy": "dev-secret", "Remote-User": "dev" } }
    : {};

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      // API_TARGET points the dev server at a backend on another port.
      "/api": { target, changeOrigin: true, ...devSso },
      // The PDF book, which the backend serves from book/.
      "/book": { target, changeOrigin: true, ...devSso },
    },
  },
  // Monaco ships its workers as ES modules, so the worker bundle must be one too.
  worker: { format: "es" },
  build: {
    outDir: "dist",
    // Monaco lands in its own chunk, which is large by nature.
    chunkSizeWarningLimit: 4500,
  },
});

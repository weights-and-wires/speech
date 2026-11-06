// Renders every ```figure block in the named lessons to a PNG, so you can look
// at a figure while you draw it without starting the site.
//
//   node scripts/render_figures.mjs OUT_DIR content/lessons/005-least-squares-line.md [...]
//   node scripts/render_figures.mjs OUT_DIR --dark FILE.md   # the dark theme
//
// Each figure lands at OUT_DIR/<lesson>-<n>.png with the site's own figure
// styles, light theme unless --dark. Chrome comes from CHROME_PATH or the
// usual places on macOS and Linux.

import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
const require = createRequire(path.join(root, "frontend", "package.json"));
const puppeteer = require("puppeteer-core");

const argv = process.argv.slice(2);
const dark = argv.includes("--dark");
const [outDir, ...files] = argv.filter((a) => a !== "--dark");
if (!outDir || files.length === 0) {
  console.error("usage: render_figures.mjs OUT_DIR [--dark] LESSON.md [...]");
  process.exit(2);
}

const chrome = [
  process.env.CHROME_PATH,
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  "/usr/bin/google-chrome",
  "/usr/bin/chromium",
].find((p) => p && fs.existsSync(p));
if (!chrome) {
  console.error("render_figures: no Chrome found; set CHROME_PATH");
  process.exit(2);
}

// The figure rules from the site's stylesheet, from their comment to the
// caption rule, so a PNG here looks like the figure on the page.
const css = fs.readFileSync(path.join(root, "frontend", "src", "index.css"), "utf8");
const start = css.indexOf("/* Figures:");
const end = css.indexOf("}", css.indexOf(".figure figcaption")) + 1;
const tokens = dark
  ? "--canvas:#1a1a18;--sunk:#22221f;--rule:#2e2d29;--ink:#eeece6;--ink-2:#d3d0c7;--ink-3:#a19d92;--ink-4:#75726a;--accent:#8cadff;--good:#5fc48f;--bad:#f08074;"
  : "--canvas:#fcfbf8;--sunk:#f0eee7;--rule:#e6e3da;--ink:#1c1b18;--ink-2:#37352f;--ink-3:#6b675d;--ink-4:#9a9589;--accent:#2b59c3;--good:#1b7a4a;--bad:#b3261e;";
const style = `:root{${tokens}--font-sans:Inter,Helvetica,Arial,sans-serif;--font-mono:Menlo,monospace}
body{margin:0;padding:16px;width:640px;background:var(--canvas)}
${css.slice(start, end)}`;

fs.mkdirSync(outDir, { recursive: true });
const browser = await puppeteer.launch({ executablePath: chrome, headless: true });
const page = await browser.newPage();
await page.setViewport({ width: 672, height: 400, deviceScaleFactor: 1.5 });

let count = 0;
for (const file of files) {
  const text = fs.readFileSync(file, "utf8");
  const blocks = [...text.matchAll(/^```figure\n([\s\S]*?)^```/gm)].map((m) => m[1]);
  const name = path.basename(file, ".md");
  for (const [i, block] of blocks.entries()) {
    const cut = block.lastIndexOf("</svg>") + "</svg>".length;
    const caption = block.slice(cut).trim().replace(/</g, "&lt;");
    const html = `<style>${style}</style><div${dark ? ' data-theme="dark"' : ""}><figure class="figure"><div class="figure-art">${block.slice(0, cut)}</div><figcaption>${caption}</figcaption></figure></div>`;
    await page.setContent(html);
    const box = await page.$("figure");
    const out = path.join(outDir, `${name}-${i + 1}.png`);
    await box.screenshot({ path: out });
    console.log(out);
    count += 1;
  }
}
await browser.close();
if (count === 0) console.error("render_figures: no ```figure blocks found");

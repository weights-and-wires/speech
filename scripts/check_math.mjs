// Compiles every $...$ and $$...$$ segment in the named lessons with the
// frontend's own KaTeX, so math that fails only at render time fails here.
//
//   node scripts/check_math.mjs content/lessons/01-the-learning-problem.md
//   node scripts/check_math.mjs            # every lesson

import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
const require = createRequire(path.join(root, "frontend", "package.json"));
const katex = require("katex");

let files = process.argv.slice(2);
if (files.length === 0) {
  const dir = path.join(root, "content", "lessons");
  files = fs.readdirSync(dir).filter((f) => f.endsWith(".md")).map((f) => path.join(dir, f));
}

let failures = 0;
for (const file of files) {
  let text = fs.readFileSync(file, "utf8");
  text = text.replace(/```[\s\S]*?```/g, "").replace(/`[^`\n]*`/g, "");
  const segments = [];
  text = text.replace(/\$\$([\s\S]+?)\$\$/g, (_, m) => { segments.push([m, true]); return ""; });
  text.replace(/\$([^$\n]+?)\$/g, (_, m) => { segments.push([m, false]); return ""; });
  for (const [src, display] of segments) {
    try {
      katex.renderToString(src, { displayMode: display, throwOnError: true, strict: "ignore" });
    } catch (err) {
      failures += 1;
      console.log(`${path.basename(file)}: ${err.message.split("\n")[0]}\n    ${src.trim().slice(0, 120)}`);
    }
  }
}
console.log(failures ? `${failures} math segment(s) failed to compile` : `All math compiles in ${files.length} file(s)`);
process.exit(failures ? 1 : 0);

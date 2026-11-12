// Makes `dist/image-tags/`: the Plugin as a Host loads it, and nothing else. It is the folder to pick with
// "Choose folder…" in the app, whose installer copies a folder whole and refuses one holding a link (pnpm's
// node_modules is made of them). The Python project goes in without its tests, scripts, caches or environment: the
// Host makes its own environment from `uv.lock`.
import { cp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));

/** What the Plugin needs at run time, beside a package.json of its own. */
const FILES = ["plugin.ts", "pick-build.ts", "README.md", "python/pyproject.toml", "python/uv.lock"];
/** Python packages, copied without their bytecode. */
const PACKAGES = ["python/image_tags"];

/** Makes the Plugin folder at `to`, replacing what is there. */
export async function makePluginFolder(to: string): Promise<void> {
  await rm(to, { recursive: true, force: true });
  await mkdir(join(to, "python"), { recursive: true });
  for (const name of FILES) await cp(join(root, name), join(to, name));
  for (const name of PACKAGES)
    await cp(join(root, name), join(to, name), {
      recursive: true,
      dereference: true,
      filter: (source) => !source.includes("__pycache__"),
    });
  const {
    devDependencies: _d,
    scripts: _s,
    packageManager: _p,
    engines: _e,
    ...manifest
  } = JSON.parse(await readFile(join(root, "package.json"), "utf8"));
  await writeFile(join(to, "package.json"), `${JSON.stringify(manifest, null, 2)}\n`);
}

if (import.meta.main) {
  const to = resolve(process.argv[2] ?? join(root, "dist", "image-tags"));
  await makePluginFolder(to);
  console.log(`Made the Plugin folder ${to}: pick it with "Choose folder…" on the app's Plugins page`);
}

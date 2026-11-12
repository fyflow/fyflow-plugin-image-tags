// The Plugin folder `pnpm plugin-folder` makes: everything `plugin.ts` imports of its own goes in with it.
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, expect, it } from "vitest";
import { makePluginFolder } from "../scripts/plugin-folder.ts";

const root = fileURLToPath(new URL("..", import.meta.url));

let to: string;
afterEach(() => rmSync(to, { recursive: true, force: true }));

it("holds every module of the Plugin's own that plugin.ts imports", async () => {
  to = mkdtempSync(join(tmpdir(), "image-tags-folder-"));

  await makePluginFolder(to);

  const own = [...readFileSync(join(root, "plugin.ts"), "utf8").matchAll(/from "\.\/([^"]+)"/g)].map(
    ([, m]) => m,
  );
  expect(own).toContain("pick-build.ts");
  for (const module of ["plugin.ts", ...own]) expect(existsSync(join(to, module ?? "")), module).toBe(true);
});

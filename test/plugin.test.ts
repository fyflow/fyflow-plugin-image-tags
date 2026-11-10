// The Plugin's declarations as a Host takes them: handed to fyflow's own `createRuntime`, each Python Worker standing
// in for the Host's bridge, with the checks fyflow-app adds for Python Workers. Python itself is tested in
// `python/tests`.
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { availableParallelism } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { type FlowDefinition, type RegistryDocument, validateDefinition } from "fyflow-orchestration";
import { createRuntime } from "fyflow-orchestration/scheduler";
import { CPU_GROUP, type PluginNodeType } from "fyflow-plugin-sdk";
import { describe, expect, it } from "vitest";
import plugin, { CLASSIFIERS, DEFAULT_CLASSIFIER } from "../plugin.ts";

const root = fileURLToPath(new URL("..", import.meta.url));
const project = join(root, "python");
const STAND_IN = new URL("./python-stand-in.worker.ts", import.meta.url).href;

/** A Python Worker as fyflow-app hands it to fyflow: its class swapped for the bridge, here a stand-in. */
const workers = Object.fromEntries(
  Object.entries(plugin.workers ?? {}).map(([key, worker]) => {
    const { python: _, ...rest } = worker as { python: string };
    return [key, { ...rest, url: STAND_IN }];
  }),
);

const registry = (): RegistryDocument => ({
  workers,
  nodeTypes: Object.fromEntries(
    Object.entries(plugin.nodeTypes ?? {}).map(
      ([name, { icon: _i, description: _d, profile: _p, ...declaration }]) => [
        name,
        { ...(declaration as PluginNodeType), config: { host: { dataDir: "D:/data" } } },
      ],
    ),
  ),
  // The `cpu` group its Node Types name, as the Host's built-in Plugin declares it.
  groups: { [CPU_GROUP]: { type: "ConcurrentLimitGroup", config: { limit: availableParallelism() } } },
});

const flowOf = (settings: Record<string, unknown>): FlowDefinition => ({
  flowId: "classify",
  version: 1,
  entry: "classify",
  nodes: [
    { id: "classify", kind: "task", type: "ClassifyImage", settings, edges: [{ to: "done" }] },
    { id: "done", kind: "end", outcome: "completed" },
  ],
});

const findingsOf = (flow: FlowDefinition) => {
  const { nodeTypes = {} } = registry();
  return validateDefinition(flow, { predicates: [], workers, nodeTypes });
};

describe("the Plugin's declarations", () => {
  it("is what fyflow accepts: its schemas in fyflow's subset, its Node Types on their pools", () => {
    expect(() => createRuntime(registry(), {}, { base: plugin.base })).not.toThrow();
  });

  it("names each Python Worker as module:Class, a class its project has", () => {
    for (const worker of Object.values(plugin.workers ?? {})) {
      const target = (worker as { python: string }).python;
      expect(target).toMatch(/^[A-Za-z_]\w*(\.[A-Za-z_]\w*)*:[A-Za-z_]\w*$/);
      const [module, cls] = target.split(":") as [string, string];
      const file = `${join(project, ...module.split("."))}.py`;
      expect(existsSync(file), file).toBe(true);
      expect(readFileSync(file, "utf8")).toMatch(new RegExp(`^class ${cls}\\b`, "m"));
    }
  });

  it("has a Python project the Host can make an environment of", () => {
    expect(existsSync(join(project, "pyproject.toml"))).toBe(true);
    expect(existsSync(join(project, "uv.lock"))).toBe(true);
  });

  it("runs no Python Worker inline, as fyflow-app refuses", () => {
    for (const nodeType of Object.values(plugin.nodeTypes ?? {})) expect(nodeType.inline).not.toBe(true);
  });

  it("offers the classifiers the Python catalogue has, with their labels", () => {
    const catalogue = JSON.parse(
      execFileSync(
        "uv",
        [
          "run",
          "--quiet",
          "python",
          "-c",
          "import json; from image_tags.catalogue import MODELS, DEFAULT_CLASSIFIER as d; " +
            "print(json.dumps({'default': d, 'classifiers': {k: {'title': c.title, 'labels': list(c.labels)} " +
            "for k, c in MODELS.items()}}))",
        ],
        { cwd: project, encoding: "utf8" },
      ),
    );
    expect(catalogue).toEqual({ default: DEFAULT_CLASSIFIER, classifiers: CLASSIFIERS });
  });

  it("takes a Flow that names a classifier, and refuses one the catalogue lacks", () => {
    expect(findingsOf(flowOf({ classifier: "anime_real" }))).toEqual([]);
    expect(findingsOf(flowOf({})).map((f) => f.rule)).toEqual([]);
    // FD-V33: a Node's settings break its Worker's settingsSchema.
    expect(findingsOf(flowOf({ classifier: "nope" })).map((f) => f.rule)).toContain("FD-V33");
  });
});

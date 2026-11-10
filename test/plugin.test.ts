// The Plugin's declarations as a Host takes them: handed to fyflow's own `createRuntime`, each Python Worker standing
// in for the Host's bridge, with the checks fyflow-app adds for Python Workers. Python itself is tested in
// `python/tests`.
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { availableParallelism, tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  type FlowDefinition,
  findings,
  type JsonSchema,
  type RegistryDocument,
  schemaProblem,
  validateDefinition,
} from "fyflow-orchestration";
import { createRuntime } from "fyflow-orchestration/scheduler";
import { CPU_GROUP, type PluginNodeType } from "fyflow-plugin-sdk";
import { afterEach, describe, expect, it } from "vitest";
import plugin, { CLASSIFIERS, DEFAULT_CLASSIFIER, localPathOf, MODELS } from "../plugin.ts";

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
        {
          ...(declaration as PluginNodeType),
          config: { modelDir: "D:/models", host: { dataDir: "D:/data" } },
        },
      ],
    ),
  ),
  // The `cpu` group its Node Types name, as the Host's built-in Plugin declares it.
  groups: { [CPU_GROUP]: { type: "ConcurrentLimitGroup", config: { limit: availableParallelism() } } },
});

/** The Plugin's Node Types: one image per Task, and a sheet or folder of frames per Task. */
const NODE_TYPES = ["ClassifyImage", "ClassifyFrames"] as const;

const flowOf = (settings: Record<string, unknown>, type: string = "ClassifyImage"): FlowDefinition => ({
  flowId: "classify",
  version: 1,
  entry: "classify",
  nodes: [
    { id: "classify", kind: "task", type, settings, edges: [{ to: "done" }] },
    { id: "done", kind: "end", outcome: "completed" },
  ],
});

/** The Python catalogue's classifiers and model files, as the TypeScript mirrors spell them (asked once). */
let catalogue: { default: string; classifiers: unknown; models: unknown } | undefined;
const pythonCatalogue = () => {
  catalogue ??= JSON.parse(
    execFileSync(
      "uv",
      [
        "run",
        "--quiet",
        "python",
        "-c",
        "import json; from image_tags.catalogue import CLASSIFIERS, MODELS, DEFAULT_CLASSIFIER as d; " +
          "print(json.dumps({'default': d, 'classifiers': {k: {'kind': c.kind, 'title': c.title, " +
          "'labels': list(c.labels), 'models': [m.id for m in c.models], 'minConfidence': c.min_confidence} " +
          "for k, c in CLASSIFIERS.items()}, 'models': {k: {'repo': m.repo, 'commit': m.commit, 'name': m.name, " +
          "'file': m.file, 'localPath': m.local_path} for k, m in MODELS.items()}}))",
      ],
      { cwd: project, encoding: "utf8" },
    ),
  );
  return catalogue as { default: string; classifiers: unknown; models: unknown };
};

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

  it("offers the classifiers the Python catalogue has: its presets and models, their labels and thresholds", () => {
    expect(pythonCatalogue().classifiers).toEqual(CLASSIFIERS);
    expect(pythonCatalogue().default).toBe(DEFAULT_CLASSIFIER);
    expect(Object.keys(CLASSIFIERS)).toContain(DEFAULT_CLASSIFIER);
  });

  it("knows the model files the Python catalogue pins, and keeps them where it does", () => {
    expect(pythonCatalogue().models).toEqual(
      Object.fromEntries(
        Object.entries(MODELS).map(([id, model]) => [id, { ...model, localPath: localPathOf(model) }]),
      ),
    );
  });

  it("runs presets over models it knows", () => {
    for (const { models } of Object.values(CLASSIFIERS))
      for (const model of models) expect(Object.keys(MODELS)).toContain(model);
  });

  it.each(NODE_TYPES)("takes a %s that names any classifier, and refuses one the catalogue lacks", (type) => {
    for (const classifier of Object.keys(CLASSIFIERS))
      expect(findingsOf(flowOf({ classifier }, type))).toEqual([]);
    expect(findingsOf(flowOf({}, type))).toEqual([]);
    // FD-V33: a Node's settings break its Worker's settingsSchema.
    expect(findingsOf(flowOf({ classifier: "nope" }, type)).map((f) => f.rule)).toContain("FD-V33");
  });

  it.each(NODE_TYPES)("takes a %s minConfidence from 0 to 1, and refuses anything else", (type) => {
    for (const minConfidence of [0, 0.35, 1])
      expect(findingsOf(flowOf({ classifier: "style", minConfidence }, type))).toEqual([]);
    expect(findingsOf(flowOf({ classifier: "anime_real", minConfidence: 0.9 }, type))).toEqual([]);
    for (const minConfidence of [-0.1, 1.5, "0.6", null])
      expect(findingsOf(flowOf({ minConfidence }, type)).map((f) => f.rule)).toContain("FD-V33");
  });

  it.each(NODE_TYPES)("tells, in %s's tag description, every classifier's labels and uncertain", (type) => {
    const worker = plugin.nodeTypes?.[type]?.worker ?? "";
    const schema = plugin.workers?.[worker]?.outputSchema as
      | { properties: { tag: { description: string } } }
      | undefined;
    const description = schema?.properties.tag.description ?? "";
    for (const [id, { labels }] of Object.entries(CLASSIFIERS)) {
      expect(description).toContain(id);
      for (const label of labels) expect(description).toContain(label);
    }
    expect(description).toContain("uncertain");
  });

  it("runs ClassifyFrames two Tasks at a time in one process, on a Worker of its own", () => {
    const frames = plugin.nodeTypes?.ClassifyFrames;
    expect(frames?.worker).toBe("classifyFrames");
    expect(frames?.pool).toMatchObject({ maxThreads: 1, maxConcurrentTasks: 2 });
    expect((plugin.workers?.classifyFrames as { python?: string } | undefined)?.python).toBe(
      "image_tags.worker:ClassifyFrames",
    );
  });

  it("hands on, from ClassifyFrames, the agreement and each frame's timestamp, tag and scores", () => {
    const schema = plugin.workers?.classifyFrames?.outputSchema as JsonSchema & {
      required: string[];
      examples: unknown[];
    };
    expect(schema.required).toEqual(expect.arrayContaining(["agreement", "frames", "tag", "scores"]));
    for (const example of schema.examples) expect(findings(schema, example)).toEqual([]);
    const result = {
      path: "D:/Thumbnails/clip.jpg",
      video: "D:/Videos/clip.mp4",
      classifier: "anime_real",
      labels: ["anime", "real"],
      tag: "real",
      confidence: 0.7,
      scores: { anime: 0.3, real: 0.7 },
      models: ["anime_real"],
      agreement: 0.5,
      frames: [
        { timestamp: 1.5, tag: "real", scores: { anime: 0.1, real: 0.9 } },
        { timestamp: null, tag: "anime", scores: { anime: 0.5, real: 0.5 } },
      ],
      warnings: ["timestamps drawn"],
    };
    expect(findings(schema, result)).toEqual([]);
    expect(
      findings(schema, { ...result, frames: [{ timestamp: "1.5", tag: "real", scores: {} }] }),
    ).not.toEqual([]);
  });

  it("shows each Node Type with an icon of the app's set", () => {
    const readme = readFileSync(join(root, "node_modules", "fyflow-plugin-sdk", "README.md"), "utf8");
    const line = readme
      .split("\n")
      .find((l) => l.startsWith("A Node Type's `icon` names one of the app's icons"));
    const icons = [...(line ?? "").matchAll(/`([a-z-]+)`/g)].map(([, icon]) => icon);
    expect(icons).toContain("tag");
    for (const nodeType of Object.values(plugin.nodeTypes ?? {})) expect(icons).toContain(nodeType.icon);
  });

  it("has Profile fields as fyflow-app takes them: a schema in fyflow's subset, start-up fields it declares", () => {
    const { profile, dataDir } = plugin;
    if (profile === undefined) throw new Error("no profile");
    const schema = profile.schema as JsonSchema & { properties: Record<string, JsonSchema> };
    expect(schemaProblem(schema)).toBeNull();
    const fields = Object.keys(schema.properties);
    for (const field of profile.startup ?? []) expect(fields).toContain(field);
    // A Plugin with a data directory finds it under `host` in its config, so no start-up field may be called that.
    if (dataDir === true) expect(profile.startup ?? []).not.toContain("host");
    for (const nodeType of Object.values(plugin.nodeTypes ?? {}))
      for (const field of nodeType.profile ?? []) expect(fields).toContain(field);
  });

  it("takes a model folder on a Profile, as a start-up field both Node Types read", () => {
    const schema = plugin.profile?.schema as JsonSchema;
    expect(plugin.profile?.startup).toContain("modelDir");
    for (const type of NODE_TYPES) expect(plugin.nodeTypes?.[type]?.profile).toContain("modelDir");
    expect(findings(schema, { modelDir: "D:/models" })).toEqual([]);
    expect(findings(schema, {})).toEqual([]);
    expect(findings(schema, { modelDir: 3 })).not.toEqual([]);
  });
});

describe("the Plugin's status", () => {
  let dataDir: string;
  afterEach(() => rmSync(dataDir, { recursive: true, force: true }));

  const statusOf = async () => {
    if (plugin.status === undefined) throw new Error("no status()");
    return await plugin.status({ dataDir });
  };

  it("says the models download on first use, listing the ones not there yet", async () => {
    dataDir = mkdtempSync(join(tmpdir(), "image-tags-"));

    const status = await statusOf();

    expect(status.level).toBe("info");
    expect(status.summary).toBe("Models download on first use");
    for (const id of Object.keys(MODELS)) expect(JSON.stringify(status.details)).toContain(id);
  });

  it("is ok once every model is in the data directory", async () => {
    dataDir = mkdtempSync(join(tmpdir(), "image-tags-"));
    for (const model of Object.values(MODELS)) {
      const file = join(dataDir, "models", localPathOf(model));
      mkdirSync(join(file, ".."), { recursive: true });
      writeFileSync(file, "a model");
    }

    const status = await statusOf();

    expect(status.level).toBe("ok");
    for (const id of Object.keys(MODELS)) expect(status.summary).toContain(id);
  });
});

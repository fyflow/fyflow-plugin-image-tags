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
import { CPU_GROUP, GPU_GROUP, type PluginNodeType, type PluginPool } from "fyflow-plugin-sdk";
import { afterEach, describe, expect, it } from "vitest";
import plugin, {
  CLASSIFIERS,
  CPU_UNITS,
  cpuUnits,
  DEFAULT_BATCH_SIZE,
  DEFAULT_CLASSIFIER,
  DEFAULT_TAG_BATCH_SIZE,
  DEFAULT_TAGGER,
  localPathOf,
  MODELS,
  type ModelFile,
  RESIDENT_GROUPS,
  TAGGERS,
} from "../plugin.ts";

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

const withoutVariants = ({ variants: _, ...pool }: PluginPool) => pool;

const registry = (): RegistryDocument => ({
  workers,
  nodeTypes: Object.fromEntries(
    Object.entries(plugin.nodeTypes ?? {}).map(
      ([name, { icon: _i, description: _d, profile: _p, pool, ...declaration }]) => [
        name,
        {
          ...(declaration as PluginNodeType),
          // The Host hands fyflow a pool without `variants`: the base's, with the build's over it.
          ...(pool === undefined ? {} : { pool: withoutVariants(pool) }),
          config: { modelDir: "D:/models", host: { dataDir: "D:/data" } },
        },
      ],
    ),
  ),
  // The `cpu` and `gpu` groups its Node Types hold, as the Host's built-in Plugin declares them.
  groups: {
    [CPU_GROUP]: { type: "ConcurrentLimitGroup", config: { limit: availableParallelism() } },
    [GPU_GROUP]: { type: "UnitGroup", config: { units: 1 } },
  },
});

/** The Plugin's classifying Node Types: one image per Task, and a sheet or folder of frames per Task. */
const NODE_TYPES = ["ClassifyImage", "ClassifyFrames"] as const;
/** The Plugin's tagging Node Types: one image per Task, and a sheet or folder of frames per Task. */
const TAG_NODE_TYPES = ["TagImage", "TagFrames"] as const;
/** Every Node Type: the classifying ones and the tagging ones. */
const ALL_NODE_TYPES = [...NODE_TYPES, ...TAG_NODE_TYPES] as const;

/** Every model file the Plugin downloads, as `status()` and `data()` see them: the classifiers' and the taggers'. */
const MODEL_FILES: Record<string, ModelFile> = {
  ...MODELS,
  ...Object.fromEntries(Object.entries(TAGGERS).map(([id, { model }]) => [id, model])),
};

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
let catalogue:
  | { default: string; classifiers: unknown; models: unknown; defaultTagger: string; taggers: unknown }
  | undefined;
const pythonCatalogue = () => {
  catalogue ??= JSON.parse(
    execFileSync(
      "uv",
      [
        "run",
        "--quiet",
        "--extra",
        "cpu",
        "python",
        "-c",
        "import json; from image_tags.catalogue import CLASSIFIERS, MODELS, TAGGERS, DEFAULT_TAGGER, " +
          "DEFAULT_CLASSIFIER as d; " +
          "print(json.dumps({'default': d, 'classifiers': {k: {'kind': c.kind, 'title': c.title, " +
          "'labels': list(c.labels), 'models': [m.id for m in c.models], 'minConfidence': c.min_confidence} " +
          "for k, c in CLASSIFIERS.items()}, 'models': {k: {'repo': m.repo, 'commit': m.commit, 'name': m.name, " +
          "'file': m.file, 'localPath': m.local_path} for k, m in MODELS.items()}, " +
          "'defaultTagger': DEFAULT_TAGGER, 'taggers': {k: {'title': t.title, 'model': {'repo': t.repo, " +
          "'commit': t.commit, 'name': t.name, 'file': t.file}, 'localPath': t.local_path, " +
          "'tagList': t.tag_list.file, 'thresholds': {c.name: c.threshold for c in t.categories}, " +
          "'ratings': t.rating_code is not None} for k, t in TAGGERS.items()}}))",
      ],
      { cwd: project, encoding: "utf8" },
    ),
  );
  return catalogue as NonNullable<typeof catalogue>;
};

/** The Python Worker's constants that the declarations repeat (asked once). */
let workerConstants:
  | {
      tasksAtOnce: number;
      tagTasksAtOnce: number;
      cpuUnits: number;
      defaultBatchSize: number;
      defaultTagBatchSize: number;
    }
  | undefined;
const pythonWorker = () => {
  workerConstants ??= JSON.parse(
    execFileSync(
      "uv",
      [
        "run",
        "--quiet",
        "--extra",
        "cpu",
        "python",
        "-c",
        "import json; from image_tags import worker as w; print(json.dumps({'tasksAtOnce': w.TASKS_AT_ONCE, " +
          "'tagTasksAtOnce': w.TAG_TASKS_AT_ONCE, 'cpuUnits': w.CPU_UNITS, " +
          "'defaultBatchSize': w.DEFAULT_BATCH_SIZE, 'defaultTagBatchSize': w.DEFAULT_TAG_BATCH_SIZE}))",
      ],
      { cwd: project, encoding: "utf8" },
    ),
  );
  return workerConstants as NonNullable<typeof workerConstants>;
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

  it("has a build of ONNX Runtime per variant, as conflicting extras, and none in the base", () => {
    expect(plugin.python?.variants).toEqual({ cpu: ["cpu"], cuda: ["cuda"], directml: ["directml"] });
    const { base, extras, conflicts } = JSON.parse(
      execFileSync(
        "uv",
        [
          "run",
          "--quiet",
          "--extra",
          "cpu",
          "python",
          "-c",
          "import json, tomllib; p = tomllib.load(open('pyproject.toml', 'rb')); " +
            "print(json.dumps({'base': p['project']['dependencies'], " +
            "'extras': p['project']['optional-dependencies'], 'conflicts': p['tool']['uv']['conflicts']}))",
        ],
        { cwd: project, encoding: "utf8" },
      ),
    ) as { base: string[]; extras: Record<string, string[]>; conflicts: { extra: string }[][] };
    // One environment holds one build: each installs the `onnxruntime` module.
    expect(base.filter((dependency) => dependency.startsWith("onnxruntime"))).toEqual([]);
    const builds = Object.fromEntries(Object.entries(extras).map(([extra, [build = ""]]) => [extra, build]));
    expect(builds).toEqual({
      cpu: expect.stringMatching(/^onnxruntime>=/),
      cuda: expect.stringMatching(/^onnxruntime-gpu\[cuda,cudnn\]>=/),
      directml: expect.stringMatching(/^onnxruntime-directml>=/),
    });
    expect(conflicts.map((set) => set.map(({ extra }) => extra).sort())).toEqual([
      ["cpu", "cuda", "directml"],
    ]);
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

  it("runs ClassifyImage 32 Tasks at a time in one process, as many as its Worker makes room for", () => {
    const image = plugin.nodeTypes?.ClassifyImage;
    expect(image?.worker).toBe("classifyImage");
    expect(image?.pool).toMatchObject({ maxThreads: 1, maxConcurrentTasks: 32 });
    expect(pythonWorker().tasksAtOnce).toBe(image?.pool?.maxConcurrentTasks);
  });

  it("picks its own build on each machine, one of its variants", () => {
    expect(plugin.python?.pickVariant).toBeTypeOf("function");
  });

  it.each(ALL_NODE_TYPES)(
    "holds a GPU and up to 4 processors per %s process, while it lives, and nothing per Task",
    (type) => {
      const nodeType = plugin.nodeTypes?.[type];
      expect(nodeType?.groups ?? []).toEqual([]);
      expect(nodeType?.pool).toMatchObject({
        maxThreads: 1,
        idleTimeout: 5_000,
        residentGroups: { [GPU_GROUP]: 1, [CPU_GROUP]: cpuUnits(availableParallelism()) },
      });
      // The Worker runs a forward pass on as many threads as the cpu units its pool says, else this many.
      expect(pythonWorker().cpuUnits).toBe(CPU_UNITS);
    },
  );

  it("holds 4 processors per process, or every processor of a machine with fewer", () => {
    expect([1, 2, 3, 4, 5, 64].map(cpuUnits)).toEqual([1, 2, 3, 4, 4, 4]);
    expect(cpuUnits(0)).toBe(1);
    expect(RESIDENT_GROUPS[CPU_GROUP]).toBe(cpuUnits(availableParallelism()));
  });

  it.each(ALL_NODE_TYPES)("holds no GPU per %s process on the cpu build, only its processors", (type) => {
    const pool = plugin.nodeTypes?.[type]?.pool;
    expect(pool?.variants).toEqual({ cpu: { residentGroups: { [CPU_GROUP]: RESIDENT_GROUPS[CPU_GROUP] } } });
    for (const build of Object.keys(pool?.variants ?? {}))
      expect(Object.keys(plugin.python?.variants ?? {})).toContain(build);
  });

  it("lets a Profile set the most frames in a forward pass, read when a Worker starts, 16 by default", () => {
    // fyflow's schema subset has no `default`: the description says it, and the Worker applies it.
    const schema = plugin.profile?.schema as {
      properties: Record<string, JsonSchema & { description?: string }>;
    };
    expect(schema.properties.batchSize).toMatchObject({ type: "integer", minimum: 1 });
    expect(schema.properties.batchSize?.description).toContain(`left empty, ${DEFAULT_BATCH_SIZE}`);
    expect(plugin.profile?.startup).toContain("batchSize");
    for (const type of NODE_TYPES) expect(plugin.nodeTypes?.[type]?.profile).toContain("batchSize");
    expect(pythonWorker().defaultBatchSize).toBe(DEFAULT_BATCH_SIZE);
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
    // Its Workers find what the Host gives them under `host` in their config (`dataDir`, `pool`, `variant`), so a Host
    // on 0.11 skips a Plugin with a field called that.
    expect(fields).not.toContain("host");
    expect(profile.startup ?? []).not.toContain("host");
    expect(dataDir).toBe(true);
    for (const nodeType of Object.values(plugin.nodeTypes ?? {}))
      for (const field of nodeType.profile ?? []) expect(fields).toContain(field);
  });

  it("takes the Host's Allowed roots, which only the Host puts in a Task's settings", () => {
    expect(plugin.roots).toBe(true);
    // The Host writes each Task's reach at `settings.roots`, so a Worker that names it is refused.
    for (const worker of Object.values(plugin.workers ?? {})) {
      const schema = worker.settingsSchema as { properties?: Record<string, unknown> } | undefined;
      expect(Object.keys(schema?.properties ?? {})).not.toContain("roots");
    }
  });

  it("takes a model folder on a Profile, as a start-up field both Node Types read", () => {
    const schema = plugin.profile?.schema as JsonSchema;
    expect(plugin.profile?.startup).toContain("modelDir");
    for (const type of ALL_NODE_TYPES) expect(plugin.nodeTypes?.[type]?.profile).toContain("modelDir");
    expect(findings(schema, { modelDir: "D:/models" })).toEqual([]);
    expect(findings(schema, {})).toEqual([]);
    expect(findings(schema, { modelDir: 3 })).not.toEqual([]);
  });
});

describe("TagImage", () => {
  const tagFlowOf = (settings: Record<string, unknown>) => flowOf(settings, "TagImage");
  const settingsSchema = plugin.workers?.tagImage?.settingsSchema as JsonSchema & {
    properties: Record<string, JsonSchema & { description?: string }>;
  };
  const outputSchema = plugin.workers?.tagImage?.outputSchema as JsonSchema & {
    required: string[];
    examples: unknown[];
  };

  it("offers the taggers the Python catalogue has: their files, categories, defaults and ratings", () => {
    expect(pythonCatalogue().taggers).toEqual(
      Object.fromEntries(
        Object.entries(TAGGERS).map(([id, tagger]) => [
          id,
          { ...tagger, localPath: localPathOf(tagger.model) },
        ]),
      ),
    );
    expect(pythonCatalogue().defaultTagger).toBe(DEFAULT_TAGGER);
    expect(DEFAULT_TAGGER).toBe("wd-eva02-large");
  });

  it("keeps a tagger at the top of its repo's commit, with no subfolder", () => {
    expect(localPathOf(TAGGERS["wd-eva02-large"].model)).toBe(
      "SmilingWolf/wd-eva02-large-tagger-v3/c5303bb71394/model.onnx",
    );
    expect(localPathOf(TAGGERS.pixai.model)).toBe(
      "Mexes/pixai-tagger-v1.0-onnx-fp32-fp16-int8/080077856314/model.onnx",
    );
    expect(localPathOf(TAGGERS["pixai-fp16"].model)).toBe(
      "Mexes/pixai-tagger-v1.0-onnx-fp32-fp16-int8/080077856314/model_fp16.onnx",
    );
  });

  it("offers PixAI in fp32 and fp16, with its authors' thresholds and five categories", () => {
    for (const id of ["pixai", "pixai-fp16"] as const)
      expect(TAGGERS[id].thresholds).toEqual({
        general: 0.17,
        character: 0.27,
        copyright: 0.24,
        style: 0.15,
        meta: 0.17,
      });
    const properties = settingsSchema.properties.thresholds?.properties ?? {};
    expect(Object.keys(properties)).toEqual(["general", "character", "copyright", "style", "meta"]);
  });

  it("runs on the tagImage Worker, a class of the Python project", () => {
    expect(plugin.nodeTypes?.TagImage?.worker).toBe("tagImage");
    expect((plugin.workers?.tagImage as { python?: string } | undefined)?.python).toBe(
      "image_tags.worker:TagImage",
    );
  });

  it("takes a tagger the catalogue has, and refuses one it lacks", () => {
    for (const tagger of Object.keys(TAGGERS)) expect(findingsOf(tagFlowOf({ tagger }))).toEqual([]);
    expect(findingsOf(tagFlowOf({}))).toEqual([]);
    expect(findingsOf(tagFlowOf({ tagger: "nope" })).map((f) => f.rule)).toContain("FD-V33");
    expect(findingsOf(tagFlowOf({ tagger: "style" })).map((f) => f.rule)).toContain("FD-V33");
  });

  it("takes thresholds as an open object, each category a tagger has a number from 0 to 1", () => {
    // A key no tagger has gets through to the Worker, which fails the Task naming the tagger's categories.
    for (const thresholds of [{}, { character: 0.5 }, { general: 0, character: 1 }, { copyright: 0.2 }])
      expect(findingsOf(tagFlowOf({ thresholds }))).toEqual([]);
    for (const thresholds of [{ general: 1.5 }, { general: -0.1 }, { character: "0.3" }, 0.35, [0.35]])
      expect(findingsOf(tagFlowOf({ thresholds })).map((f) => f.rule)).toContain("FD-V33");
    const properties = (settingsSchema.properties.thresholds?.properties ?? {}) as Record<string, JsonSchema>;
    for (const { thresholds } of Object.values(TAGGERS))
      for (const category of Object.keys(thresholds))
        expect(properties[category]).toMatchObject({ type: "number", minimum: 0, maximum: 1 });
  });

  it("says, in the thresholds' description, how they merge and each tagger's defaults", () => {
    const description = settingsSchema.properties.thresholds?.description ?? "";
    for (const [id, { thresholds }] of Object.entries(TAGGERS)) {
      expect(description).toContain(id);
      for (const [category, threshold] of Object.entries(thresholds))
        expect(description).toContain(`${category} ${threshold}`);
    }
    expect(description).toContain("only");
  });

  it("takes maxTags as an integer of at least 1, or none", () => {
    for (const maxTags of [1, 30]) expect(findingsOf(tagFlowOf({ maxTags }))).toEqual([]);
    for (const maxTags of [0, 1.5, "10"])
      expect(findingsOf(tagFlowOf({ maxTags })).map((f) => f.rule)).toContain("FD-V33");
  });

  it("hands on the tags over their thresholds and the rating, its examples among what it hands on", () => {
    expect(outputSchema.required).toEqual(expect.arrayContaining(["path", "tagger", "tags", "rating"]));
    expect(outputSchema.examples.length).toBeGreaterThan(0);
    for (const example of outputSchema.examples) expect(findings(outputSchema, example)).toEqual([]);
    const result = {
      path: "D:/Pictures/a.png",
      tagger: "wd-eva02-large",
      tags: [
        { name: "1girl", score: 0.98, category: "general" },
        { name: "hatsune_miku", score: 0.91, category: "character" },
      ],
      rating: {
        label: "general",
        scores: { general: 0.8, sensitive: 0.15, questionable: 0.03, explicit: 0.01 },
      },
      warnings: ["wd-eva02-large ran on the CPU"],
    };
    expect(findings(outputSchema, result)).toEqual([]);
    expect(findings(outputSchema, { ...result, rating: null })).toEqual([]);
    expect(findings(outputSchema, { ...result, tags: [{ name: "1girl", score: 0.98 }] })).not.toEqual([]);
    expect(findings(outputSchema, { ...result, rating: { label: "general" } })).not.toEqual([]);
  });

  it("runs 16 Tasks at a time in one process, as many as its Worker makes room for", () => {
    const pool = plugin.nodeTypes?.TagImage?.pool;
    expect(pool).toMatchObject({ maxThreads: 1, maxConcurrentTasks: 16 });
    expect(pythonWorker().tagTasksAtOnce).toBe(pool?.maxConcurrentTasks);
  });

  it("lets a Profile set the most images in a tagger's forward pass, apart from batchSize, 4 by default", () => {
    const schema = plugin.profile?.schema as {
      properties: Record<string, JsonSchema & { description?: string }>;
    };
    expect(schema.properties.tagBatchSize).toMatchObject({ type: "integer", minimum: 1 });
    expect(schema.properties.tagBatchSize?.description).toContain(`left empty, ${DEFAULT_TAG_BATCH_SIZE}`);
    expect(DEFAULT_TAG_BATCH_SIZE).toBe(4);
    expect(plugin.profile?.startup).toContain("tagBatchSize");
    for (const type of TAG_NODE_TYPES)
      expect(plugin.nodeTypes?.[type]?.profile).toEqual(["modelDir", "tagBatchSize"]);
    for (const type of NODE_TYPES) expect(plugin.nodeTypes?.[type]?.profile).not.toContain("tagBatchSize");
    expect(pythonWorker().defaultTagBatchSize).toBe(DEFAULT_TAG_BATCH_SIZE);
  });
});

describe("TagFrames", () => {
  const worker = plugin.workers?.tagFrames;
  const outputSchema = worker?.outputSchema as JsonSchema & { required: string[]; examples: unknown[] };

  it("runs two Tasks at a time in one process, on a Worker of its own holding the GPU as TagImage's does", () => {
    const frames = plugin.nodeTypes?.TagFrames;
    expect(frames?.worker).toBe("tagFrames");
    expect((worker as { python?: string } | undefined)?.python).toBe("image_tags.worker:TagFrames");
    expect(frames?.pool).toMatchObject({ maxThreads: 1, maxConcurrentTasks: 2 });
    // The same resident units, so it takes turns with TagImage on the `gpu` unit.
    expect(frames?.pool?.residentGroups).toEqual(plugin.nodeTypes?.TagImage?.pool?.residentGroups);
    expect(frames?.pool?.variants).toEqual(plugin.nodeTypes?.TagImage?.pool?.variants);
  });

  it("takes TagImage's settings and ClassifyFrames' input", () => {
    expect(worker?.settingsSchema).toBe(plugin.workers?.tagImage?.settingsSchema);
    expect(worker?.inputSchema).toBe(plugin.workers?.classifyFrames?.inputSchema);
    const tagFramesFlowOf = (settings: Record<string, unknown>) => flowOf(settings, "TagFrames");
    expect(findingsOf(tagFramesFlowOf({}))).toEqual([]);
    expect(findingsOf(tagFramesFlowOf({ thresholds: { character: 0.5 }, maxTags: 10 }))).toEqual([]);
    expect(findingsOf(tagFramesFlowOf({ maxTags: 0 })).map((f) => f.rule)).toContain("FD-V33");
  });

  it("hands on each frame's timestamp, tags and rating, its examples among what it hands on", () => {
    expect(outputSchema.required).toEqual(expect.arrayContaining(["path", "tagger", "frames"]));
    expect(outputSchema.examples.length).toBeGreaterThan(0);
    for (const example of outputSchema.examples) expect(findings(outputSchema, example)).toEqual([]);
    const rating = {
      label: "general",
      scores: { general: 0.8, sensitive: 0.15, questionable: 0.03, explicit: 0.01 },
    };
    const result = {
      path: "D:/Thumbnails/clip.jpg",
      video: "D:/Videos/clip.mp4",
      tagger: "wd-eva02-large",
      frames: [
        { timestamp: 1.5, tags: [{ name: "1girl", score: 0.98, category: "general" }], rating },
        { timestamp: null, tags: [], rating: null },
      ],
      warnings: ["timestamps drawn"],
    };
    expect(findings(outputSchema, result)).toEqual([]);
    expect(
      findings(outputSchema, { ...result, frames: [{ timestamp: "1.5", tags: [], rating }] }),
    ).not.toEqual([]);
    expect(findings(outputSchema, { ...result, frames: [{ timestamp: 1.5, tags: [] }] })).not.toEqual([]);
    expect(
      findings(outputSchema, { ...result, frames: [{ timestamp: 1.5, tags: [{ name: "1girl" }], rating }] }),
    ).not.toEqual([]);
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
    for (const id of Object.keys(MODEL_FILES)) expect(JSON.stringify(status.details)).toContain(id);
  });

  it("is ok once every model is in the data directory", async () => {
    dataDir = mkdtempSync(join(tmpdir(), "image-tags-"));
    for (const model of Object.values(MODEL_FILES)) {
      const file = join(dataDir, "models", localPathOf(model));
      mkdirSync(join(file, ".."), { recursive: true });
      writeFileSync(file, "a model");
    }

    const status = await statusOf();

    expect(status.level).toBe("ok");
    for (const id of Object.keys(MODEL_FILES)) expect(status.summary).toContain(id);
  });
});

describe("the Plugin's data", () => {
  let dataDir: string;
  afterEach(() => rmSync(dataDir, { recursive: true, force: true }));

  const dataOf = async () => {
    if (plugin.data === undefined) throw new Error("no data()");
    return await plugin.data({ dataDir });
  };

  /** A model's file in the data directory, as a download leaves it. */
  const downloaded = (model: ModelFile) => {
    const file = join(dataDir, "models", localPathOf(model));
    mkdirSync(join(file, ".."), { recursive: true });
    writeFileSync(file, "a model");
  };

  /** A model's folder in the data directory, at a commit the catalogue does not pin, as an earlier version left it. */
  const OLD = { ...(MODELS.anime_real as ModelFile), commit: "0123456789abcdef0123456789abcdef01234567" };

  it("has no items without a models folder", async () => {
    dataDir = mkdtempSync(join(tmpdir(), "image-tags-"));

    const data = await dataOf();

    expect(data.items).toEqual([]);
    expect(data.summary).toBe("Downloaded models");
  });

  /** The catalogue's version folders, in its order, each with the ids of the models kept in it. */
  const versionFolders = () => {
    const folders = new Map<string, string[]>();
    for (const [id, { repo, commit }] of Object.entries(MODEL_FILES)) {
      const folder = `${repo}/${commit.slice(0, 12)}`;
      folders.set(folder, [...(folders.get(folder) ?? []), id]);
    }
    return folders;
  };

  it("has an item per downloaded catalogue model version, by its ids and source, which downloads again", async () => {
    dataDir = mkdtempSync(join(tmpdir(), "image-tags-"));
    for (const model of Object.values(MODEL_FILES)) downloaded(model);

    const { items } = await dataOf();

    expect(items).toEqual(
      [...versionFolders()].map(([folder, ids]) => {
        const at = folder.lastIndexOf("/");
        return {
          path: `models/${folder}`,
          label: `${ids.join(", ")}: ${folder.slice(0, at)} at ${folder.slice(at + 1)}`,
          lost: "Downloads again when next used",
        };
      }),
    );
    expect(items[0]?.label).toBe("anime_real: deepghs/anime_real_cls at 097a6c1e9e62");
    expect(items).toContainEqual({
      path: "models/SmilingWolf/wd-eva02-large-tagger-v3/c5303bb71394",
      label: "wd-eva02-large: SmilingWolf/wd-eva02-large-tagger-v3 at c5303bb71394",
      lost: "Downloads again when next used",
    });
  });

  it("names both PixAI precisions on the one item of the commit they share", async () => {
    dataDir = mkdtempSync(join(tmpdir(), "image-tags-"));
    downloaded(TAGGERS["pixai-fp16"].model);

    const { items } = await dataOf();

    expect(items).toEqual([
      {
        path: "models/Mexes/pixai-tagger-v1.0-onnx-fp32-fp16-int8/080077856314",
        label: "pixai, pixai-fp16: Mexes/pixai-tagger-v1.0-onnx-fp32-fp16-int8 at 080077856314",
        lost: "Downloads again when next used",
      },
    ]);
  });

  it("lists only the models on disk", async () => {
    dataDir = mkdtempSync(join(tmpdir(), "image-tags-"));
    downloaded(MODELS.anime_classification as ModelFile);

    const { items } = await dataOf();

    expect(items.map(({ path }) => path)).toEqual(["models/deepghs/anime_classification/5ee62e06f5f4"]);
  });

  it("marks a commit the catalogue no longer pins as not used by this version", async () => {
    dataDir = mkdtempSync(join(tmpdir(), "image-tags-"));
    downloaded(MODELS.anime_real as ModelFile);
    downloaded(OLD);

    const { items } = await dataOf();

    expect(items).toContainEqual({
      path: "models/deepghs/anime_real_cls/0123456789ab",
      label: "deepghs/anime_real_cls at 0123456789ab: not used by this version",
      lost: "Nothing this version needs",
    });
    expect(items).toHaveLength(2);
  });

  it("has no item for a download's leftover temp file", async () => {
    dataDir = mkdtempSync(join(tmpdir(), "image-tags-"));
    const folder = join(dataDir, "models", localPathOf(MODELS.anime_real as ModelFile), "..");
    mkdirSync(folder, { recursive: true });
    writeFileSync(join(folder, ".model.onnx.abc123.part"), "half a model");

    const { items } = await dataOf();

    // The commit's folder is there, so it is listed, temp file and all; nothing names the temp file itself.
    expect(items.map(({ path }) => path)).toEqual(["models/deepghs/anime_real_cls/097a6c1e9e62"]);
  });

  it("gives every item a relative, /-separated path inside the data directory", async () => {
    dataDir = mkdtempSync(join(tmpdir(), "image-tags-"));
    for (const model of [...Object.values(MODEL_FILES), OLD]) downloaded(model);
    // Files where folders are expected are no model's.
    writeFileSync(join(dataDir, "models", "stray.txt"), "");
    writeFileSync(join(dataDir, "models", "deepghs", "stray.txt"), "");

    const { items } = await dataOf();

    expect(items).toHaveLength(versionFolders().size + 1);
    for (const { path } of items) {
      expect(path).toMatch(/^models\/[^/\\]+\/[^/\\]+\/[^/\\]+$/);
      expect(path.split("/")).not.toContain("..");
      expect(existsSync(join(dataDir, ...path.split("/")))).toBe(true);
    }
  });

  it("answers without throwing when the models folder cannot be read", async () => {
    dataDir = mkdtempSync(join(tmpdir(), "image-tags-"));
    // A file where the models folder should be: listing it fails.
    writeFileSync(join(dataDir, "models"), "");

    await expect(dataOf()).resolves.toEqual({ summary: "Downloaded models", items: [] });
  });

  it("answers without throwing for a data directory that is not there", async () => {
    dataDir = join(tmpdir(), "image-tags-missing", "nowhere");

    await expect(dataOf()).resolves.toMatchObject({ items: [] });
  });
});

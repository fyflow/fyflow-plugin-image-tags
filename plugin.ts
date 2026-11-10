// The image-tags Plugin: classifies images with small local ONNX models, in a Python Worker (ADR-0026 in fyflow-app;
// fyflow-plugin-sdk 0.9.0). What it declares is here; the work is in `python/`, a uv project the Host makes an
// environment of.
import { existsSync } from "node:fs";
import { join } from "node:path";
import { CPU_GROUP, definePlugin, type PluginStatus } from "fyflow-plugin-sdk";

/**
 * What a Node's `classifier` setting may name, as `python/image_tags/catalogue.py` has it (`CLASSIFIERS`; a test holds
 * the two together): each preset, a fixed set of labels computed by a rule from the models it runs, and each model as
 * it is. `minConfidence` is the threshold the classifier's top score must reach to be the `tag` when a Node sets none:
 * a raw model's is 0, so it never answers `uncertain` unless asked.
 */
export const CLASSIFIERS = {
  style: {
    kind: "preset",
    title: "Anime, CG or live action (anime_real and anime_classification)",
    labels: ["anime", "cg", "live_action"],
    models: ["anime_real", "anime_classification"],
    minConfidence: 0.6,
  },
  anime_real: {
    kind: "model",
    title: "Anime or real (deepghs anime_real_cls)",
    labels: ["anime", "real"],
    models: ["anime_real"],
    minConfidence: 0,
  },
  anime_classification: {
    kind: "model",
    title: "3D, bangumi, comic, illustration or not a painting (deepghs anime_classification)",
    labels: ["3d", "bangumi", "comic", "illustration", "not_painting"],
    models: ["anime_classification"],
    minConfidence: 0,
  },
} as const;

/** What a classifier answers when its top score is below its `minConfidence`: a result, never a failure. */
export const UNCERTAIN = "uncertain";

/** A catalogue model's file on Hugging Face: `<repo>/resolve/<commit>/<name>/<file>`. */
export interface ModelFile {
  repo: string;
  commit: string;
  name: string;
  file: string;
}

/**
 * The catalogue's model files, as `catalogue.py` pins them (`MODELS`; a test holds them together), for `status()`.
 * Every model a classifier runs is one of them.
 */
export const MODELS: Readonly<Record<string, ModelFile>> = {
  anime_real: {
    repo: "deepghs/anime_real_cls",
    commit: "097a6c1e9e62866a435810c883a45a7cb2c7077d",
    name: "mobilenetv3_v1.4_dist",
    file: "model.onnx",
  },
  anime_classification: {
    repo: "deepghs/anime_classification",
    commit: "5ee62e06f5f4cd68a1c2f3bc5dc9805e827d37df",
    name: "mobilenetv3_v1.3_dist",
    file: "model.onnx",
  },
};

/** Where a model's file is kept below the models' folder: its commit cut to 12 characters, as `Model.local_path`. */
export const localPathOf = ({ repo, commit, name, file }: ModelFile): string =>
  `${repo}/${commit.slice(0, 12)}/${name}/${file}`;

/**
 * Which models are in the data directory. It sees only the data directory: `status()` is handed no Profile, so a
 * model folder set on a Profile (`modelDir`) is not looked at.
 */
function modelsStatus(dataDir: string | undefined): PluginStatus {
  const ids = Object.keys(MODELS);
  const missing = ids.filter(
    (id) =>
      dataDir === undefined || !existsSync(join(dataDir, "models", localPathOf(MODELS[id] as ModelFile))),
  );
  if (missing.length === 0) return { level: "ok", summary: `Models downloaded: ${ids.join(", ")}` };
  return {
    level: "info",
    summary: "Models download on first use",
    details: [
      { label: "Not downloaded yet", value: missing.join(", ") },
      ...(dataDir === undefined ? [] : [{ label: "Models folder", value: join(dataDir, "models") }]),
    ],
  };
}

export const DEFAULT_CLASSIFIER = "style";

const classifierIds = Object.keys(CLASSIFIERS);
const labelsLine = Object.entries(CLASSIFIERS)
  .map(([id, { labels }]) => `${id}: ${labels.join(", ")}`)
  .join("; ");

const settingsSchema = {
  type: "object",
  properties: {
    classifier: {
      enum: classifierIds,
      description: `The classifier to run; left empty, ${DEFAULT_CLASSIFIER}. ${Object.entries(CLASSIFIERS)
        .map(([id, { title }]) => `${id}: ${title}`)
        .join("; ")}`,
    },
    minConfidence: {
      type: "number",
      minimum: 0,
      maximum: 1,
      description:
        `The score the top label must reach to be the tag, else the tag is "${UNCERTAIN}"; left empty, the ` +
        `classifier's own: ${Object.entries(CLASSIFIERS)
          .map(([id, { minConfidence }]) => `${id} ${minConfidence}`)
          .join(", ")}`,
    },
  },
  additionalProperties: false,
};

// The Profile's fields. `modelDir` is a start-up field: the Worker reads it once, when its process starts.
const profileSchema = {
  type: "object",
  properties: {
    modelDir: {
      type: "string",
      description:
        "For a machine without internet: an absolute folder holding the models, laid out as the data directory's " +
        "`models` folder (<repo>/<commit, 12 characters>/<name>/model.onnx; see the README). When set, nothing is " +
        "downloaded, and a Task whose model is not there fails. Takes effect at the next Reload.",
    },
  },
  additionalProperties: false,
};

// A made-up path for the schemas' examples, which the app's editor shows beside each Node's fields.
const EXAMPLE_IMAGE = "D:/Pictures/frame.jpg";

// Fields it doesn't list are left alone, so a ListFolder entry or a VideoThumbnail result fits as it is.
const inputSchema = {
  type: "object",
  description: "The image at `path`: anything Pillow opens (JPEG, PNG, WebP, BMP, …)",
  required: ["path"],
  properties: {
    path: { type: "string", minLength: 1, description: "The image, an absolute path" },
  },
  examples: [{ path: EXAMPLE_IMAGE }],
};

const outputSchema = {
  type: "object",
  required: ["path", "classifier", "labels", "tag", "confidence", "scores", "models"],
  properties: {
    path: { type: "string", description: "the image's absolute path" },
    classifier: { type: "string", description: "the classifier that ran" },
    labels: {
      type: "array",
      items: { type: "string" },
      description: "the classifier's labels, which `tag` is one of",
    },
    tag: {
      type: "string",
      description:
        `the label with the highest score, or "${UNCERTAIN}" when that score is below minConfidence. ` +
        `By classifier: ${labelsLine}`,
    },
    confidence: {
      type: "number",
      description: "the top label's score, from 0 to 1, even when the tag is uncertain",
    },
    // Keyed by the classifier's labels, so the schema can only say "an object": fyflow's subset takes no schema for
    // `additionalProperties`.
    scores: { type: "object", description: "every label's score, from 0 to 1, by label" },
    models: {
      type: "array",
      items: { type: "string" },
      description: "the catalogue models the classifier ran",
    },
    warnings: {
      type: "array",
      items: { type: "string" },
      description: "what may make the answer less sure, when anything does",
    },
  },
  examples: [
    {
      path: EXAMPLE_IMAGE,
      classifier: "style",
      labels: ["anime", "cg", "live_action"],
      tag: "anime",
      confidence: 0.93,
      scores: { anime: 0.93, cg: 0.01, live_action: 0.06 },
      models: ["anime_real", "anime_classification"],
    },
    {
      path: "D:/Pictures/street.jpg",
      classifier: "style",
      labels: ["anime", "cg", "live_action"],
      tag: UNCERTAIN,
      confidence: 0.55,
      scores: { anime: 0.4, cg: 0.05, live_action: 0.55 },
      models: ["anime_real", "anime_classification"],
    },
  ],
};

export default definePlugin({
  id: "fyflow-ext.image-tags",
  version: "0.1.0",
  base: import.meta.url,
  // The Host makes the Plugin's Python environment from `python/uv.lock` with uv, at start and at each Reload.
  python: { project: "./python" },
  // Models are downloaded into it on first use: `<dataDir>/models/<repo>/<commit, 12 characters>/<name>/`.
  dataDir: true,
  profile: { schema: profileSchema, startup: ["modelDir"] },
  // Asked every few seconds while the Plugins page is open: a check that each model's file exists, nothing slower.
  status: ({ dataDir }) => modelsStatus(dataDir),
  workers: {
    classifyImage: {
      python: "image_tags.worker:ClassifyImage",
      settingsSchema,
      inputSchema,
      outputSchema,
    },
  },
  nodeTypes: {
    ClassifyImage: {
      worker: "classifyImage",
      profile: ["modelDir"],
      // One Python process per Profile, holding each model's session once; up to four Tasks at once on threads of
      // its own, sharing those sessions. Stopped after 60 s with no Task, which frees the models' memory.
      pool: { maxThreads: 1, maxConcurrentTasks: 4, idleTimeout: 60_000 },
      // A forward pass keeps a processor busy.
      groups: [CPU_GROUP],
      icon: "tag",
      description: "Tells what one image is (anime, CG or live action, …) with local classifiers.",
    },
  },
});

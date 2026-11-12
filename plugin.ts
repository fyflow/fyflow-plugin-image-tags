// The image-tags Plugin: classifies images with small local ONNX models, in a Python Worker (ADR-0026 in fyflow-app;
// fyflow-plugin-sdk 0.11.0). What it declares is here; the work is in `python/`, a uv project the Host makes an
// environment of.
import { existsSync } from "node:fs";
import { join } from "node:path";
import { CPU_GROUP, definePlugin, GPU_GROUP, type PluginStatus } from "fyflow-plugin-sdk";
import { pickBuild } from "./pick-build.ts";

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
    minConfidence: 0.55,
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
    name: "mobilenetv3_v1.5_dist",
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

/** The most frames in one forward pass when a Profile sets no `batchSize` (`DEFAULT_BATCH_SIZE` in `worker.py`). */
export const DEFAULT_BATCH_SIZE = 16;

/**
 * What each process holds while it lives, not each Task: Tasks waiting on a shared forward pass use no processor, and
 * a pass runs on as many threads as the `cpu` units held (the Worker reads them from `host.pool`, else `CPU_UNITS` in
 * `worker.py`). A Profile may set other counts.
 */
export const RESIDENT_GROUPS = { [GPU_GROUP]: 1, [CPU_GROUP]: 4 };

/** What the `cpu` build holds instead: the same processors, and no `gpu` unit, which it would not use. */
export const CPU_RESIDENT_GROUPS = { [CPU_GROUP]: RESIDENT_GROUPS[CPU_GROUP] };

/** Each pool's holds by build: a build's `residentGroups` replaces the base's whole. */
const POOL_VARIANTS = { cpu: { residentGroups: CPU_RESIDENT_GROUPS } };

// The Profile's fields. `modelDir` and `batchSize` are start-up fields: the Worker reads them once, when its process
// starts.
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
    batchSize: {
      type: "integer",
      minimum: 1,
      description:
        `The most frames in one forward pass; left empty, ${DEFAULT_BATCH_SIZE}. For a GPU, where the best size ` +
        "depends on its memory. ClassifyImage batches the images of the Tasks running at once, so a higher size " +
        "only helps with a maxConcurrentTasks at least as high. Takes effect at the next Reload.",
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

// What both Node Types hand on, as schema properties; ClassifyFrames adds its own.
const resultProperties = {
  path: { type: "string", description: "the image's absolute path" },
  video: {
    type: "string",
    description:
      "the video a frame sheet was made of, as its maker wrote it, when the image is a frame sheet",
  },
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
};

const resultRequired = ["path", "classifier", "labels", "tag", "confidence", "scores", "models"];

const STYLE_RESULT = {
  classifier: "style",
  labels: ["anime", "cg", "live_action"],
  models: ["anime_real", "anime_classification"],
};

const outputSchema = {
  type: "object",
  required: resultRequired,
  properties: resultProperties,
  examples: [
    {
      path: EXAMPLE_IMAGE,
      ...STYLE_RESULT,
      tag: "anime",
      confidence: 0.93,
      scores: { anime: 0.93, cg: 0.01, live_action: 0.06 },
    },
    {
      path: "D:/Pictures/street.jpg",
      ...STYLE_RESULT,
      tag: UNCERTAIN,
      confidence: 0.5,
      scores: { anime: 0.45, cg: 0.05, live_action: 0.5 },
    },
  ],
};

// A made-up frame sheet, as VideoThumbnail hands one on, for ClassifyFrames' examples.
const EXAMPLE_SHEET = "D:/Thumbnails/episode-01.3f9a1c2b7d4e.jpg";

const framesInputSchema = {
  type: "object",
  description:
    "The frames at `path`: a frame sheet made by VideoThumbnail (its frames are cut out by the places it wrote in " +
    "it), a folder of images (each image directly inside it, in name order, at most 1000), or any other image as " +
    "one frame",
  required: ["path"],
  properties: {
    path: {
      type: "string",
      minLength: 1,
      description: "A frame sheet, a folder of images or an image, an absolute path",
    },
  },
  examples: [{ path: EXAMPLE_SHEET }, { path: "D:/Frames/episode-01" }],
};

const framesOutputSchema = {
  type: "object",
  required: [...resultRequired, "agreement", "frames"],
  properties: {
    ...resultProperties,
    path: { type: "string", description: "the frame sheet's, folder's or image's absolute path" },
    tag: {
      ...resultProperties.tag,
      description: `over all the frames: ${resultProperties.tag.description}`,
    },
    confidence: {
      type: "number",
      description: "the top label's mean score over the frames, from 0 to 1, even when the tag is uncertain",
    },
    scores: {
      type: "object",
      description: "every label's mean score over the frames, from 0 to 1, by label",
    },
    agreement: {
      type: "number",
      description:
        "the share of frames whose own top label is the tag (the top label, when the tag is uncertain), from 0 to 1",
    },
    frames: {
      type: "array",
      description: "each frame's own answer, in the sheet's or folder's order",
      items: {
        type: "object",
        required: ["timestamp", "tag", "scores"],
        properties: {
          timestamp: {
            type: ["number", "null"],
            description: "where in the video the frame is, in seconds; null for a folder's frame or an image",
          },
          tag: { type: "string", description: "the frame's top label, or uncertain below minConfidence" },
          scores: { type: "object", description: "the frame's score for every label, by label" },
        },
      },
    },
  },
  examples: [
    {
      path: EXAMPLE_SHEET,
      video: "D:/Videos/episode-01.mkv",
      ...STYLE_RESULT,
      tag: "anime",
      confidence: 0.88,
      scores: { anime: 0.88, cg: 0.04, live_action: 0.08 },
      agreement: 1,
      frames: [
        { timestamp: 312.4, tag: "anime", scores: { anime: 0.91, cg: 0.03, live_action: 0.06 } },
        { timestamp: 624.8, tag: "anime", scores: { anime: 0.85, cg: 0.05, live_action: 0.1 } },
      ],
    },
  ],
};

export default definePlugin({
  id: "fyflow-ext.image-tags",
  version: "0.1.0",
  base: import.meta.url,
  // The Host makes the Plugin's Python environment from `python/uv.lock` with uv, at start and at each Reload, with
  // the extras of the variant `pickVariant` picks for this machine (`pick-build.ts`), and of each variant a Profile
  // sets instead: each a build of ONNX Runtime (`pyproject.toml`), whose sessions run on the GPU the process holds when
  // it has a GPU provider.
  python: {
    project: "./python",
    variants: { cpu: ["cpu"], cuda: ["cuda"], directml: ["directml"] },
    pickVariant: ({ platform }) => pickBuild(platform),
  },
  // Models are downloaded into it on first use: `<dataDir>/models/<repo>/<commit, 12 characters>/<name>/`.
  dataDir: true,
  // The Host's Allowed roots (ADR-0030 in fyflow-app): each Task carries the folders its Profile lets it read, at
  // `settings.roots`, and the Workers refuse every path outside them before reading it. So `settingsSchema` names no
  // `roots`.
  roots: true,
  profile: { schema: profileSchema, startup: ["modelDir", "batchSize"] },
  // Asked every few seconds while the Plugins page is open: a check that each model's file exists, nothing slower.
  status: ({ dataDir }) => modelsStatus(dataDir),
  workers: {
    classifyImage: {
      python: "image_tags.worker:ClassifyImage",
      settingsSchema,
      inputSchema,
      outputSchema,
    },
    classifyFrames: {
      python: "image_tags.worker:ClassifyFrames",
      settingsSchema,
      inputSchema: framesInputSchema,
      outputSchema: framesOutputSchema,
    },
  },
  nodeTypes: {
    ClassifyImage: {
      worker: "classifyImage",
      profile: ["modelDir", "batchSize"],
      // One Python process per Profile, holding each model's session once and its resident units; up to 32 Tasks at
      // once on its event loop (`TASKS_AT_ONCE` in `worker.py`), whose images go through each model together, up to
      // `batchSize` to a forward pass, a batch waiting at most 20 ms for more. Stopped after 5 s with no Task, which
      // frees the models' memory and the units, so another pool can have the GPU soon.
      pool: {
        maxThreads: 1,
        maxConcurrentTasks: 32,
        idleTimeout: 5_000,
        residentGroups: RESIDENT_GROUPS,
        variants: POOL_VARIANTS,
      },
      icon: "tag",
      description: "Tells what one image is (anime, CG or live action, …) with local classifiers.",
    },
    ClassifyFrames: {
      worker: "classifyFrames",
      profile: ["modelDir", "batchSize"],
      // One Python process per Profile, holding its resident units; two Tasks at once, so that one Task's decoding
      // and cutting out of frames overlaps the other's forward passes. A Task runs its frames through the models
      // `batchSize` at a time.
      pool: {
        maxThreads: 1,
        maxConcurrentTasks: 2,
        idleTimeout: 5_000,
        residentGroups: RESIDENT_GROUPS,
        variants: POOL_VARIANTS,
      },
      icon: "image",
      description:
        "Tells what a whole video is (anime, CG or live action, …) from its VideoThumbnail frame sheet, or a folder of frames.",
    },
  },
});

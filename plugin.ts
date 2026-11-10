// The image-tags Plugin: classifies images with small local ONNX models, in a Python Worker (ADR-0026 in fyflow-app;
// fyflow-plugin-sdk 0.9.0). What it declares is here; the work is in `python/`, a uv project the Host makes an
// environment of.
import { CPU_GROUP, definePlugin } from "fyflow-plugin-sdk";

/** The catalogue's classifiers, as `python/image_tags/catalogue.py` has them (a test holds the two together). */
export const CLASSIFIERS = {
  anime_real: { title: "Anime or real (deepghs anime_real_cls)", labels: ["anime", "real"] },
} as const;

export const DEFAULT_CLASSIFIER = "anime_real";

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
    tag: { type: "string", description: `the label with the highest score. By classifier: ${labelsLine}` },
    confidence: { type: "number", description: "the tag's score, from 0 to 1" },
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
      classifier: "anime_real",
      labels: ["anime", "real"],
      tag: "anime",
      confidence: 0.94,
      scores: { anime: 0.94, real: 0.06 },
      models: ["anime_real"],
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
      // One Python process per Profile, holding each model's session once; up to four Tasks at once on threads of
      // its own, sharing those sessions. Stopped after 60 s with no Task, which frees the models' memory.
      pool: { maxThreads: 1, maxConcurrentTasks: 4, idleTimeout: 60_000 },
      // A forward pass keeps a processor busy.
      groups: [CPU_GROUP],
      icon: "scan-eye",
      description: "Tells what one image is (anime or real, …) with a local classifier.",
    },
  },
});

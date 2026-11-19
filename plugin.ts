// The image-tags Plugin: classifies images with small local ONNX models, and tags them with larger ones, in Python
// Workers (ADR-0026 in fyflow-app; fyflow-plugin-sdk 0.11.0, and 0.12.0's `data()` for a Host that asks). What it
// declares is here; the work is in `python/`, a uv project the Host makes an environment of.
import { existsSync, readdirSync } from "node:fs";
import { availableParallelism } from "node:os";
import { join } from "node:path";
import {
  CPU_GROUP,
  definePlugin,
  GPU_GROUP,
  type PluginData,
  type PluginDataItem,
  type PluginStatus,
} from "fyflow-plugin-sdk";
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

/** A catalogue model's file on Hugging Face: `<repo>/resolve/<commit>/<name>/<file>`, or `<repo>/resolve/<commit>/<file>`
 * when its `name` is empty (a file at the top of the repo, as the taggers' are). */
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

/**
 * What a Node's `tagger` setting may name, as `catalogue.py` has it (`TAGGERS`; a test holds the two together): each
 * tagger's model file, beside which its tag list is kept; the default threshold of each category of tags it hands on;
 * and whether it rates an image.
 */
export const TAGGERS = {
  "wd-eva02-large": {
    title: "WD EVA02-large tagger v3 (SmilingWolf): general tags, characters and a rating",
    model: {
      repo: "SmilingWolf/wd-eva02-large-tagger-v3",
      commit: "c5303bb7139430db980e4c680a778fe79d72b541",
      name: "",
      file: "model.onnx",
    },
    tagList: "selected_tags.csv",
    thresholds: { general: 0.35, character: 0.85 },
    ratings: true,
  },
  pixai: {
    title:
      "PixAI tagger v1.0 (PixAI; ONNX by Mexes): general tags, characters, copyrights, styles, meta and a rating",
    model: {
      repo: "Mexes/pixai-tagger-v1.0-onnx-fp32-fp16-int8",
      commit: "0800778563144a0e6fdf41ddadd84aae3cb0dbcf",
      name: "",
      file: "model.onnx",
    },
    tagList: "selected_tags.csv",
    thresholds: { general: 0.17, character: 0.27, copyright: 0.24, style: 0.15, meta: 0.17 },
    ratings: true,
  },
  "pixai-fp16": {
    title: "PixAI tagger v1.0 in fp16 (PixAI; ONNX by Mexes): half the size, for a GPU",
    model: {
      repo: "Mexes/pixai-tagger-v1.0-onnx-fp32-fp16-int8",
      commit: "0800778563144a0e6fdf41ddadd84aae3cb0dbcf",
      name: "",
      file: "model_fp16.onnx",
    },
    tagList: "selected_tags.csv",
    thresholds: { general: 0.17, character: 0.27, copyright: 0.24, style: 0.15, meta: 0.17 },
    ratings: true,
  },
} as const satisfies Record<
  string,
  { title: string; model: ModelFile; tagList: string; thresholds: Record<string, number>; ratings: boolean }
>;

export const DEFAULT_TAGGER = "wd-eva02-large";

/** Every model file the Plugin downloads, by catalogue id: the classifiers' and the taggers'. */
const MODEL_FILES: Readonly<Record<string, ModelFile>> = {
  ...MODELS,
  ...Object.fromEntries(Object.entries(TAGGERS).map(([id, { model }]) => [id, model])),
};

/** A model version's folder below the models' folder, `<repo>/<commit, 12 characters>`. */
const versionFolderOf = ({ repo, commit }: ModelFile): string => `${repo}/${commit.slice(0, 12)}`;

/** Where a model's file is kept below the models' folder: its commit cut to 12 characters, as `Pinned.local_path`. */
export const localPathOf = (model: ModelFile): string =>
  [versionFolderOf(model), ...(model.name ? [model.name] : []), model.file].join("/");

/** Every file the Plugin downloads, by what `status()` calls it: each model's, then each tagger's tag list. */
const DOWNLOADED_FILES: ReadonlyArray<readonly [string, ModelFile]> = [
  ...Object.entries(MODEL_FILES),
  ...Object.entries(TAGGERS).map(
    ([id, { model, tagList }]) => [`${id}'s ${tagList}`, { ...model, file: tagList }] as const,
  ),
];

/**
 * Which models, and taggers' tag lists, are in the data directory. It sees only the data directory: `status()` is
 * handed no Profile, so a model folder set on a Profile (`modelDir`) is not looked at.
 */
function modelsStatus(dataDir: string | undefined): PluginStatus {
  const ids = Object.keys(MODEL_FILES);
  const missing = DOWNLOADED_FILES.filter(
    ([, file]) => dataDir === undefined || !existsSync(join(dataDir, "models", localPathOf(file))),
  ).map(([what]) => what);
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

/** The folders directly in `folder`, by name; none when it cannot be read. Links are not followed. */
function foldersIn(folder: string): string[] {
  try {
    return readdirSync(folder, { withFileTypes: true })
      .filter((entry) => entry.isDirectory())
      .map((entry) => entry.name)
      .sort();
  } catch {
    return [];
  }
}

/**
 * What the data directory holds, for the Host's Data link: an item per model version downloaded, each
 * `models/<repo>/<commit, 12 characters>` folder there is. A catalogue model's downloads again when next used; one at
 * a commit the catalogue no longer pins was left by an earlier version, and this one never reads it. A download's temp
 * file (`.<file>.*.part`, left by one that was interrupted) is in its version's folder, so it has no item of its own:
 * it goes with that version. Only directory listings, and nothing throws: a folder that cannot be read has no items.
 */
/** The Plugin's own folder for the frames' answers in its data directory (`RESULTS_FOLDER` in `results.py`). */
export const RESULTS_FOLDER = "results";

function modelsData(dataDir: string): PluginData {
  // By version folder, the ids of the catalogue models kept in it: two precisions of one tagger share their commit's.
  const pinned = new Map<string, string[]>();
  for (const [id, model] of Object.entries(MODEL_FILES))
    pinned.set(versionFolderOf(model), [...(pinned.get(versionFolderOf(model)) ?? []), id]);
  const models = join(dataDir, "models");
  const found = foldersIn(models).flatMap((owner) =>
    foldersIn(join(models, owner)).flatMap((repo) =>
      foldersIn(join(models, owner, repo)).map((commit) => `${owner}/${repo}/${commit}`),
    ),
  );
  const itemOf = (folder: string): PluginDataItem => {
    const at = folder.lastIndexOf("/");
    const source = `${folder.slice(0, at)} at ${folder.slice(at + 1)}`;
    const ids = pinned.get(folder);
    return ids === undefined
      ? {
          path: `models/${folder}`,
          label: `${source}: not used by this version`,
          lost: "Nothing this version needs",
        }
      : {
          path: `models/${folder}`,
          label: `${ids.join(", ")}: ${source}`,
          lost: "Downloads again when next used",
        };
  };
  // The catalogue's models first, in its order, then what earlier versions left.
  const order = [...pinned.keys()];
  const rank = (folder: string) => (order.includes(folder) ? order.indexOf(folder) : order.length);
  // The answers ClassifyFrames, TagFrames and TagScenes wrote with `lists: "file"` and no folder of their own.
  const answers: PluginDataItem[] = existsSync(join(dataDir, RESULTS_FOLDER))
    ? [
        {
          path: RESULTS_FOLDER,
          label: "Frames' answers written by ClassifyFrames, TagFrames and TagScenes",
          lost: "A Task runs its models again; a Flow that reads a file from here can no longer",
        },
      ]
    : [];
  return {
    summary: answers.length > 0 ? "Downloaded models and frames' answers" : "Downloaded models",
    items: [...found.sort((a, b) => rank(a) - rank(b)).map(itemOf), ...answers],
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

/** The most images in a tagger's forward pass when a Profile sets no `tagBatchSize` (`DEFAULT_TAG_BATCH_SIZE`). */
export const DEFAULT_TAG_BATCH_SIZE = 4;

/** The most `cpu` units a process holds, and so the threads of a forward pass (`CPU_UNITS` in `worker.py`). */
export const CPU_UNITS = 4;

/**
 * The `cpu` units a process holds on a machine of this many logical processors: `CPU_UNITS`, or every processor where
 * there are fewer. The Host's `cpu` group has a unit per logical processor and refuses a Plugin whose pool holds more,
 * so a fixed 4 would keep the Plugin off a 2-processor machine.
 */
export const cpuUnits = (processors: number): number => Math.max(1, Math.min(CPU_UNITS, processors));

/**
 * What each process holds while it lives, not each Task: Tasks waiting on a shared forward pass use no processor, and
 * a pass runs on as many threads as the `cpu` units held (the Worker reads them from `host.pool`, else `CPU_UNITS` in
 * `worker.py`). A Profile may set other counts.
 */
export const RESIDENT_GROUPS = { [GPU_GROUP]: 1, [CPU_GROUP]: cpuUnits(availableParallelism()) };

/** What the `cpu` build holds instead: the same processors, and no `gpu` unit, which it would not use. */
export const CPU_RESIDENT_GROUPS = { [CPU_GROUP]: RESIDENT_GROUPS[CPU_GROUP] };

/** Each pool's holds by build: a build's `residentGroups` replaces the base's whole. */
const POOL_VARIANTS = { cpu: { residentGroups: CPU_RESIDENT_GROUPS } };

// The Profile's fields. `modelDir`, `batchSize` and `tagBatchSize` are start-up fields: the Worker reads them once,
// when its process starts.
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
    tagBatchSize: {
      type: "integer",
      minimum: 1,
      description:
        `The most images in one forward pass of a tagger; left empty, ${DEFAULT_TAG_BATCH_SIZE}. A tagger is over ` +
        "1 GB, and a batch needs GPU memory on top of it: lower it if a Task fails for want of GPU memory. TagImage " +
        "batches the images of the Tasks running at once; TagFrames and TagScenes run a Task's frames that many at a " +
        "time. " +
        "Takes effect at the next Reload.",
    },
    outputDir: {
      type: "string",
      description:
        "Where ClassifyFrames, TagFrames and TagScenes write their answers when `lists` is file and the Node names no " +
        "`output`: an absolute folder in an Allowed root with write access; left empty, the Plugin's own results " +
        "folder in its data directory. Read as each Task is dispatched, so a Flow or Node may set its own.",
    },
  },
  additionalProperties: false,
};

// What ClassifyFrames, TagFrames and TagScenes take beside their image Node Type's settings: where each frame's answer goes
// (`results.py`).
const fileSettingsProperties = {
  lists: {
    enum: ["result", "file"],
    description:
      "Where each frame's answer goes; left empty, result: in the Task's result. file: in a JSON file holding the " +
      "whole result, so a large one does not sit in the Flow; the Task then hands on the rest, with `file`, `count` " +
      "and `reused`. A file made before for the same frames (unchanged), settings and models is handed on again " +
      "without running a model.",
  },
  output: {
    type: "string",
    minLength: 1,
    description:
      "With lists file: a .json file to write (every time), or a folder to name one in; an absolute path in an " +
      "Allowed root with write access. Left empty, the Profile's outputDir, else the Plugin's own results folder.",
  },
  overwrite: {
    type: "boolean",
    description:
      "With lists file: run the models again even when a file made before for the same frames is there",
  },
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

// Exactly one of `path` and `frames`: the schema subset has no `oneOf`, so the Worker refuses neither or both.
const framesInputSchema = {
  type: "object",
  description:
    "The frames at `path`: a frame sheet made by VideoThumbnail (its frames are cut out by the places it wrote in " +
    "it), a folder of images (each image directly inside it, in name order, at most 10000), or any other image as " +
    "one frame; or the images `frames` lists, in its order. Give one of `path` and `frames`. A thumbnail made by " +
    "VideoThumbnail (a JPEG of one frame) gives its frame the timestamp and video it was written with",
  properties: {
    path: {
      type: "string",
      minLength: 1,
      description: "A frame sheet, a folder of images or an image, an absolute path",
    },
    frames: {
      type: "array",
      minItems: 1,
      // The subset has no `maxItems`: the Worker refuses more than 10000.
      description:
        "The frames, each an image (a frame sheet too is read as one image), in this order, at most 10000",
      items: {
        type: "object",
        required: ["path"],
        properties: {
          path: { type: "string", minLength: 1, description: "The frame's image, an absolute path" },
          timestamp: {
            type: ["number", "null"],
            minimum: 0,
            description:
              "Where in the video the frame is, in seconds; left out or null, a thumbnail's own timestamp, if any",
          },
        },
      },
    },
  },
  examples: [
    { path: EXAMPLE_SHEET },
    { path: "D:/Frames/episode-01" },
    {
      frames: [
        { path: "D:/Frames/episode-01/0001.jpg", timestamp: 312.4 },
        { path: "D:/Frames/episode-01/0002.jpg", timestamp: 624.8 },
      ],
    },
  ],
};

// What a frames Task hands on with `lists: "file"`, instead of `frames`.
const fileResultProperties = {
  file: {
    type: "string",
    description: "with lists file: the JSON file holding the whole result, `frames` included",
  },
  count: { type: "integer", description: "with lists file: how many frames the file has an answer for" },
  reused: {
    type: "boolean",
    description:
      "with lists file: whether the file was made before, for the same frames and settings, and no model ran",
  },
};

// Each frame's own image, for a folder's or a list's frame.
const framePathProperty = {
  type: "string",
  description:
    "the frame's own image, for a folder's frame (the folder joined with its file name) or a listed one",
};

const framesOutputSchema = {
  type: "object",
  required: [...resultRequired.filter((key) => key !== "path"), "agreement"],
  properties: {
    ...resultProperties,
    path: {
      type: "string",
      description: "the frame sheet's, folder's or image's absolute path, as given; none for a `frames` list",
    },
    video: {
      type: "string",
      description:
        "the video a frame sheet was made of, or the one video the frames' thumbnails name, as its maker wrote it",
    },
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
    ...fileResultProperties,
    frames: {
      type: "array",
      description:
        "each frame's own answer, in the sheet's, folder's or list's order; in `file` instead with lists file",
      items: {
        type: "object",
        required: ["timestamp", "tag", "scores"],
        properties: {
          path: framePathProperty,
          timestamp: {
            type: ["number", "null"],
            description:
              "where in the video the frame is, in seconds: a sheet's, the one listed, or a thumbnail's own; null " +
              "when none is known",
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

const taggerIds = Object.keys(TAGGERS);
/** Every category of tags some tagger hands on, each once. */
const categories = [...new Set(Object.values(TAGGERS).flatMap(({ thresholds }) => Object.keys(thresholds)))];
const defaultsLine = Object.entries(TAGGERS)
  .map(
    ([id, { thresholds }]) =>
      `${id}: ${Object.entries(thresholds)
        .map(([category, threshold]) => `${category} ${threshold}`)
        .join(", ")}`,
  )
  .join("; ");

const tagSettingsSchema = {
  type: "object",
  properties: {
    tagger: {
      enum: taggerIds,
      description: `The tagger to run; left empty, ${DEFAULT_TAGGER}. ${Object.entries(TAGGERS)
        .map(([id, { title }]) => `${id}: ${title}`)
        .join("; ")}`,
    },
    // Open, so a key any tagger may name gets through to the Worker, which fails a Task naming one its tagger lacks.
    // fyflow's subset takes no schema for `additionalProperties`, so each category some tagger has is a property of
    // its own, a number from 0 to 1; any other key is left to the Worker.
    thresholds: {
      type: "object",
      properties: Object.fromEntries(
        categories.map((category) => [
          category,
          {
            type: "number",
            minimum: 0,
            maximum: 1,
            description: `The ${category} tags' threshold; left empty, the tagger's default`,
          },
        ]),
      ),
      additionalProperties: true,
      description:
        "By category, the score a tag must be over to be handed on, from 0 to 1, merged over the tagger's defaults: " +
        'a category set here changes only that one, so { "character": 0.5 } keeps general at its default. A ' +
        `category the tagger lacks fails the Task. Defaults: ${defaultsLine}`,
      examples: [{ character: 0.5 }, { general: 0.25, character: 0.9 }],
    },
    maxTags: {
      type: "integer",
      minimum: 1,
      description:
        "The most tags handed on, the highest scores first; left empty, every tag over its threshold",
    },
  },
  additionalProperties: false,
};

const framesSettingsSchema = {
  ...settingsSchema,
  properties: { ...settingsSchema.properties, ...fileSettingsProperties },
};

const tagFramesSettingsSchema = {
  ...tagSettingsSchema,
  properties: { ...tagSettingsSchema.properties, ...fileSettingsProperties },
};

const RATING_EXAMPLE = {
  label: "general",
  scores: { general: 0.86, sensitive: 0.12, questionable: 0.01, explicit: 0 },
};

// What a tagger hands on of one image: TagImage's result, and each frame of TagFrames'.
const tagProperties = {
  tags: {
    type: "array",
    description:
      "the tags whose score is over their category's threshold, the highest first, at most maxTags, named as the " +
      "tagger names them (underscores kept)",
    items: {
      type: "object",
      required: ["name", "score", "category"],
      properties: {
        name: { type: "string", description: "the tag, as the tagger's tag list has it" },
        score: { type: "number", description: "its score, from 0 to 1" },
        category: {
          type: "string",
          description: `its category, as the tagger names it: ${Object.entries(TAGGERS)
            .map(([id, { thresholds }]) => `${id}: ${Object.keys(thresholds).join(", ")}`)
            .join("; ")}`,
        },
      },
    },
  },
  // Keyed by the tagger's rating labels, so `scores` can only say "an object".
  rating: {
    type: ["object", "null"],
    required: ["label", "scores"],
    description: "the tagger's rating of the image, or null for a tagger that rates nothing",
    properties: {
      label: { type: "string", description: "the rating with the highest score" },
      scores: {
        type: "object",
        description:
          "every rating's score, from 0 to 1, by rating (general, sensitive, questionable, explicit)",
      },
    },
  },
};

const tagOutputSchema = {
  type: "object",
  required: ["path", "tagger", "tags", "rating"],
  properties: {
    path: { type: "string", description: "the image's absolute path" },
    video: resultProperties.video,
    tagger: { type: "string", description: "the tagger that ran" },
    ...tagProperties,
    warnings: {
      type: "array",
      items: { type: "string" },
      description:
        "what may make the answer less good or slower than it could be (a frame sheet tagged as one image, which " +
        "TagFrames tags frame by frame; a tagger run on the CPU)",
    },
  },
  examples: [
    {
      path: EXAMPLE_IMAGE,
      tagger: "wd-eva02-large",
      tags: [
        { name: "1girl", score: 0.97, category: "general" },
        { name: "hatsune_miku", score: 0.95, category: "character" },
        { name: "long_hair", score: 0.88, category: "general" },
        { name: "twintails", score: 0.81, category: "general" },
      ],
      rating: RATING_EXAMPLE,
    },
  ],
};

// One frame of TagFrames' result, and of TagScenes' with includeFrames: its own tags and rating, as TagImage gives them.
const TAGGED_FRAME = {
  type: "object",
  required: ["timestamp", "tags", "rating"],
  properties: {
    path: framePathProperty,
    timestamp: framesOutputSchema.properties.frames.items.properties.timestamp,
    ...tagProperties,
  },
};

const tagFramesOutputSchema = {
  type: "object",
  required: ["tagger"],
  properties: {
    path: framesOutputSchema.properties.path,
    video: framesOutputSchema.properties.video,
    tagger: { type: "string", description: "the tagger that ran" },
    ...fileResultProperties,
    frames: {
      type: "array",
      description:
        "each frame's own tags and rating, in the sheet's, folder's or list's order, as TagImage gives them for an " +
        "image; in `file` instead with lists file",
      items: TAGGED_FRAME,
    },
    warnings: {
      type: "array",
      items: { type: "string" },
      description:
        "what may make the answer less good or slower than it could be (timestamps drawn on a sheet's frames, a " +
        "folder's image outside the Allowed roots, a tagger run on the CPU)",
    },
  },
  examples: [
    {
      path: EXAMPLE_SHEET,
      video: "D:/Videos/episode-01.mkv",
      tagger: "wd-eva02-large",
      frames: [
        {
          timestamp: 312.4,
          tags: [
            { name: "1girl", score: 0.97, category: "general" },
            { name: "hatsune_miku", score: 0.95, category: "character" },
            { name: "twintails", score: 0.81, category: "general" },
          ],
          rating: RATING_EXAMPLE,
        },
        {
          timestamp: 624.8,
          tags: [
            { name: "outdoors", score: 0.9, category: "general" },
            { name: "night", score: 0.84, category: "general" },
          ],
          rating: RATING_EXAMPLE,
        },
      ],
    },
  ],
};

/** The most entries in any one list of a TagScenes result when a Node sets no `maxTags` (`SCENE_MAX_TAGS`). */
export const SCENE_MAX_TAGS = 40;

/** TagScenes' grouping settings when a Node sets none, as `scenes.Settings` has them (a test holds the two together). */
export const SCENE_DEFAULTS = {
  embeddingWeight: 0.6,
  minSceneDuration: 15,
  minCoverage: 0.3,
  includeTags: true,
  includeFrames: false,
} as const;

const tagScenesSettingsSchema = {
  type: "object",
  properties: {
    tagger: tagSettingsSchema.properties.tagger,
    thresholds: {
      ...tagSettingsSchema.properties.thresholds,
      description:
        "By category, the score at which a tag counts as on in a frame, from 0 to 1, merged over the tagger's " +
        'defaults: a category set here changes only that one, so { "character": 0.5 } keeps general at its ' +
        "default. A tag stays on a little below it (0.10 for general, 0.15 for character). A character must reach " +
        "it in two frames to be a highlight. With includeFrames, each frame's own tags are those over it. A " +
        `category the tagger lacks fails the Task. Defaults: ${defaultsLine}`,
    },
    maxTags: {
      type: "integer",
      minimum: 1,
      description:
        "The most entries in any one list: a scene's tags, a scene's highlights, the summary's tags, and with " +
        `includeFrames a frame's own tags; left empty, ${SCENE_MAX_TAGS}`,
    },
    embeddingWeight: {
      type: "number",
      minimum: 0,
      maximum: 1,
      description:
        "How much the image's embedding counts against its tags when comparing two frames: 0, the tags only; 1, the " +
        `image only; left empty, ${SCENE_DEFAULTS.embeddingWeight}`,
    },
    minSceneDuration: {
      type: "number",
      minimum: 0,
      description:
        "The shortest scene, in seconds: a shorter one merges into the neighbour it is more alike; left empty, " +
        `${SCENE_DEFAULTS.minSceneDuration}`,
    },
    ignoreTags: {
      type: "array",
      items: { type: "string", minLength: 1 },
      description:
        "Tags to leave out everywhere: comparing frames, the scenes' and the summary's tags, highlights, and frames' " +
        "own tags. Named as the tagger names them (underscores kept). Shot-specific tags (framing, camera angle, " +
        "gaze, expression, head counts) are left out of comparing frames and highlights already",
      examples: [["simple_background", "letterboxed"]],
    },
    minCoverage: {
      type: "number",
      minimum: 0,
      maximum: 1,
      description:
        "The share of a scene's time a tag must be on in for the scene to list it, from 0 to 1 (the summary's, of " +
        `the video's time); left empty, ${SCENE_DEFAULTS.minCoverage}`,
    },
    includeTags: {
      type: "boolean",
      description:
        "Hand on each scene's tags and highlights, and the summary's tags; false leaves a scene only its times, " +
        "frames, rating and keyframe. Left empty, true",
    },
    includeFrames: {
      type: "boolean",
      description:
        "Also hand on each frame's own tags and rating, as TagFrames does, with whether it is blank and its scene. " +
        "Left empty, false: no frame's own tags anywhere, in the result or a file",
    },
    ...fileSettingsProperties,
  },
  additionalProperties: false,
};

// Where a scene's highlight or rating peaks, and its keyframe: a frame of the input, as the result points at it.
const pointedFrameProperties = {
  frame: {
    type: "integer",
    minimum: 0,
    description: "the frame's place in the input: the folder's order by name, or the sheet's or list's order",
  },
  timestamp: { type: "number", description: "where in the video the frame is, in seconds" },
  path: { type: "string", description: "the frame's own image; none for a frame sheet's frame" },
};

const sceneTagsProperty = {
  type: "array",
  description:
    "the tags on for at least minCoverage of the time, general and character only, by coverage then score, at most " +
    "maxTags; none with includeTags false",
  items: {
    type: "object",
    required: ["name", "category", "coverage", "score"],
    properties: {
      name: { type: "string", description: "the tag, as the tagger's tag list has it" },
      category: { type: "string", description: "general or character" },
      coverage: { type: "number", description: "the share of the time the tag is on, from 0 to 1" },
      score: { type: "number", description: "its mean score while on, from 0 to 1" },
    },
  },
};

const SCENE = {
  type: "object",
  required: ["index", "start", "end", "duration", "frames", "keyframe"],
  properties: {
    index: { type: "integer", minimum: 0, description: "the scene's place, from 0, in time order" },
    start: { type: "number", description: "where the scene starts, in seconds: halfway between two frames" },
    end: { type: "number", description: "where it ends, in seconds" },
    duration: { type: "number", description: "its length, in seconds" },
    frames: {
      type: "array",
      items: { type: "integer", minimum: 0 },
      description: "its frames' places in the input, in time order",
    },
    boundary: {
      type: "object",
      required: ["link"],
      description: "the cut before the scene; none for the first",
      properties: {
        link: {
          type: ["number", "null"],
          description:
            "the strongest similarity of two frames across the cut, in the units of `threshold`; null when no two " +
            "were compared",
        },
      },
    },
    keyframe: {
      type: "object",
      required: ["frame", "timestamp"],
      description: "the frame most like the scene as a whole",
      properties: pointedFrameProperties,
    },
    rating: {
      type: "object",
      required: ["label", "scores", "peaks"],
      description: "the scene's rating; none for a tagger that rates nothing",
      properties: {
        label: {
          type: "string",
          description:
            "the most severe rating reaching 0.5 in two frames or 0.9 in one, else the highest mean score",
        },
        scores: { type: "object", description: "every rating's mean score over the scene, by rating" },
        peaks: {
          type: "array",
          description:
            "each rating more severe than `label` that reaches 0.5 in a frame, at the frame where it peaks",
          items: {
            type: "object",
            required: ["label", "score", "frame", "timestamp"],
            properties: {
              label: { type: "string", description: "the rating" },
              score: { type: "number", description: "its peak score" },
              ...pointedFrameProperties,
            },
          },
        },
      },
    },
    tags: sceneTagsProperty,
    highlights: {
      type: "array",
      description:
        "short-lived tags not among `tags`: a general tag reaching 0.7 in one frame or 0.5 in two, a character over " +
        "its threshold in two; by score, at most maxTags. None with includeTags false",
      items: {
        type: "object",
        required: ["name", "category", "score", "frames", "frame", "timestamp"],
        properties: {
          name: { type: "string", description: "the tag, as the tagger's tag list has it" },
          category: { type: "string", description: "general or character" },
          score: { type: "number", description: "its peak score" },
          frames: { type: "integer", minimum: 0, description: "how many frames reach 0.5 on it" },
          ...pointedFrameProperties,
        },
      },
    },
  },
};

const tagScenesOutputSchema = {
  type: "object",
  required: ["tagger", "frameCount", "duration", "threshold", "summary", "count"],
  properties: {
    path: framesOutputSchema.properties.path,
    video: framesOutputSchema.properties.video,
    tagger: { type: "string", description: "the tagger that ran" },
    frameCount: { type: "integer", minimum: 1, description: "how many frames were tagged" },
    duration: {
      type: "number",
      description:
        "the video's length in seconds, as its thumbnails give it; else where the last frame's span ends (half a " +
        "frame's gap after it)",
    },
    threshold: {
      type: ["number", "null"],
      description:
        "the similarity two frames needed to be linked, in standard deviations over the video's compared pairs; " +
        "null when no two frames were compared",
    },
    summary: {
      type: "object",
      description: "the whole video's tags and rating",
      properties: {
        tags: {
          ...sceneTagsProperty,
          description:
            "each tag's coverage of the whole video (scene by scene, weighted by their time) and mean score while " +
            "on, at least minCoverage, at most maxTags; none with includeTags false",
        },
        rating: {
          type: "object",
          required: ["label", "scores"],
          description: "the video's rating; none for a tagger that rates nothing",
          properties: {
            label: { type: "string", description: "the most severe label of any scene" },
            scores: { type: "object", description: "every rating's mean score over the video, by rating" },
          },
        },
      },
    },
    count: { type: "integer", minimum: 1, description: "how many scenes, with lists result or file" },
    scenes: {
      type: "array",
      description: "the scenes, in time order; in `file` instead with lists file",
      items: SCENE,
    },
    frames: {
      type: "array",
      description:
        "with includeFrames: each frame's own tags and rating, as TagFrames gives them, in the input's order, with " +
        "whether it is blank and its scene; in `file` instead with lists file",
      items: {
        ...TAGGED_FRAME,
        required: [...TAGGED_FRAME.required, "blank", "scene"],
        properties: {
          ...TAGGED_FRAME.properties,
          blank: {
            type: "boolean",
            description:
              "whether the frame is blank (black, white or a solid card): it counts for time, not for tags",
          },
          scene: { type: "integer", minimum: 0, description: "its scene's `index`" },
        },
      },
    },
    file: {
      type: "string",
      description: "with lists file: the JSON file holding the whole result, `scenes` and `frames` included",
    },
    reused: fileResultProperties.reused,
    warnings: tagFramesOutputSchema.properties.warnings,
  },
  examples: [
    {
      path: "D:/Scenes/episode-01.1c8fcf3f1311",
      video: "D:/Videos/episode-01.mkv",
      tagger: "wd-eva02-large",
      frameCount: 278,
      duration: 1510.26,
      threshold: 1.12,
      summary: {
        tags: [
          { name: "1girl", category: "general", coverage: 0.81, score: 0.93 },
          { name: "hatsune_miku", category: "character", coverage: 0.64, score: 0.9 },
          { name: "outdoors", category: "general", coverage: 0.42, score: 0.71 },
        ],
        rating: RATING_EXAMPLE,
      },
      count: 31,
      scenes: [
        {
          index: 0,
          start: 0,
          end: 74.3,
          duration: 74.3,
          frames: [0, 1, 2],
          keyframe: { frame: 1, timestamp: 12.4, path: "D:/Scenes/episode-01.1c8fcf3f1311/frame_0005.jpg" },
          rating: {
            ...RATING_EXAMPLE,
            peaks: [
              {
                label: "sensitive",
                score: 0.62,
                frame: 2,
                timestamp: 30.1,
                path: "D:/Scenes/episode-01.1c8fcf3f1311/frame_0009.jpg",
              },
            ],
          },
          tags: [
            { name: "1girl", category: "general", coverage: 0.92, score: 0.95 },
            { name: "night", category: "general", coverage: 0.88, score: 0.81 },
          ],
          highlights: [
            {
              name: "umbrella",
              category: "general",
              score: 0.83,
              frames: 1,
              frame: 2,
              timestamp: 30.1,
              path: "D:/Scenes/episode-01.1c8fcf3f1311/frame_0009.jpg",
            },
          ],
        },
        {
          index: 1,
          start: 74.3,
          end: 120.9,
          duration: 46.6,
          frames: [3, 4, 5],
          boundary: { link: 0.41 },
          keyframe: { frame: 4, timestamp: 98.2, path: "D:/Scenes/episode-01.1c8fcf3f1311/frame_0012.jpg" },
          rating: { ...RATING_EXAMPLE, peaks: [] },
          tags: [{ name: "outdoors", category: "general", coverage: 1, score: 0.9 }],
          highlights: [],
        },
      ],
      warnings: [],
    },
    {
      frameCount: 278,
      tagger: "wd-eva02-large",
      duration: 1510.26,
      threshold: 1.12,
      summary: { rating: RATING_EXAMPLE },
      count: 31,
      file: "D:/Tags/episode-01.0123456789abcdef.json",
      reused: false,
    },
  ],
};

export default definePlugin({
  id: "fyflow-ext.image-tags",
  version: "0.4.0",
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
  profile: { schema: profileSchema, startup: ["modelDir", "batchSize", "tagBatchSize"] },
  // Asked every few seconds while the Plugins page is open: a check that each model's file exists, nothing slower.
  status: ({ dataDir }) => modelsStatus(dataDir),
  // Asked by a Host of SDK 0.12 or later when someone opens the Plugin's Data link (a Host before never asks): what the
  // data directory holds, which the Host measures and deletes when a person asks. Workers download a deleted model
  // again, as on a first run.
  data: ({ dataDir }) => modelsData(dataDir),
  workers: {
    classifyImage: {
      python: "image_tags.worker:ClassifyImage",
      settingsSchema,
      inputSchema,
      outputSchema,
    },
    classifyFrames: {
      python: "image_tags.worker:ClassifyFrames",
      settingsSchema: framesSettingsSchema,
      inputSchema: framesInputSchema,
      outputSchema: framesOutputSchema,
    },
    tagImage: {
      python: "image_tags.worker:TagImage",
      settingsSchema: tagSettingsSchema,
      inputSchema,
      outputSchema: tagOutputSchema,
    },
    tagFrames: {
      python: "image_tags.worker:TagFrames",
      settingsSchema: tagFramesSettingsSchema,
      inputSchema: framesInputSchema,
      outputSchema: tagFramesOutputSchema,
    },
    tagScenes: {
      python: "image_tags.worker:TagScenes",
      settingsSchema: tagScenesSettingsSchema,
      inputSchema: framesInputSchema,
      outputSchema: tagScenesOutputSchema,
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
      profile: ["modelDir", "batchSize", "outputDir"],
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
    TagImage: {
      worker: "tagImage",
      profile: ["modelDir", "tagBatchSize"],
      // A Worker and pool of its own, holding the tagger's session once (over 1 GB of GPU memory) and the same
      // resident units as the classifiers' pools, so it takes turns with them on the `gpu` unit. Up to 16 Tasks at
      // once (`TAG_TASKS_AT_ONCE` in `worker.py`), whose images go through the tagger together, up to `tagBatchSize`
      // to a forward pass, a batch waiting at most 20 ms for more. Stopped after 5 s with no Task, as the others.
      pool: {
        maxThreads: 1,
        maxConcurrentTasks: 16,
        idleTimeout: 5_000,
        residentGroups: RESIDENT_GROUPS,
        variants: POOL_VARIANTS,
      },
      icon: "tag",
      description: "Tags one image (characters, general tags and a rating) with a local tagger.",
    },
    TagFrames: {
      worker: "tagFrames",
      profile: ["modelDir", "tagBatchSize", "outputDir"],
      // A Worker and pool of its own, as ClassifyFrames has: fyflow gives each Node Type its own pool, so it cannot
      // share TagImage's process and session. It holds the same resident units, so the two take turns on the `gpu`
      // unit and a GPU never holds two copies of the tagger. Two Tasks at once, so that one Task's decoding and
      // cutting out of frames overlaps the other's forward passes, each `tagBatchSize` frames.
      pool: {
        maxThreads: 1,
        maxConcurrentTasks: 2,
        idleTimeout: 5_000,
        residentGroups: RESIDENT_GROUPS,
        variants: POOL_VARIANTS,
      },
      icon: "image",
      description:
        "Tags each frame of a VideoThumbnail frame sheet, or a folder of frames (characters, general tags and a rating), with a local tagger.",
    },
    TagScenes: {
      worker: "tagScenes",
      profile: ["modelDir", "tagBatchSize", "outputDir"],
      // A Worker and pool of its own, as TagFrames has, holding the same resident units: its tagger's embedding
      // session never shares a process with a plain one, and the GPU never holds two copies of the tagger. Two Tasks
      // at once, as TagFrames, each running its frames through the tagger `tagBatchSize` at a time.
      pool: {
        maxThreads: 1,
        maxConcurrentTasks: 2,
        idleTimeout: 5_000,
        residentGroups: RESIDENT_GROUPS,
        variants: POOL_VARIANTS,
      },
      icon: "tag",
      description:
        "Tags a video's frames and groups them into scenes: each scene's tags, highlights and rating, and the video's.",
    },
  },
});

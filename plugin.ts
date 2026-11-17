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
  return {
    summary: "Downloaded models",
    items: found.sort((a, b) => rank(a) - rank(b)).map(itemOf),
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
        "batches the images of the Tasks running at once; TagFrames runs a Task's frames that many at a time. " +
        "Takes effect at the next Reload.",
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

const tagFramesOutputSchema = {
  type: "object",
  required: ["path", "tagger", "frames"],
  properties: {
    path: { type: "string", description: "the frame sheet's, folder's or image's absolute path" },
    video: resultProperties.video,
    tagger: { type: "string", description: "the tagger that ran" },
    frames: {
      type: "array",
      description:
        "each frame's own tags and rating, in the sheet's or folder's order, as TagImage gives them for an image",
      items: {
        type: "object",
        required: ["timestamp", "tags", "rating"],
        properties: {
          timestamp: framesOutputSchema.properties.frames.items.properties.timestamp,
          ...tagProperties,
        },
      },
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

export default definePlugin({
  id: "fyflow-ext.image-tags",
  version: "0.2.1",
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
      settingsSchema,
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
      settingsSchema: tagSettingsSchema,
      inputSchema: framesInputSchema,
      outputSchema: tagFramesOutputSchema,
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
      profile: ["modelDir", "tagBatchSize"],
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
  },
});

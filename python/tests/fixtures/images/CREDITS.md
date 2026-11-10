# Test image credits

The reference frames here (`<category>-NN-<what>.jpg`) are other people's work under the licences below, not under
this repository's MIT licence. Each is redistributed under its own licence; the CC BY-SA ones are shared under the same
licence (CC BY-SA 4.0). Keep this file with them.

The other images (`tiny.jpg`, `transparent.png`, `keyed.png`, `keyed-grey.png`) are synthetic, made by
`python/scripts/golden.py`, and are part of this repository (MIT).

Licences were read from each source page (the Hugging Face dataset card or the Wikimedia Commons file page) when the
files were downloaded, on 2026-11-10.

**Changes made to every reference frame**: resized (Lanczos) to 640 px on its long side and saved as JPEG quality 90,
to keep the repository small (the models see 384×384 anyway). Nothing was cropped, recoloured or otherwise edited.
"Original sha256" is that of the file as downloaded (for a PNG source, also that of the JPEG quality 92 copy it was
first re-encoded to, at full size); "fixture sha256" is that of the file here.

| File | Category | Licence |
| --- | --- | --- |
| `anime-01-rainy-bus-stop.jpg` … `anime-06-riverside-picnic.jpg` | anime | MIT |
| `anime-07-yakitori-stall-portrait.jpg` | anime | MIT |
| `cg-01-big-buck-bunny.jpg`, `cg-02-sintel.jpg` | cg | CC BY 3.0 |
| `cg-03-elephants-dream.jpg` | cg | CC BY 2.5 |
| `cg-04-cosmos-laundromat.jpg` | cg | CC BY-SA 4.0 |
| `cg-05-coffee-run.jpg` | cg | CC BY 4.0 |
| `live_action-01-office-interview.jpg`, `live_action-02-bristol-street.jpg` | live_action | CC BY 3.0 |
| `live_action-03-redwood-creek.jpg` | live_action | CC BY 4.0 |
| `live_action-04-dim-kitchen-ghana.jpg`, `live_action-05-oral-history-interview.jpg` | live_action | CC BY-SA 4.0 |

## Anime

### anime-01 to anime-06: common source

- Dataset: `suparnojit/ghibli-dataset-flux`, https://huggingface.co/datasets/suparnojit/ghibli-dataset-flux (commit
  85ccba3f0dc39c8fa7da2f385ef2af39e18056cd, 2025-08-27)
- Author/uploader: Hugging Face user `suparnojit`
- Generator: AI text-to-image; the dataset's name says "flux" (FLUX.1 family, exact variant not stated). The prompts
  describe original scenes "in the Studio Ghibli style" and name no existing characters or works.
- Licence: MIT (dataset card `license: mit`; https://opensource.org/license/mit). MIT asks that its notice be kept:
  this file is that notice.
- Licence doubts: (1) The dataset card is just the licence tag: no description, no statement of the model. (2) If the
  model was FLUX.1 [dev], its non-commercial licence says BFL claims no rights in outputs and outputs may be used for
  any purpose, so the uploader's MIT grant should hold; if FLUX.1 [schnell] (Apache-2.0), outputs are unrestricted.
  (3) Purely AI-generated images may not be copyrightable at all in the US, which only makes redistribution safer.
  (4) The style imitates a real studio; style is not protected, and no recognisable characters appear.
- Look: Ghibli feature film rather than modern TV anime: painted backgrounds, cel-like characters, about 1.85:1.

### anime-01-rainy-bus-stop.jpg

- Title: `image_22.png`
- Original URL: https://huggingface.co/datasets/suparnojit/ghibli-dataset-flux/resolve/main/image_22.png (982×531)
- Shows: a young woman in a raincoat in a wooden rural bus shelter, rain on an empty road, overcast green palette.
- Original sha256: PNG 9315f7d7c9271568a546345e184b961f78c174d7f195a43be03ec9b7e4652622; JPEG q92
  399f6bb930edfc7fbd1c184951b55bd811bb68ffa5a4e23a965e1df5b1454fff
- Fixture: 640×346, sha256 0646a63161012c355339a2d8d6b0afc9f76050eeb63ef642523d573f1cca830a

### anime-02-train-window.jpg

- Title: `image_28.png`
- Original URL: https://huggingface.co/datasets/suparnojit/ghibli-dataset-flux/resolve/main/image_28.png (983×525)
- Shows: a woman in a coat seated in a train carriage, sunset countryside through the window, an apple on the table.
- Original sha256: PNG 1a5921056179b1d3ad47579db66783935fc8750450594938bedf3b11f2905349; JPEG q92
  ba3d82d87ea6e509766846a2f56176dce1c8367ae791ca2102ffe2b0e899eb1a
- Fixture: 640×342, sha256 b3d07f8b49fb239eb7edbd38d0da614f6284dae0188a26343d48873c8d346b48

### anime-03-festival-night-market.jpg

- Title: `image_27.png`
- Original URL: https://huggingface.co/datasets/suparnojit/ghibli-dataset-flux/resolve/main/image_27.png (985×530)
- Shows: a crowded summer-festival street at night, people in yukata, lanterns, a mask stall with an old vendor. Signs
  carry pseudo-Japanese text.
- Original sha256: PNG 73f63eb06a5e1f776f7a0b1ad1b7895588326f6ffb4d036f42957a542916dc21; JPEG q92
  f3e0d0f0d64ba576447e245010e07b4cff5b52da750358cca9ef9044781d2a71
- Fixture: 640×344, sha256 3746f4428cb22015dd058dde01b709302d9b2d7d923ca0e01290fdb951b58683

### anime-04-seaside-railway.jpg

- Title: `image_13.png`
- Original URL: https://huggingface.co/datasets/suparnojit/ghibli-dataset-flux/resolve/main/image_13.png (982×526)
- Shows: a boy sketching on a bench beside a cliff-top railway, sea and big cumulus clouds, a small train in the
  distance.
- Original sha256: PNG b9c68af3a9f57233ece00485b64de5ce536892452017447f370ca97fab6ec530; JPEG q92
  14b4cdf072ff27726b65dd902e75e41e0f088eba2344b30342b0e7f34a5b2ad2
- Fixture: 640×343, sha256 8ca45055104dcec4bfc13eb6e672beeb3db0048cad4b0c15f2cbda5fc6384ae2

### anime-05-bakery-interior.jpg

- Title: `image_20.png`
- Original URL: https://huggingface.co/datasets/suparnojit/ghibli-dataset-flux/resolve/main/image_20.png (980×533)
- Shows: a warm-lit bakery interior, two bakers kneading dough and a girl in a headscarf glazing buns, ovens behind.
- Original sha256: PNG dbfc0d29deada74d67aea14ae1a1d30eb058729020e0b10d1f80119f61f5f64d; JPEG q92
  3f947b176e7e8b77c5eb6dd89825f98a87c93161e47a4abaf20588c272713c90
- Fixture: 640×348, sha256 1b1e060a2ed29749d8bd7885e5e98f3e61310f36337202393bfa2b7ae88b0e3f

### anime-06-riverside-picnic.jpg

- Title: `image_26.png`
- Original URL: https://huggingface.co/datasets/suparnojit/ghibli-dataset-flux/resolve/main/image_26.png (983×521)
- Shows: a girl reading under a willow and a boy in a straw hat on a checked picnic blanket by a river, dappled
  sunlight.
- Original sha256: PNG 75e335e335c88cd4d180e24014a2332820b96bc9871efb44a8871fc11ad8fe82; JPEG q92
  f525bb0fad2b054cd8ef805296524024d2fda5ad75631638649f867230b75102
- Fixture: 640×339, sha256 95a0e72cea7bbd9d133d27b0e67e3b35bff1a59af4fdd9769d7eba2e376d3f4e

### anime-07-yakitori-stall-portrait.jpg

- Title: `video_dataset/ComfyUI_17575_.png`
- Dataset: `trojblue/test-HunyuanVideo-anime-images`,
  https://huggingface.co/datasets/trojblue/test-HunyuanVideo-anime-images
- Author/generator: Hugging Face user `trojblue`. The card says the images were generated with "a V-prediction ZtSNR
  SDXL model that I trained on anime images", from GPT-suggested prompts, and hand-selected.
- Licence: MIT (the dataset card's "License" section; https://opensource.org/license/mit)
- Original URL:
  https://huggingface.co/datasets/trojblue/test-HunyuanVideo-anime-images/resolve/main/video_dataset/ComfyUI_17575_.png
  (896×1152, portrait)
- Shows: a night market, an old man and a smiling girl at a yakitori grill under paper lanterns, a starry sky.
- Licence doubts: the base SDXL is under CreativeML OpenRAIL++-M, which does not restrict ownership or redistribution
  of outputs. The fine-tune's training data (anime images, probably scraped) is not stated. The frame is portrait and
  more illustration than TV frame.
- Original sha256: PNG ed124f61f87321396d80d634a89b76447ff7a4ba5c43192eefebac5d0aed1be6; JPEG q92
  b5ec2c4e282c8c0912e588444f3935fb8d16ff4cc4042553237558ac1f25347f
- Fixture: 498×640, sha256 d9ce518ffd519a8d11880c13393874c8f01b50706681e69d9eefb7d417d2fb18

## CG

All from Wikimedia Commons; the original URL is the Commons-rendered 1280 px thumbnail that was downloaded.

### cg-01-big-buck-bunny.jpg

- Title: `Big.Buck.Bunny.-.Frank.Rinky.Gimera.png`
- Author: Blender Foundation (Big Buck Bunny, 2008)
- Licence: CC BY 3.0, https://creativecommons.org/licenses/by/3.0/
- Source page: https://commons.wikimedia.org/wiki/File:Big.Buck.Bunny.-.Frank.Rinky.Gimera.png
- Original URL:
  https://thumb.wikimedia.org/wikipedia/commons/thumb/2/2b/Big.Buck.Bunny.-.Frank.Rinky.Gimera.png/1280px-Big.Buck.Bunny.-.Frank.Rinky.Gimera.png
  (of a 1920×1000 original)
- Shows: three cartoony 3D rodents (squirrel, flying squirrel, chinchilla) on a grassy hill under a tree; a bright,
  saturated render.
- Original sha256: JPEG q92 of the 1280×667 PNG 54efffb32202561c0776290c3cd498e763c63936794cec46e505b7dbbcdffb8c
- Fixture: 640×334, sha256 3529dba55399464c02dd3dd6b72c732294d95f445b2fa365547b630795b72074

### cg-02-sintel.jpg

- Title: `Sintel-screenshot-1.jpg`
- Author: Blender Foundation (Sintel, 2010)
- Licence: CC BY 3.0, https://creativecommons.org/licenses/by/3.0/
- Source page: https://commons.wikimedia.org/wiki/File:Sintel-screenshot-1.jpg
- Original URL: https://upload.wikimedia.org/wikipedia/commons/f/fa/Sintel-screenshot-1.jpg (1280×544)
- Shows: a semi-realistic 3D render of Sintel fighting an old bald man, desaturated, 2.35:1.
- Original sha256: b425532acda899b4900d24c3d45cc640a0995c5fc15007a45382007d8c43a3a1
- Fixture: 640×272, sha256 9f87244cffae58429baee169588820edea96c017de712cff9957643b0261faf3

### cg-03-elephants-dream.jpg

- Title: `Elephants Dream s5 both.jpg`
- Author: Ton Roosendaal et al. / Blender Foundation (Elephants Dream, 2006)
- Licence: CC BY 2.5, https://creativecommons.org/licenses/by/2.5/
- Source page: https://commons.wikimedia.org/wiki/File:Elephants_Dream_s5_both.jpg
- Original URL:
  https://thumb.wikimedia.org/wikipedia/commons/thumb/e/e8/Elephants_Dream_s5_both.jpg/1280px-Elephants_Dream_s5_both.jpg
  (of a 1920×1080 original)
- Shows: the characters Emo and Proog in a dark mechanical interior; an early-2000s 3D render with stylised faces.
- Original sha256: 1e1b8f3eebd15f2f40c492c249a0e211fc11868c614ae5066519129590d10524
- Fixture: 640×360, sha256 92e5793e677b88f7b11e581330958f0c84c4bd60e00c5126c4586b938dbcfd27

### cg-04-cosmos-laundromat.jpg

- Title: `CosmosLaundromat-final-scene.jpg`
- Author: Blender Foundation (Cosmos Laundromat, 2015)
- Licence: CC BY-SA 4.0, https://creativecommons.org/licenses/by-sa/4.0/. This fixture is shared under the same
  licence.
- Source page: https://commons.wikimedia.org/wiki/File:CosmosLaundromat-final-scene.jpg
- Original URL:
  https://thumb.wikimedia.org/wikipedia/commons/thumb/2/22/CosmosLaundromat-final-scene.jpg/1280px-CosmosLaundromat-final-scene.jpg
  (of a 1920×814 original)
- Shows: a long laundromat aisle of shiny washing machines receding to a palm-tree mural; a near-photoreal render.
- Licence doubts: Commons tags the still CC BY-SA 4.0, while Blender's own release of the film is CC BY 4.0; the
  stricter Commons tag is followed here.
- Original sha256: 20a35ec6dc693e9d45cd7e611c5d2771c888cef9288a767fe5f9b87b5e2e3054
- Fixture: 640×272, sha256 6c1237b84017df34e92979e405b86e7c4e494bd988b7fa4eb32ad8c695540002

### cg-05-coffee-run.jpg

- Title: `Coffee Run - screenshot-Location 06 Wedding-Run 2.png`
- Author: Blender Foundation (Coffee Run, 2020)
- Licence: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/
- Source page: https://commons.wikimedia.org/wiki/File:Coffee_Run_-_screenshot-Location_06_Wedding-Run_2.png
- Original URL:
  https://thumb.wikimedia.org/wikipedia/commons/thumb/1/1d/Coffee_Run_-_screenshot-Location_06_Wedding-Run_2.png/1280px-Coffee_Run_-_screenshot-Location_06_Wedding-Run_2.png
  (of a 1920×804 original)
- Shows: pastel, flat-lit stylised 3D: a wedding in a pink church with silhouetted guests; close to 2D illustration.
- Original sha256: JPEG q92 of the 1280×536 PNG 9eb38fddbe3fc9f60086742ea8e27ea1cda3aa19b4305f139509d141d632e307
- Fixture: 640×268, sha256 6017adfceefed20b89bb5502795ff01e9dd856a25f5dacb5b4671223d7b1a5d3

## Live action

All are Wikimedia Commons server-rendered stills (`seek=`) of the listed video, 1280×720, as downloaded.

### live_action-01-office-interview.jpg

- Title: `Becoming an Engaged Citizen- A Discussion with Ryan Quarles, Kentucky Commissioner of Agriculture.webm`, frame
  at 300 s
- Author: UK Martin-Gatton CAFE (University of Kentucky College of Agriculture, Food and Environment), via YouTube
  (id yfGNW4N-OEE, 2022-05-05)
- Licence: CC BY 3.0, https://creativecommons.org/licenses/by/3.0/ (Commons tag `{{YouTube CC-BY}}`)
- Source page:
  https://commons.wikimedia.org/wiki/File:Becoming_an_Engaged_Citizen-_A_Discussion_with_Ryan_Quarles,_Kentucky_Commissioner_of_Agriculture.webm
- Original URL:
  https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/Becoming_an_Engaged_Citizen-_A_Discussion_with_Ryan_Quarles%2C_Kentucky_Commissioner_of_Agriculture.webm/1280px-seek%3D300-Becoming_an_Engaged_Citizen-_A_Discussion_with_Ryan_Quarles%2C_Kentucky_Commissioner_of_Agriculture.webm.jpg
- Shows: a man in a blazer talking and gesturing in an office (desk, US flag, shelves), soft indoor light.
- Licence doubts: the Commons page still carries an unreviewed `{{LicenseReview}}` tag: no reviewer has yet confirmed
  the YouTube CC BY licence.
- Original sha256: 7714dc56eb7a1f4f9328ec0d833fb19e25cbfd91a3c7affd040afcb1ff8c252f
- Fixture: 640×360, sha256 d57a47a4536fabd6b2a4ec55dc737191d65c1c3031f2c2de0cb76aed4b455c9d

### live_action-02-bristol-street.jpg

- Title: `Walking in BRISTOL - UK - 4K 60fps (UHD).webm`, frame at 400 s
- Author: POPtravel (YouTube channel)
- Licence: CC BY 3.0, https://creativecommons.org/licenses/by/3.0/ (Commons tag `{{YouTube CC-BY}}`, with a
  completed `{{YouTubeReview}}`, 2021-05-19)
- Source page: https://commons.wikimedia.org/wiki/File:Walking_in_BRISTOL_-_UK_-_4K_60fps_(UHD).webm
- Original URL:
  https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/Walking_in_BRISTOL_-_UK_-_4K_60fps_%28UHD%29.webm/1280px-seek%3D400-Walking_in_BRISTOL_-_UK_-_4K_60fps_%28UHD%29.webm.jpg
- Shows: a low-angle walking shot down a pedestrian shopping street (Broadmead), overcast daylight, a few pedestrians.
- Original sha256: de15a95482b97411528a245acbb915793e330049a82797d8969207b9310bdf98
- Fixture: 640×360, sha256 fcf1608c71a7582d50844b17c66d5c4561d62229b589e21ca94dd09d19fd4c22

### live_action-03-redwood-creek.jpg

- Title: `Fife Creek - January 2023 - Sarah Stierch.webm`, frame at 10 s
- Author: Sarah Stierch (User:Missvain)
- Licence: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/
- Source page: https://commons.wikimedia.org/wiki/File:Fife_Creek_-_January_2023_-_Sarah_Stierch.webm
- Original URL:
  https://upload.wikimedia.org/wikipedia/commons/thumb/5/5a/Fife_Creek_-_January_2023_-_Sarah_Stierch.webm/1280px-seek%3D10-Fife_Creek_-_January_2023_-_Sarah_Stierch.webm.jpg
- Shows: a redwood forest with a creek, ferns and a wooden footbridge (Armstrong Woods, California), diffuse daylight.
- Original sha256: b3c809c97462d793ac2decfe4bbf61fd6ff894e12813401ec06b9f1b18fcda27
- Fixture: 640×360, sha256 a6e06ad08d56478fd00a420f1b89d8d931de8d332df0cb0c6612acb2475c0fe1

### live_action-04-dim-kitchen-ghana.jpg

- Title: `Traditional kitchen in Northern Ghana.webm`, frame at 8 s
- Author: Salifu Wumpini Hussein (User:OJjnr)
- Licence: CC BY-SA 4.0, https://creativecommons.org/licenses/by-sa/4.0/. This fixture is shared under the same
  licence.
- Source page: https://commons.wikimedia.org/wiki/File:Traditional_kitchen_in_Northern_Ghana.webm
- Original URL:
  https://upload.wikimedia.org/wikipedia/commons/thumb/8/82/Traditional_kitchen_in_Northern_Ghana.webm/1280px-seek%3D8-Traditional_kitchen_in_Northern_Ghana.webm.jpg
- Shows: a woman in a striped top cooking at a hearth in a dark traditional kitchen, pots and bowls; low light, phone
  video.
- Original sha256: 71dcad2d06705dad65c3b20b5d9b19a5847f2c59bfc2ab6ded38ac28c0333e4a
- Fixture: 640×360, sha256 b5ecab64d56153ccfeb60003efe0fdad8b445dd0ff28ce47ed9a54bdb86eb061

### live_action-05-oral-history-interview.jpg

- Title: `Pramila Devi oral history part 3.webm`, frame at 60 s
- Author: Amrit Sufi (User:Amrit Sufi)
- Licence: CC BY-SA 4.0, https://creativecommons.org/licenses/by-sa/4.0/. This fixture is shared under the same
  licence.
- Source page: https://commons.wikimedia.org/wiki/File:Pramila_Devi_oral_history_part_3.webm
- Original URL:
  https://upload.wikimedia.org/wikipedia/commons/thumb/b/bc/Pramila_Devi_oral_history_part_3.webm/1280px-seek%3D60-Pramila_Devi_oral_history_part_3.webm.jpg
- Shows: an elderly woman in a brown shawl, seated and speaking indoors, a straw broom in the foreground, natural
  window light, off-centre framing.
- Original sha256: d7dc4adbfccc70d287e202ff88ca9b0711d04894834ae409f07fae5cd051addc
- Fixture: 640×360, sha256 b1f5ff0271e640749140f8b2c7e2bce52921308459be5c0069fad41f57c227b4

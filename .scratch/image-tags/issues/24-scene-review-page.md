# 24: A page to review TagScenes' scenes

**What to build:** `python/scripts/review_scenes.py`, which runs TagScenes on real frame folders and writes a local
HTML page. On the page, the user marks only what is wrong, then saves a review file keyed by time. See
`TagScenes-spec.md` §7.3.

**Blocked by:** 23

**Status:** ready-for-agent

- [ ] Runs the `TagScenes` worker directly (no Host) on each folder or sheet given.
  - `--set NAME=VALUE` overrides a setting or a `scenes.py` constant.
  - It writes `<name>.result.json` and `<name>.review.html` to the gitignored `.scratch/image-tags/notes/review/`.
- [ ] The page has one card per scene: times, rating and peaks, tags and highlights as chips, and thumbnails from file
  URLs. It is self-contained and works in light and dark.
- [ ] Review by exception:
  - "merge with previous" per card
  - click a gap between thumbnails to mark a cut
  - click a chip to strike it out
  - a note per scene
- [ ] **Save review** downloads `{video, settings, merges, cuts, wrong, notes}`, keyed by times.
- [ ] With a saved review beside the page: "N boundaries changed since your last review", and each earlier comment is
  marked addressed or still open.
- [ ] Prints times and counts only, never tag names.

## Notes

- The page is never published as an Artifact or committed: it shows frames from the user's videos.

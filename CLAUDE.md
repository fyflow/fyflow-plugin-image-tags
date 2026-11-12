## Agent skills

### Issue tracker

Issues and specs live as local markdown files under `.scratch/<feature>/`; private notes (requests and replies to other repos, prompts) go in the gitignored `.scratch/<feature>/notes/` and are never linked from tracked files. See `docs/agents/issue-tracker.md`.

### Triage labels

Default triage vocabulary (needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix), recorded as a `Status:` line in each issue file. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.

# FreeAgent installer

The **minimal online installer** described in [`WEBAPP_DESIGN.md` §5A](../WEBAPP_DESIGN.md).
macOS + Linux. No changes to any existing FreeAgent code — this directory is
entirely additive.

## What's here

| File | Role |
|---|---|
| `freeagent-install.sh` | The installer **template** — also a working installer on its own. Fetches the repo, writes `.env`, creates a launcher, optionally starts FreeAgent. |
| `generate-installer.sh` | Bakes a user's `.env` choices into the template and emits a single personalized download. This is exactly what the web wizard does server-side. |

## How it works (the super-simple path)

1. The web wizard asks a few questions (Local Ollama / Remote / OpenAI, model, a
   URL or key if relevant) and produces a small set of `.env` **overrides**.
2. It runs the equivalent of `generate-installer.sh` to bake those overrides into
   the template, and serves **one file** sized to the user's OS:
   - macOS → `FreeAgent-Installer.command` (double-clickable → opens Terminal)
   - Linux → `freeagent-install.sh`
3. The user downloads that one file and opens it. It:
   - downloads the repo tarball into `~/Applications/FreeAgent` (**user-space, no
     sudo**, `curl` only — no git needed yet),
   - writes the baked-in `.env` (backing up any existing one to `.env.bak`),
   - creates the launcher / icon,
   - and (by default) starts FreeAgent, whose unchanged first-run logic installs
     Ollama/OpenCode, pulls the model, and drops into the OpenCode TUI.

Nothing in the repo is modified. It works because [`FreeAgent`](../FreeAgent)
already (a) respects a pre-existing `.env` and (b) fills every unset setting from
its own defaults, so a tiny overrides-only `.env` is enough.

## The launcher / icon

- **macOS:** a minimal `FreeAgent.app` in `~/Applications` (real Launchpad /
  Spotlight icon; opens Terminal on the CLI, which needs a TTY), plus a
  `Launch FreeAgent.command` in the install dir and a copy on the Desktop.
- **Linux:** `~/.local/share/applications/freeagent.desktop` with `Terminal=true`,
  so "FreeAgent" appears in the application menu and launches in the terminal.

## Try it locally (no web app)

```sh
# 1. Describe the settings you want changed (overrides only — the rest use defaults)
printf 'FA_MAIN_PROVIDER=openai\nOPENAI_API_KEY=sk-...\nFA_MODEL=gpt-oss:20b\n' > /tmp/my.env

# 2. Generate a personalized installer for macOS
./generate-installer.sh --env /tmp/my.env --os mac --out ~/Downloads/FreeAgent-Installer.command

# 3. Double-click it (macOS) or run it. To install without launching:
FA_LAUNCH_AFTER=0 bash ~/Downloads/FreeAgent-Installer.command
```

Runtime overrides accepted by the installer: `FA_INSTALL_DIR`, `FA_REPO_URL`,
`FA_BRANCH`, `FA_LAUNCH_AFTER`.

## Caveats (see §5A.5 of the design doc)

- **macOS Gatekeeper:** an unsigned `.command` downloaded from the web is
  quarantined; the first open needs a right-click → *Open*. Removing that step
  means signing + notarizing a `.pkg`/`.app` (cost + CI) — deferred for v1.
- **Re-install:** a fresh download overwrites `.env`, but the previous one is
  saved to `.env.bak` first, so hand edits are recoverable.
- **Windows** is out of scope for this minimal installer (bash launcher).
- The Linux `.desktop` path is structurally standard but has not been exercised
  on a live desktop environment here; verify on a target distro before shipping.

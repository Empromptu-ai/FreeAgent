# FreeAgent Web App — Design Spec

A hosted web app (servable from any URL) that lets a user **install, configure, and
start OpenCode + FreeAgent on their own machine**, and then either open a real
Terminal running OpenCode or drive OpenCode from an **in-browser terminal + code
viewer**.

This document is the multi-phase design. It is deliberately explicit about what a
website *can* and *cannot* do, because that boundary determines the whole
architecture.

---

## 1. Goals & non-goals

**Goals**
- A page at an arbitrary URL that walks a novice from "nothing installed" to a
  running, configured FreeAgent + OpenCode.
- An **Install button** on that page that begins the local install (as
  frictionlessly as the browser allows).
- Configuration in the browser: Local Ollama / Remote Ollama / OpenAI, model
  choice sized to the machine — the same decisions [`Setup_FreeAgent`](Setup_FreeAgent)
  makes today.
- A **Start button**: either launch a native Terminal running OpenCode, or render
  OpenCode inside the page (xterm.js) alongside a code viewer.

**Non-goals**
- Running the model or the proxy in the browser. The model, Ollama, the proxy,
  and OpenCode all run **on the user's machine**. The web app is a control plane.
- Replacing the bash scripts wholesale. The companion daemon reuses the same
  install/launch logic; the scripts remain valid for terminal-only users.

---

## 2. The hard constraint (read this first)

A page served from `https://app.freeagent.dev` runs in the browser sandbox. It
**cannot**, on its own:

- install software, write to `~/.config`, or run `./FreeAgent`;
- spawn a terminal or any process;
- read the machine's real RAM (only the coarse, Chrome-only, 8 GB-capped
  `navigator.deviceMemory`);
- freely call `http://127.0.0.1` from an HTTPS origin (Chrome's Local/Private
  Network Access now gates this with preflights and, increasingly, a prompt).

Therefore the system is necessarily **two parts**:

1. **The web app** (hosted anywhere) — guides, configures, generates, and *controls*.
2. **A local companion daemon** (installed once on the user's machine) — does
   everything privileged: install, configure, spawn OpenCode, serve the PTY.

Everything interactive ("start OpenCode", in-page terminal, code viewer) lives on
the companion side. The web app talks to it over an authenticated local channel.

### 2.1 What the "Install" button really does

The button cannot silently install. The smoothest thing it *can* do, in order of
preference by platform:

1. **Detect the OS** (User-Agent + `navigator.userAgentData`) and, on click,
   **download the correct signed/notarized installer** (`.pkg`/`.dmg` on macOS,
   `.deb`/AppImage on Linux, `.msi`/`.exe` on Windows). The page then shows a
   single instruction ("open the downloaded file") and starts **polling for the
   companion to come online** (§6.3). When it appears, the UI flips to
   "Connected" automatically.
2. **Fallback: copy a one-liner** (`curl -fsSL https://get.freeagent.dev | bash`)
   to the clipboard with a Copy button and a "paste this in Terminal" step —
   exactly the bootstrap Ollama and OpenCode already use.

So "push Install and it connects" is achievable as: **click → download signed
installer → user double-clicks once → companion registers back to the page.** The
one unavoidable user action is approving/opening the installer, which is a browser
+ OS security requirement, not a design choice.

---

## 3. Architecture

```
   ┌─────────────────────────────────────────────┐
   │  Browser  (https://app.freeagent.dev)         │
   │                                               │
   │  Wizard UI ── reads ──▶ shared config spec     │
   │  xterm.js terminal                             │
   │  code viewer (Monaco / file tree)              │
   └───────────────┬───────────────────────────────┘
                   │  HTTPS + WSS to local companion
                   │  https://local.freeagent.dev:PORT
                   │  (real cert, A-record → 127.0.0.1)
                   │  paired with a one-time token
                   ▼
   ┌─────────────────────────────────────────────┐
   │  Companion daemon  (user's machine)            │
   │                                               │
   │   control API  /status /install /config /start │
   │   /pty  (WebSocket ⟷ PTY running opencode)      │
   │   ── wraps ──▶ existing install/launch logic    │
   │                                               │
   │   ┌──────────────┐   ┌───────────────────┐     │
   │   │ fa_proxy.py   │   │  opencode (TUI)    │     │
   │   │ 127.0.0.1:49786│◀─│  in a PTY          │     │
   │   └──────┬────────┘   └───────────────────┘     │
   │          │ /v1/chat/completions                 │
   │          ▼                                       │
   │   Ollama (local/remote)  or  OpenAI              │
   └─────────────────────────────────────────────┘
```

The companion **is essentially [`examples/fa_proxy.py`](examples/fa_proxy.py)
grown up**: same FastAPI app, plus a control API, a PTY WebSocket, a static-file
mount for the (optionally self-hosted) UI, CORS, a health route, and the
cert/pairing layer. The proxy today has none of the last four.

---

## 4. Phase 0 — Extract the setup "brains" into one shared spec

Today the intelligence lives in bash + the `.env` template:

- Model catalog + suggested RAM: [`Setup_FreeAgent:319-327`](Setup_FreeAgent#L319-L327)
  (`MODEL_NAMES`, `MODEL_MINGB`).
- "Largest model that fits": [`suggest_model_index_for_ram`](Setup_FreeAgent#L332-L338).
- RAM detection: [`detect_ram_gb`](Setup_FreeAgent#L294-L312).
- Settings schema, help text, sections, field types (bool/enum/int/path):
  parsed out of [`.env.example`](.env.example) and
  [`field_type`](Setup_FreeAgent#L140-L154).

**Deliverable:** move this into one declarative source of truth, e.g.
`spec/freeagent.config.json`:

```jsonc
{
  "models": [
    { "tag": "qwen3.5:2b",   "minGB": 4 },
    { "tag": "qwen3.5:9b",   "minGB": 8 },
    { "tag": "gpt-oss:20b",  "minGB": 20 },
    { "tag": "qwen3.5:27b",  "minGB": 24 },
    { "tag": "qwen3.6:35b",  "minGB": 32 },
    { "tag": "gpt-oss:120b", "minGB": 90 }
  ],
  "settings": [
    {
      "key": "OLLAMA_BASE_URL", "section": "Server", "type": "text",
      "default": "http://localhost:11434",
      "help": "Upstream Ollama the proxy forwards to."
    }
    // …one entry per .env.example variable, with type/enum/help/default/optional
  ]
}
```

Both consumers read it:
- `Setup_FreeAgent` (or a thin Python replacement) renders the same interactive
  flow from the spec instead of hard-coded arrays.
- The web wizard renders its form and its `.env` generator from the same spec.

This is the single most important refactor — without it the web wizard and the
script drift immediately. Keep `.env.example`'s *rendering* (comment-preserving
output, see [`render_env`](Setup_FreeAgent#L494-L511)) but drive its *content*
from the spec.

---

## 5. Phase 1 — Guided web app (no new local software)

A static site (Next/Vite/plain — no backend required) that delivers the
"install, configure, start" story using only guide-and-generate:

1. **Detect OS**; ask the user their RAM (browser can't read it reliably).
2. **Choose a path**: Local Ollama / Remote Ollama / OpenAI — mirroring the
   `L` / `R` quick setups in [`quick_setup`](Setup_FreeAgent#L395-L449) and the
   OpenAI provider added in recent commits (`FA_MAIN_PROVIDER`, `OPENAI_API_KEY`
   in [`fa_proxy.py:186-189`](examples/fa_proxy.py#L186-L189)).
3. **Pick a model** from the shared catalog, pre-highlighting the RAM-fit
   suggestion (§4).
4. **Generate**:
   - a downloadable `.env` (rendered from the spec), and
   - the exact copy-paste bootstrap (`git clone … && ./FreeAgent`, or a hosted
     `curl … | bash`), with a Copy button and a step-by-step checklist.
5. **Start**: instructions to run `./FreeAgent` (with `FA_LAUNCH_CLI=1` to drop
   straight into OpenCode). A `freeagent://` deep link lights up here *if* the
   companion is later installed.

**Outcome:** the "website anywhere that helps install/configure/start" goal, met
honestly and safely, with zero trust or certificate problems. This is also the
exact page the Install button (§2.1) attaches to.

---

## 5A. The minimal online installer (recommended first build)

This is the concrete artifact the web app hands the user, and it is intentionally
the **simplest possible** thing: **macOS + Linux only, no changes to any existing
code, no companion daemon, no certs.** It leans entirely on the fact that
[`FreeAgent`](FreeAgent) is already idempotent and self-installing — the installer
only has to *put the folder down, drop in the user's `.env`, and make a
double-clickable launcher.* Everything else (venv, deps, Ollama, model pull,
OpenCode, config) happens on first run, unchanged.

### 5A.1 Why this needs zero code changes

Two existing behaviors make it free:

1. **`FreeAgent` respects a pre-existing `.env`.** [`ensure_env`](FreeAgent#L217-L229)
   only copies `.env.example → .env` when `.env` is missing. So if the installer
   writes the user's personalized `.env` *before* the first run, the script uses
   it verbatim and never overwrites it.
2. **A minimal `.env` is enough.** The proxy reads every setting with a code-level
   default ([`fa_proxy.py:152-189`](examples/fa_proxy.py#L152-L189)), so the
   `.env` only needs the handful of keys the user actually changed; everything
   else falls back to the same defaults `.env.example` documents.

So the installer is **all new files** (a new `installer/` dir + the web page).
Nothing in [`FreeAgent`](FreeAgent), [`Setup_FreeAgent`](Setup_FreeAgent), or the
proxy is touched.

### 5A.2 The one downloaded file

The web app generates a **single, personalized installer script per OS**, with the
user's `.env` baked in as a heredoc. The user downloads one file and opens it.

- **macOS:** `FreeAgent-Installer.command` — the `.command` extension makes it
  **double-clickable and it opens in Terminal automatically** (a plain `.sh`
  would open in an editor instead). One notarization caveat in §5A.5.
- **Linux:** `freeagent-install.sh` — run it (or serve a tiny `.desktop` that
  does), and it installs a menu entry for subsequent launches.

Skeleton of the generated file (the web app fills in the `.env` block and the
launcher for the detected OS):

```bash
#!/usr/bin/env bash
set -euo pipefail

INSTALL_DIR="$HOME/Applications/FreeAgent"   # user-space → NO sudo, no password

echo "Installing FreeAgent into $INSTALL_DIR …"

# 1. Fetch the repo as a tarball (curl only — no git needed yet; FreeAgent
#    installs git itself on first run if it's missing).
mkdir -p "$INSTALL_DIR"
curl -fsSL https://github.com/Empromptu-ai/FreeAgent/archive/refs/heads/main.tar.gz \
  | tar xz --strip-components=1 -C "$INSTALL_DIR"

# 2. Write the personalized .env the web wizard produced (baked in below).
#    Only the keys the user changed; the rest use FreeAgent's own defaults.
cat > "$INSTALL_DIR/.env" <<'ENV'
FA_LAUNCH_CLI=1
OLLAMA_BASE_URL=http://localhost:11434
FA_MODEL=qwen3.6:35b
# …whatever else the wizard set (provider, remote URL, OpenAI key, …)
ENV

# 3. Create the double-clickable launcher / menu icon (per-OS, see §5A.3).
# 4. Optionally launch it now.
```

Design choices baked into that skeleton, all in service of "as simple as
possible":

- **User-space install dir** (`~/Applications/FreeAgent`) → **never prompts for a
  password.** (Any sudo needed later, e.g. installing Ollama via the OS package
  manager, is surfaced by the existing script on first run, not by the installer.)
- **Tarball via `curl`, not `git clone`** → avoids requiring git at install time
  (curl is effectively always present; FreeAgent installs git later if needed).
- **`.env` baked into the download** → the personalization travels *with* the one
  file; no second download, no config step on the machine.
- **`FA_LAUNCH_CLI=1`** in the baked `.env` → the launcher drops the user straight
  into the OpenCode TUI, matching [`FreeAgent:662-737`](FreeAgent#L662-L737).

### 5A.3 The launcher icon (what "run it from an icon" means here)

**macOS** — the installer drops a `FreeAgent.command` (into `~/Applications` and/or
the Desktop). Its whole body is:

```bash
#!/usr/bin/env bash
cd "$HOME/Applications/FreeAgent" && exec ./FreeAgent
```

Double-clicking it opens Terminal and runs FreeAgent. (For a *real* app icon in
Launchpad/Spotlight rather than a script file, wrap the same command in a minimal
`.app` bundle whose `CFBundleExecutable` is that script + a custom `.icns` — a
nice-to-have, not required for v1.)

**Linux** — the installer writes `~/.local/share/applications/freeagent.desktop`:

```ini
[Desktop Entry]
Name=FreeAgent
Exec=/home/USER/Applications/FreeAgent/FreeAgent
Terminal=true
Icon=/home/USER/Applications/FreeAgent/theme/icon.png
Type=Application
```

`Terminal=true` makes the desktop-menu entry launch it inside the user's terminal
emulator — the icon/menu-item behavior you want, no extra tooling.

### 5A.4 The web-app flow that produces it

The Phase 1 wizard (§5) becomes the installer generator:

1. Detect OS (UA); ask the user their RAM (browser can't read it).
2. Ask the few questions that actually change `.env`: **Local Ollama / Remote
   Ollama / OpenAI**, model (from the shared catalog, §4), remote URL or OpenAI
   key if relevant. Anything unasked stays at the FreeAgent default.
3. Render the `.env` override block from the shared spec (§4).
4. Inline that block into the OS-appropriate installer template and offer it as a
   **single download**. Done.

Because the wizard only emits *overrides*, the question set is short — closest to
"answer three things, download one file, double-click it."

### 5A.5 The honest caveats (still true, but small here)

- **Gatekeeper / notarization (macOS).** An unsigned `.command` downloaded from the
  web gets the quarantine flag; first open shows a right-click-Open prompt. Two
  paths: (a) accept the one-time right-click-Open step and document it, or (b)
  sign + notarize a `.pkg`/`.app` (cost + CI). For a truly minimal v1, (a) is
  acceptable and common.
- **`.env` on re-install/update.** If the user re-downloads with new answers,
  re-running overwrites `.env`. Decide whether the generated script should
  overwrite unconditionally (personalization always wins) or only write `.env`
  when absent (preserve manual edits). Recommended: **write only if `.env` is
  absent, unless the wizard was explicitly used** — a one-line guard in the
  generated script.
- **Windows** is deliberately out of scope here (bash launcher); revisit with WSL
  or a PowerShell port later.

### 5A.6 How it feeds the later phases

This installer is exactly the download the Install button (§2.1) serves, and the
same `~/Applications/FreeAgent` layout is where the Phase 2 companion daemon would
later live. Building it first is self-contained, ships value immediately, and
wastes nothing.

---

## 6. Phase 2 — The companion daemon (the interactive buttons)

This is where the Install/Start buttons, the in-page terminal, and the code
viewer become real. The companion is the proxy plus four new concerns.

### 6.1 Control API (new routes on the FastAPI app)

| Route | Purpose |
|---|---|
| `GET  /health` | liveness (the proxy has **no** health route today; the launcher works around it in [`FreeAgent:687-702`](FreeAgent#L687-L702)) |
| `GET  /status` | install state: ollama? opencode? model pulled? proxy up? codegraph? (reuse the detection in [`find_opencode`](FreeAgent#L362-L371), [`ensure_ollama_installed`](FreeAgent#L241-L352)) |
| `POST /install` | run the [`FreeAgent`](FreeAgent) steps (venv, deps, ollama, pull, opencode, config, tools); **streams progress** via SSE/WS |
| `GET/POST /config` | read/write `.env` through the shared spec (§4) |
| `POST /opencode/start` | spawn OpenCode in a PTY (in-page) **or** in a native Terminal |
| `POST /opencode/stop` | stop it |
| `POST /codegraph/init` | already exists ([`fa_proxy.py:1044`](examples/fa_proxy.py#L1044)) |

`/install` should not reimplement the bash — it should **shell out to the existing
[`FreeAgent`](FreeAgent) script** (or a refactored Python port) and stream its
stdout, so there's one install code path.

### 6.2 PTY WebSocket + terminal + code viewer

- `WS /pty` — spawns `opencode` in a pseudo-terminal (`ptyprocess`/`pexpect` on
  POSIX, ConPTY on Windows), pipes bytes both directions. Front-end is
  **xterm.js**. This is exactly the ttyd/Wetty/VSCode-web pattern. Because
  OpenCode is a TUI, rendering it in xterm.js *is* "OpenCode in the web app."
- **Native Terminal option:** the companion (native side) can instead launch
  Terminal.app / GNOME Terminal / Windows Terminal running the same command the
  launcher uses ([`FreeAgent:734`](FreeAgent#L734)), for users who want a real
  terminal. A `freeagent://open` URL scheme lets a web button deep-link into it.
- **Code viewer:** three tiers —
  1. MVP: just the TUI in xterm.js (already a code UI).
  2. Better: companion serves a file tree + Monaco read/edit view of the project
     dir (the dir resolved by [`resolve_cli_dir`](FreeAgent#L112-L134)).
  3. Richest: run `code-server` (VSCode-in-browser) pointed at the same folder.

### 6.3 Transport: HTTPS/WSS on loopback + reconnect

The proxy binds loopback HTTP today ([`fa_proxy.py:1120-1123`](examples/fa_proxy.py#L1120-L1123)).
For an HTTPS web app to talk to it reliably:

- Register a real domain, e.g. `local.freeagent.dev`, with an **A record →
  127.0.0.1** and a **wildcard TLS cert**. Ship the cert with the companion; it
  serves `https://local.freeagent.dev:PORT`. (This is the Plex/Figma/Tabnine
  pattern; the trade-off — the cert key ships with the client — is well
  understood and accepted for loopback.)
- The web app talks to `https://local.freeagent.dev:PORT` for both HTTPS and WSS,
  sidestepping mixed-content and Local Network Access prompts.
- **Reconnect loop:** after the Install button downloads the installer, the page
  polls `GET /health` on the known port until the companion answers, then flips
  to "Connected" (this is the "push install and it connects" UX from §2.1).

### 6.4 Security model (mandatory — this is a remote page driving local privilege)

A website that can call `/install` and `/pty` on your machine is a large attack
surface. Minimum bar:

- **Pairing token.** On first launch the companion generates a one-time token and
  either (a) opens the browser at `…/pair?token=…`, or (b) prints a short code the
  user types into the page. All control routes require it. No token → 401.
- **Origin allowlist.** CORS + WebSocket `Origin` checks restrict callers to the
  known web-app origin(s). (The proxy has **no CORS today** — must be added, and
  scoped, *not* `*`.)
- **Explicit confirmation** in the companion (native dialog or the paired UI) for
  every privileged action: install, config write, spawning a process.
- **Never execute arbitrary commands** received from the remote. The remote may
  only pick from a fixed set of actions (install/start/stop/config keys from the
  spec). No "run this string" endpoint, ever.
- **Bind loopback only.** Keep `127.0.0.1`; the domain trick resolves there.
- Keep the raw `/v1/chat/completions` path unauthenticated-from-localhost only if
  it stays loopback-bound; the control routes are the sensitive ones.

---

## 7. Phase 3 — Full in-browser IDE

- Replace the raw TUI with Monaco + file tree + diff view driven by the companion.
- Multi-project / multi-session switching (the proxy already keys sessions by
  `x-session-id`, see the session handling around
  [`fa_proxy.py:426-430`](examples/fa_proxy.py#L426-L430)).
- Live status: model in use, tokens, codegraph build progress
  (`GET /codegraph/status` already exists,
  [`fa_proxy.py:1100`](examples/fa_proxy.py#L1100)).
- Optionally self-host the whole UI *from the companion* so it also works fully
  offline, with the hosted site as the entry point / installer distributor.

---

## 8. Build order & deliverables

| Phase | Deliverable | Depends on |
|---|---|---|
| 0 | `spec/freeagent.config.json` + both consumers read it | — |
| 1 | Static wizard: OS detect, config form, `.env` + command generator | 0 |
| **5A** | **Minimal online installer (mac+linux): wizard → personalized one-file download → double-click. No code changes.** | 1 |
| 2a | Companion: `/health`, `/status`, `/config`, CORS, pairing token | 1 |
| 2b | HTTPS-on-loopback cert + web-app reconnect loop | 2a |
| 2c | Install button → signed installers per OS → auto-connect | 2b |
| 2d | `/install` streaming (shells out to `FreeAgent`) | 2a |
| 2e | `WS /pty` + xterm.js; native-terminal + `freeagent://` option | 2b |
| 3 | Monaco/code-server code viewer, multi-session, live status | 2e |

---

## 9. Open questions / risks

- **Installer signing.** macOS notarization + Windows code-signing certs are
  required for a non-scary Install-button download. Cost + CI setup.
- **Windows support.** The current scripts are bash-only; ConPTY and an installer
  story are net-new. Decide if Windows is in scope for v1.
- **Cert distribution.** Shipping a wildcard key in the companion is the accepted
  pattern but must be a cert we're willing to rotate if leaked; plan rotation.
- **`/install` privilege.** Installing Ollama/Homebrew may need sudo
  ([`FreeAgent:165-168`](FreeAgent#L165-L168)); the companion must surface OS
  auth prompts, not swallow them.
- **Proxy hardening.** Adding CORS/pairing must not accidentally expose
  `/v1/chat/completions` or `/install` beyond loopback + paired origin.
```

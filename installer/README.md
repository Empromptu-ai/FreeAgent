# FreeAgent installer (macOS + Linux)

A self-contained, graphical installer. An end user downloads **one file**, opens
it, answers a few questions in native dialogs, and ends up in an OpenCode
session. Nothing outside the install folder is touched (except the app icon).

## Files

| File | Role |
|---|---|
| `install.sh` | The **canonical, self-contained installer**. Downloads the repo, installs the launcher, runs the graphical setup, starts FreeAgent. Also the target of the `curl \| bash` one-liner. |
| `fa-setup-gui.sh` | The **native GUI configurator**. macOS `osascript` / Linux `zenity`\|`kdialog` dialogs that write `.env`. Reuses `Setup_FreeAgent` for RAM detection + the model catalog, so nothing drifts. |
| `fa-launch.sh` | The **launcher gate** behind the app icon: first run → setup; afterwards a little pop-up **[ Reconfigure \| Quit \| Run ]**. |
| `fa-gui-lib.sh` | Shared cross-platform dialog + "open a terminal" helpers (sourced by the above). |
| `make-artifacts.sh` | Stamps the distributable double-click copies from `install.sh`. |

## How it flows

```
Download one file (double-click OR curl | bash)
        │
        ▼
install.sh ── curl|tar repo → ~/Applications/FreeAgent (mac)         ── install FreeAgent.app / .desktop
        │                     ~/.local/share/FreeAgent  (linux)          (icon runs fa-launch.sh)
        ▼
fa-setup-gui.sh  (native dialogs: Local Ollama / Remote / OpenAI; RAM-suggested model)
        │   writes .env  (+ FA_LAUNCH_CLI=1; any prior .env → .env.bak)
        ▼
Terminal: ./FreeAgent → installs Ollama/OpenCode, pulls model → OpenCode TUI

Later, click the icon:
fa-launch.sh → native pop-up [ Reconfigure │ Quit │ Run ]
```

Only two steps use a real terminal — they must: FreeAgent's install/pull
progress and the OpenCode TUI itself. All configuration and the run/reconfigure
choice are native dialogs.

## Reusing Setup_FreeAgent (no logic duplication)

The GUI does **not** hard-code a model list. It shells out to the existing
`Setup_FreeAgent`, which grew three non-interactive query flags:

```
./Setup_FreeAgent --detect-ram      # -> 32   (whole-GB RAM, or nothing)
./Setup_FreeAgent --list-models     # -> "qwen3.5:2b<TAB>4" … one per model
./Setup_FreeAgent --suggest-model   # -> the recommended tag for this machine
```

Edit the model catalog in one place (`Setup_FreeAgent`) and both the TUI and the
GUI follow.

## Building the distributable files

```sh
./make-artifacts.sh              # writes ./dist/
#   FreeAgent-Installer.command  (macOS double-click → Terminal)
#   freeagent-install.sh         (Linux)
#   install.sh                   (host for the curl one-liner)
```

Host `install.sh` somewhere and the one-liner is:

```sh
curl -fsSL https://<host>/install.sh | bash
```

## Try it locally (no hosting)

```sh
# Point at a local checkout instead of downloading, and don't auto-launch:
FA_REPO_URL="file://$PWD/.." FA_BRANCH=... FA_LAUNCH_AFTER=0 ./install.sh
# or just exercise the GUI setup against this checkout:
FA_INSTALL_DIR="$PWD/.." ./fa-setup-gui.sh
```

Runtime overrides: `FA_INSTALL_DIR`, `FA_REPO_URL`, `FA_BRANCH`,
`FA_LAUNCH_AFTER`, `FA_FORCE_SETUP`.

## Caveats

- **macOS Gatekeeper.** An unsigned `.command` downloaded from the web is
  quarantined; the **first** open needs right-click → *Open* (one time). Every
  launch after that is a normal double-click. A warning-free first run means
  signing + notarizing a `.pkg` (Apple Developer account + CI) — deferred. The
  `curl | bash` one-liner sidesteps this entirely, which is why both are offered.
- **Linux dialogs** need `zenity` or `kdialog`; without either (or on a headless
  box) the GUI setup falls back to the text `Setup_FreeAgent` TUI in a terminal.
- **Re-install** overwrites the repo code but preserves `venv/` and `.env` (they
  aren't in the tarball). Setup is skipped when a `.env` already exists unless
  `FA_FORCE_SETUP=1`; the GUI backs up any `.env` to `.env.bak` before writing.
- **Windows** is out of scope (bash launcher).

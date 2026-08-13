#!/usr/bin/env bash
#
# FreeAgent installer  (macOS + Linux)  —  self-contained, no prerequisites.
# =========================================================================
# This ONE file is all an end user needs. Double-click it (macOS: the
# FreeAgent-Installer.command copy; Linux: freeagent-install.sh) or run it via
#   curl -fsSL <hosted-url>/install.sh | bash
# and it will:
#
#   1. download the FreeAgent repo into a standard user-space folder (no sudo,
#      curl + tar only — git is not required),
#   2. install a launcher: a real FreeAgent.app on macOS / a menu entry on Linux,
#      both of which run the little "Reconfigure / Run" pop-up (installer/fa-launch.sh),
#   3. run the graphical first-run setup (installer/fa-setup-gui.sh), and
#   4. start FreeAgent, whose own first-run logic installs Ollama/OpenCode,
#      pulls the model, and opens the OpenCode TUI.
#
# Nothing outside the install folder is modified (except the launcher icon).
#
# Override before running if you like:
#   FA_INSTALL_DIR   where to install   (default ~/Applications/FreeAgent [mac]
#                                        or ~/.local/share/FreeAgent [linux])
#   FA_REPO_URL      git repo tarball source  (default the Empromptu repo)
#   FA_BRANCH        branch to fetch          (default main)
#   FA_LAUNCH_AFTER  1 = start FreeAgent when done, 0 = just install (default 1)
#   FA_FORCE_SETUP   1 = re-run setup even if a .env already exists (default 0)

set -euo pipefail

OS="$(uname -s)"
FA_REPO_URL="${FA_REPO_URL:-https://github.com/Empromptu-ai/FreeAgent}"
# FA_BRANCH="${FA_BRANCH:-main}"
FA_BRANCH="${FA_BRANCH:-main}"
FA_LAUNCH_AFTER="${FA_LAUNCH_AFTER:-1}"
FA_FORCE_SETUP="${FA_FORCE_SETUP:-0}"

case "$OS" in
  Darwin) FA_INSTALL_DIR="${FA_INSTALL_DIR:-$HOME/Applications/FreeAgent}" ;;
  Linux)  FA_INSTALL_DIR="${FA_INSTALL_DIR:-$HOME/.local/share/FreeAgent}" ;;
  *)      FA_INSTALL_DIR="${FA_INSTALL_DIR:-$HOME/FreeAgent}" ;;
esac
TARBALL_URL="$FA_REPO_URL/archive/refs/heads/$FA_BRANCH.tar.gz"

say()  { printf '\033[1;35m[FreeAgent]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[FreeAgent] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. fetch the repo tarball. Extracting over an existing install keeps venv/
#    and any prior .env in place (neither is in the tarball).
# ---------------------------------------------------------------------------
command -v curl >/dev/null 2>&1 || die "curl is required but was not found."
command -v tar  >/dev/null 2>&1 || die "tar is required but was not found."

say "Installing into: $FA_INSTALL_DIR"
mkdir -p "$FA_INSTALL_DIR"

say "Downloading FreeAgent ($FA_BRANCH) …"
curl -fsSL "$TARBALL_URL" | tar -xz --strip-components 1 -C "$FA_INSTALL_DIR" \
  || die "download/extract failed from $TARBALL_URL (check your network and that the branch exists)."
[ -f "$FA_INSTALL_DIR/FreeAgent" ] || die "FreeAgent script missing after extract — is $FA_REPO_URL correct?"

chmod +x "$FA_INSTALL_DIR/FreeAgent" "$FA_INSTALL_DIR/Setup_FreeAgent" 2>/dev/null || true
chmod +x "$FA_INSTALL_DIR"/installer/*.sh 2>/dev/null || true

GATE="$FA_INSTALL_DIR/installer/fa-launch.sh"
SETUP_GUI="$FA_INSTALL_DIR/installer/fa-setup-gui.sh"
# shellcheck source=fa-gui-lib.sh
. "$FA_INSTALL_DIR/installer/fa-gui-lib.sh"

# ---------------------------------------------------------------------------
# 2. install the launcher / icon (it runs the gate script, not FreeAgent
#    directly, so users always get the Reconfigure/Run pop-up).
# ---------------------------------------------------------------------------
make_launcher_macos() {
  local apps="$HOME/Applications"; mkdir -p "$apps"
  local app="$apps/FreeAgent.app"
  rm -rf "$app"
  mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
  cat > "$app/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>FreeAgent</string>
  <key>CFBundleDisplayName</key><string>FreeAgent</string>
  <key>CFBundleIdentifier</key><string>ai.empromptu.freeagent</string>
  <key>CFBundleVersion</key><string>1.0</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleExecutable</key><string>FreeAgent</string>
  <key>CFBundleIconFile</key><string>FreeAgent.icns</string>
</dict>
</plist>
PLIST
  cat > "$app/Contents/MacOS/FreeAgent" <<LAUNCH
#!/bin/bash
exec /bin/bash "$GATE"
LAUNCH
  chmod +x "$app/Contents/MacOS/FreeAgent"
  [ -f "$FA_INSTALL_DIR/theme/FreeAgent.icns" ] && \
    cp "$FA_INSTALL_DIR/theme/FreeAgent.icns" "$app/Contents/Resources/FreeAgent.icns" 2>/dev/null || true
  say "Installed app: $app  (find it in Launchpad / Spotlight)"
}

make_launcher_linux() {
  local appsdir="$HOME/.local/share/applications"; mkdir -p "$appsdir"
  local desktop="$appsdir/freeagent.desktop"
  local icon_line=""
  [ -f "$FA_INSTALL_DIR/theme/icon.png" ] && icon_line="Icon=$FA_INSTALL_DIR/theme/icon.png"
  cat > "$desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=FreeAgent
Comment=Free, local, private coding agent (OpenCode + FreeAgent proxy)
Exec=/bin/bash "$GATE"
Path=$FA_INSTALL_DIR
Terminal=false
Categories=Development;
$icon_line
DESKTOP
  chmod +x "$desktop" 2>/dev/null || true
  command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$appsdir" >/dev/null 2>&1 || true
  say "Installed menu entry: look for 'FreeAgent' in your applications menu"
}

case "$OS" in
  Darwin) make_launcher_macos ;;
  Linux)  make_launcher_linux ;;
  *)      say "Unsupported OS '$OS' — skipping launcher (you can still run $FA_INSTALL_DIR/FreeAgent)" ;;
esac

# ---------------------------------------------------------------------------
# 3. graphical first-run setup (skip on a re-install that already has a .env,
#    unless FA_FORCE_SETUP=1).
# ---------------------------------------------------------------------------
if [ ! -f "$FA_INSTALL_DIR/.env" ] || [ "$FA_FORCE_SETUP" = "1" ]; then
  say "Opening graphical setup …"
  FA_INSTALL_DIR="$FA_INSTALL_DIR" /bin/bash "$SETUP_GUI" || say "Setup was cancelled — you can re-run it any time from the FreeAgent icon."
else
  say "Existing configuration found (.env) — keeping it. Use the icon → Reconfigure to change it."
fi

# ---------------------------------------------------------------------------
# 4. launch. If we own a real terminal (double-clicked .command), run inline so
#    it all stays in one window; otherwise (curl | bash) open a fresh terminal.
# ---------------------------------------------------------------------------
if [ "$FA_LAUNCH_AFTER" = "1" ] && [ -f "$FA_INSTALL_DIR/.env" ]; then
  say "Starting FreeAgent (first run installs Ollama/OpenCode and pulls the model — this can take a while) …"
  if [ -t 0 ] && [ -t 1 ]; then
    cd "$FA_INSTALL_DIR"; exec ./FreeAgent
  else
    open_terminal_run "$FA_INSTALL_DIR" "./FreeAgent"
    say "FreeAgent is starting in a new terminal window."
  fi
else
  say "Done. Launch FreeAgent any time from its icon, or run: $FA_INSTALL_DIR/FreeAgent"
fi

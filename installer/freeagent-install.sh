#!/usr/bin/env bash
#
# FreeAgent minimal online installer  (macOS + Linux)
# ---------------------------------------------------
# Downloading this one file and opening it installs FreeAgent into a standard
# user-space folder, drops in the .env below, and creates a double-clickable
# launcher / menu icon that runs ./FreeAgent (which then does the rest on first
# run — venv, deps, Ollama, model pull, OpenCode, config — all unchanged).
#
# This script makes NO changes to the FreeAgent repo. It only:
#   1. fetches the repo tarball into ~/Applications/FreeAgent   (no sudo)
#   2. writes the personalized .env baked in below
#   3. creates a launcher (macOS .app + .command, or a Linux .desktop)
#   4. optionally launches FreeAgent right away
#
# It is also a TEMPLATE: the web wizard swaps the FA_ENV_BLOCK lines (see below)
# for the user's chosen settings and serves the result as a single download
# (FreeAgent-Installer.command on macOS, freeagent-install.sh on Linux). Run
# installer/generate-installer.sh to produce a personalized copy locally.
#
# Override any of these before running if you like:
#   FA_INSTALL_DIR   where to install            (default ~/Applications/FreeAgent)
#   FA_REPO_URL      git repo (for the tarball)  (default the Empromptu repo)
#   FA_BRANCH        branch to fetch             (default main)
#   FA_LAUNCH_AFTER  1 = start FreeAgent when done, 0 = just install (default 1)

set -euo pipefail

FA_INSTALL_DIR="${FA_INSTALL_DIR:-$HOME/Applications/FreeAgent}"
FA_REPO_URL="${FA_REPO_URL:-https://github.com/Empromptu-ai/FreeAgent}"
FA_BRANCH="${FA_BRANCH:-main}"
FA_LAUNCH_AFTER="${FA_LAUNCH_AFTER:-1}"

TARBALL_URL="$FA_REPO_URL/archive/refs/heads/$FA_BRANCH.tar.gz"

say()  { printf '\033[1;35m[FreeAgent]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[FreeAgent] WARNING:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[FreeAgent] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

OS="$(uname -s)"

# ---------------------------------------------------------------------------
# 1. fetch the repo tarball (curl only — no git needed yet; FreeAgent installs
#    git itself on first run if it's missing). Extracting over an existing dir
#    keeps venv/ and any prior .env in place (they aren't in the tarball).
# ---------------------------------------------------------------------------
command -v curl >/dev/null 2>&1 || die "curl is required but was not found."
command -v tar  >/dev/null 2>&1 || die "tar is required but was not found."

say "Installing into: $FA_INSTALL_DIR"
mkdir -p "$FA_INSTALL_DIR"

say "Downloading FreeAgent ($FA_BRANCH) …"
if ! curl -fsSL "$TARBALL_URL" | tar -xz --strip-components 1 -C "$FA_INSTALL_DIR"; then
  die "download/extract failed from $TARBALL_URL (check your network and that the branch exists)."
fi
[ -f "$FA_INSTALL_DIR/FreeAgent" ] || die "FreeAgent script not found after extract — is $FA_REPO_URL correct?"
chmod +x "$FA_INSTALL_DIR/FreeAgent" "$FA_INSTALL_DIR/Setup_FreeAgent" 2>/dev/null || true

# ---------------------------------------------------------------------------
# 2. write the personalized .env. If one already exists (e.g. a re-install with
#    hand edits), back it up to .env.bak first — the wizard's choices win, but
#    nothing is lost. FreeAgent's own ensure_env then leaves this file untouched.
# ---------------------------------------------------------------------------
ENV_DEST="$FA_INSTALL_DIR/.env"
if [ -f "$ENV_DEST" ]; then
  cp "$ENV_DEST" "$ENV_DEST.bak"
  say "Existing .env backed up to .env.bak"
fi

cat > "$ENV_DEST" <<'FA_ENV_EOF'
# >>> FA_ENV_BLOCK (personalized by the web wizard; edit freely) >>>
# Only the settings you changed need to be here — everything else falls back to
# FreeAgent's built-in defaults. FA_LAUNCH_CLI=1 makes the launcher drop you
# straight into the OpenCode TUI.
FA_LAUNCH_CLI=1
# <<< FA_ENV_BLOCK <<<
FA_ENV_EOF
say "Wrote $ENV_DEST"

# ---------------------------------------------------------------------------
# 3. create the launcher / icon
# ---------------------------------------------------------------------------
make_launcher_macos() {
  local apps="$HOME/Applications"
  mkdir -p "$apps"

  # 3a. A minimal .app bundle → a real Launchpad/Spotlight icon. Because the
  #     FreeAgent CLI needs a TTY, the app opens Terminal on the script rather
  #     than running headless.
  local app="$apps/FreeAgent.app"
  rm -rf "$app"
  mkdir -p "$app/Contents/MacOS"
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
DIR="$FA_INSTALL_DIR"
osascript \\
  -e "tell application \\"Terminal\\" to do script \\"cd '\$DIR' && ./FreeAgent\\"" \\
  -e 'tell application "Terminal" to activate'
LAUNCH
  chmod +x "$app/Contents/MacOS/FreeAgent"
  # Use the bundled .icns if the repo ever ships one; harmless if absent.
  [ -f "$FA_INSTALL_DIR/theme/FreeAgent.icns" ] && cp "$FA_INSTALL_DIR/theme/FreeAgent.icns" "$app/Contents/Resources/FreeAgent.icns" 2>/dev/null || true
  say "Created app: $app"

  # 3b. A plain double-clickable .command as a simple fallback (opens Terminal
  #     directly). Placed in the install dir; copied to the Desktop if present.
  local cmd="$FA_INSTALL_DIR/Launch FreeAgent.command"
  cat > "$cmd" <<LAUNCH
#!/bin/bash
cd "$FA_INSTALL_DIR" && exec ./FreeAgent
LAUNCH
  chmod +x "$cmd"
  if [ -d "$HOME/Desktop" ]; then
    cp "$cmd" "$HOME/Desktop/FreeAgent.command" && chmod +x "$HOME/Desktop/FreeAgent.command"
    say "Put a launcher on your Desktop: FreeAgent.command"
  fi
}

make_launcher_linux() {
  local appsdir="$HOME/.local/share/applications"
  mkdir -p "$appsdir"
  local desktop="$appsdir/freeagent.desktop"
  local icon_line=""
  # Reference an icon only if one is actually present, else fall back to the
  # generic app icon (no broken-image entry).
  if [ -f "$FA_INSTALL_DIR/theme/icon.png" ]; then
    icon_line="Icon=$FA_INSTALL_DIR/theme/icon.png"
  fi
  cat > "$desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=FreeAgent
Comment=Free, local, private coding agent (OpenCode + FreeAgent proxy)
Exec=$FA_INSTALL_DIR/FreeAgent
Path=$FA_INSTALL_DIR
Terminal=true
Categories=Development;
$icon_line
DESKTOP
  chmod +x "$desktop" 2>/dev/null || true
  command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$appsdir" >/dev/null 2>&1 || true
  say "Created menu entry: $desktop  (look for 'FreeAgent' in your app menu)"
}

case "$OS" in
  Darwin) make_launcher_macos ;;
  Linux)  make_launcher_linux ;;
  *)      warn "Unsupported OS '$OS' — skipping launcher. You can still run: $FA_INSTALL_DIR/FreeAgent" ;;
esac

# ---------------------------------------------------------------------------
# 4. optionally launch now
# ---------------------------------------------------------------------------
say "Done. FreeAgent is installed in $FA_INSTALL_DIR"
if [ "$FA_LAUNCH_AFTER" = "1" ]; then
  say "Starting FreeAgent now (first run installs Ollama/OpenCode and pulls the model — this can take a while) …"
  cd "$FA_INSTALL_DIR"
  exec ./FreeAgent
else
  say "To start it later, use the launcher icon, or run: $FA_INSTALL_DIR/FreeAgent"
fi

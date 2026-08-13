#!/usr/bin/env bash
#
# fa-launch.sh — the FreeAgent launcher "gate" behind the app icon.
#
# This is what the macOS FreeAgent.app and the Linux freeagent.desktop entry
# actually run. It is deliberately tiny:
#
#   * First run (no .env yet)  -> open the graphical setup, then start FreeAgent.
#   * Every run after that     -> a little native pop-up:
#         [ Reconfigure ]  [ Quit ]  [ Run ]
#       Run         -> start FreeAgent as configured.
#       Reconfigure -> open the graphical setup, then start FreeAgent.
#       Quit        -> do nothing.
#
# Both the FreeAgent install/pull progress and the OpenCode TUI need a real
# terminal, so "start FreeAgent" always means "open a terminal running
# ./FreeAgent" rather than running it headless.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=fa-gui-lib.sh
. "$HERE/fa-gui-lib.sh"

ROOT_DIR="${FA_INSTALL_DIR:-$(cd "$HERE/.." && pwd)}"
ENV_FILE="$ROOT_DIR/.env"
SETUP_GUI="$HERE/fa-setup-gui.sh"

start_freeagent() { open_terminal_run "$ROOT_DIR" "./FreeAgent"; }
run_setup()       { FA_INSTALL_DIR="$ROOT_DIR" bash "$SETUP_GUI"; }

# First run: no configuration yet -> go straight through setup, then launch.
if [ ! -f "$ENV_FILE" ]; then
  run_setup
  [ -f "$ENV_FILE" ] && start_freeagent
  exit 0
fi

# Already configured: offer the three-way choice.
choice="$(gui_three_buttons \
  "FreeAgent is ready. What would you like to do?" \
  "Reconfigure" "Quit" "Run")"

case "$choice" in
  Run)         start_freeagent ;;
  Reconfigure) run_setup; [ -f "$ENV_FILE" ] && start_freeagent ;;
  *)           : ;;   # Quit / cancel
esac

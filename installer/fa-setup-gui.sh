#!/usr/bin/env bash
#
# fa-setup-gui.sh — native, graphical FreeAgent configurator.
#
# A dialog-driven front-end that produces the same .env the interactive
# Setup_FreeAgent TUI would, but with native OS dialogs (macOS osascript /
# Linux zenity|kdialog). It reuses Setup_FreeAgent as the single source of truth
# for RAM detection and the model catalog via its --detect-ram / --list-models /
# --suggest-model query flags, so the two never drift.
#
# It writes only the settings the user chose — everything else falls back to
# FreeAgent's built-in defaults — plus FA_LAUNCH_CLI=1 so the launcher drops
# straight into the OpenCode TUI. Any existing .env is backed up to .env.bak.
#
# If no graphical toolkit is available (headless Linux, no zenity/kdialog), it
# falls back to opening the text-mode Setup_FreeAgent TUI in a terminal.
#
# Usage:  fa-setup-gui.sh            configure the repo this script lives in
#         FA_INSTALL_DIR=... fa-setup-gui.sh   configure a specific install dir

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=fa-gui-lib.sh
. "$HERE/fa-gui-lib.sh"

ROOT_DIR="${FA_INSTALL_DIR:-$(cd "$HERE/.." && pwd)}"
SETUP="$ROOT_DIR/Setup_FreeAgent"
ENV_FILE="$ROOT_DIR/.env"

[ -x "$SETUP" ] || { gui_error "Setup_FreeAgent not found in $ROOT_DIR — is the install complete?"; exit 1; }

# ---- fallback: no GUI toolkit -> the text TUI in a terminal ------------------
if ! gui_available; then
  open_terminal_run "$ROOT_DIR" "./Setup_FreeAgent"
  exit 0
fi

# ---- write the chosen overrides to .env -------------------------------------
# Takes KEY=VALUE arguments. Backs up any existing .env, then writes a small,
# self-documenting overrides file. FA_LAUNCH_CLI=1 is always included.
write_env() {
  if [ -f "$ENV_FILE" ]; then
    cp "$ENV_FILE" "$ENV_FILE.bak"
  fi
  {
    printf '# FreeAgent settings — written by the graphical setup.\n'
    printf '# Only the values you chose are here; everything else uses FreeAgent\x27s\n'
    printf '# built-in defaults. Re-run setup any time, or edit this file by hand.\n\n'
    local kv
    for kv in "$@"; do printf '%s\n' "$kv"; done
    printf 'FA_LAUNCH_CLI=1\n'
  } > "$ENV_FILE"
}

# ---- provider choice --------------------------------------------------------
provider="$(gui_choose "$FA_TITLE" \
  "How do you want FreeAgent to run your model?" \
  "Local Ollama (runs models on this computer)" \
  "Remote Ollama (a server you already have)" \
  "OpenAI (use your OpenAI API key)")" || exit 0

case "$provider" in
  Local*)
    # Reuse Setup_FreeAgent's catalog + RAM suggestion. Present the recommended
    # model first, then the rest of the catalog, then a "type your own" escape.
    ram="$("$SETUP" --detect-ram 2>/dev/null || true)"
    suggested="$("$SETUP" --suggest-model 2>/dev/null || echo "qwen3.6:35b")"

    MANUAL="Type a different model name…"
    options=(); rec_label=""
    # Recommended entry first so it's the highlighted default.
    rec_label="$suggested (~recommended for this machine)"
    options+=("$rec_label")
    # Then every catalog entry (skipping the suggested one, already on top).
    while IFS=$'\t' read -r tag gb; do
      [ -z "$tag" ] && continue
      [ "$tag" = "$suggested" ] && continue
      options+=("$tag (~$gb GB)")
    done < <("$SETUP" --list-models 2>/dev/null)
    options+=("$MANUAL")

    if [ -n "$ram" ]; then ram_note="Detected about ${ram} GB of RAM. "; else ram_note=""; fi
    pick="$(gui_choose "$FA_TITLE" \
      "${ram_note}Choose a model to run locally (the recommended one fits comfortably):" \
      "${options[@]}")" || exit 0

    if [ "$pick" = "$MANUAL" ]; then
      model="$(gui_input "$FA_TITLE" "Enter an Ollama model tag (e.g. qwen3.6:35b):" "$suggested")" || exit 0
      [ -z "$model" ] && model="$suggested"
    elif [ "$pick" = "$rec_label" ]; then
      model="$suggested"
    else
      model="${pick%% (*}"    # strip the " (~N GB)" annotation back to the tag
    fi

    write_env "OLLAMA_BASE_URL=http://localhost:11434" "FA_MODEL=$model"
    gui_info "Local Ollama configured with model: $model"
    ;;

  Remote*)
    url="$(gui_input "$FA_TITLE" "Address of your Ollama server:" "http://192.168.1.50:11434")" || exit 0
    [ -z "$url" ] && url="http://192.168.1.50:11434"
    model="$(gui_input "$FA_TITLE" "Model to use (must already be pulled on that server):" "qwen3.6:35b")" || exit 0
    [ -z "$model" ] && model="qwen3.6:35b"
    write_env "OLLAMA_BASE_URL=$url" "FA_MODEL=$model"
    gui_info "Remote Ollama configured:\n$url\nModel: $model"
    ;;

  OpenAI*)
    key="$(gui_password "$FA_TITLE" "Paste your OpenAI API key (starts with sk-):")" || exit 0
    if [ -z "$key" ]; then gui_error "No API key entered — setup cancelled."; exit 0; fi
    model="$(gui_input "$FA_TITLE" "OpenAI model id to use:" "gpt-4o")" || exit 0
    [ -z "$model" ] && model="gpt-4o"
    # OpenAI for the agent + summary surfaces; skip the local Ollama install
    # entirely (codegraph embeddings, which are Ollama-only, are disabled).
    write_env \
      "FA_MAIN_PROVIDER=openai" \
      "FA_SUMM_PROVIDER=openai" \
      "FA_SKIP_OLLAMA=1" \
      "OPENAI_API_KEY=$key" \
      "FA_MODEL=$model"
    gui_info "OpenAI configured with model: $model"
    ;;

  *) exit 0 ;;
esac

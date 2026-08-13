#!/usr/bin/env bash
#
# fa-gui-lib.sh — shared native-dialog + terminal helpers for the FreeAgent
# installer, GUI configurator, and run-time launcher gate.
#
# This is *sourced*, never executed directly. It provides a tiny cross-platform
# GUI toolkit so the rest of the installer never has to care whether it is on
# macOS (osascript) or Linux (zenity / kdialog). Every helper prints its result
# to stdout and returns non-zero when the user cancels, so callers can write:
#
#     choice="$(gui_choose "Title" "Prompt" "Local" "Remote")" || exit 0
#
# Nothing here writes files or touches .env — presentation only.

# ---- platform ---------------------------------------------------------------
FA_OS="$(uname -s)"

# Which Linux dialog backend is available (set on first use). Empty = none, in
# which case Linux callers fall back to the text-mode TUI.
_FA_LINUX_GUI=""
_fa_linux_gui() {
  [ -n "$_FA_LINUX_GUI" ] && { printf '%s' "$_FA_LINUX_GUI"; return 0; }
  if   command -v zenity  >/dev/null 2>&1; then _FA_LINUX_GUI="zenity"
  elif command -v kdialog >/dev/null 2>&1; then _FA_LINUX_GUI="kdialog"
  else _FA_LINUX_GUI="none"; fi
  printf '%s' "$_FA_LINUX_GUI"
}

# True if a real graphical dialog toolkit is available on this machine.
gui_available() {
  case "$FA_OS" in
    Darwin) command -v osascript >/dev/null 2>&1 ;;
    Linux)  [ "$(_fa_linux_gui)" != "none" ] && [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ] ;;
    *)      return 1 ;;
  esac
}

FA_TITLE="FreeAgent"

# osascript strings can't contain a raw double-quote or backslash; escape both.
_fa_osa_escape() { printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g'; }

# ---- info / error -----------------------------------------------------------
gui_info() {  # message
  local msg="$1"
  case "$FA_OS" in
    Darwin) osascript -e "display dialog \"$(_fa_osa_escape "$msg")\" with title \"$FA_TITLE\" buttons {\"OK\"} default button \"OK\"" >/dev/null 2>&1 || true ;;
    Linux)
      case "$(_fa_linux_gui)" in
        zenity)  zenity  --info --title="$FA_TITLE" --text="$msg" >/dev/null 2>&1 || true ;;
        kdialog) kdialog --title "$FA_TITLE" --msgbox "$msg"       >/dev/null 2>&1 || true ;;
        *)       printf '[FreeAgent] %s\n' "$msg" ;;
      esac ;;
    *) printf '[FreeAgent] %s\n' "$msg" ;;
  esac
}

gui_error() {  # message
  local msg="$1"
  case "$FA_OS" in
    Darwin) osascript -e "display dialog \"$(_fa_osa_escape "$msg")\" with title \"$FA_TITLE\" buttons {\"OK\"} default button \"OK\" with icon stop" >/dev/null 2>&1 || true ;;
    Linux)
      case "$(_fa_linux_gui)" in
        zenity)  zenity  --error --title="$FA_TITLE" --text="$msg" >/dev/null 2>&1 || true ;;
        kdialog) kdialog --title "$FA_TITLE" --error "$msg"         >/dev/null 2>&1 || true ;;
        *)       printf '[FreeAgent] ERROR: %s\n' "$msg" >&2 ;;
      esac ;;
    *) printf '[FreeAgent] ERROR: %s\n' "$msg" >&2 ;;
  esac
}

# ---- single-choice list -----------------------------------------------------
# gui_choose TITLE PROMPT OPTION... -> prints the chosen option (first option is
# the default/pre-selected one); returns 1 on cancel.
gui_choose() {
  local prompt_title="$1" prompt="$2"; shift 2
  local first="$1"
  case "$FA_OS" in
    Darwin)
      local items="" o
      for o in "$@"; do items="$items, \"$(_fa_osa_escape "$o")\""; done
      items="{${items#, }}"
      osascript \
        -e "set choice to choose from list $items with title \"$FA_TITLE\" with prompt \"$(_fa_osa_escape "$prompt")\" default items {\"$(_fa_osa_escape "$first")\"}" \
        -e 'if choice is false then error number -128' \
        -e 'item 1 of choice' 2>/dev/null || return 1
      ;;
    Linux)
      case "$(_fa_linux_gui)" in
        zenity)
          local args=() o
          for o in "$@"; do args+=("$o"); done
          zenity --list --title="$FA_TITLE" --text="$prompt" \
                 --column="Option" "${args[@]}" 2>/dev/null || return 1
          ;;
        kdialog)
          local args=() o
          for o in "$@"; do args+=("$o" "$o"); done
          kdialog --title "$FA_TITLE" --menu "$prompt" "${args[@]}" 2>/dev/null || return 1
          ;;
        *) return 2 ;;   # signal: no GUI, caller should use text fallback
      esac ;;
    *) return 2 ;;
  esac
}

# ---- free-text / password input --------------------------------------------
gui_input() {  # TITLE PROMPT [DEFAULT] -> prints entered text; 1 on cancel
  local prompt="$2" def="${3:-}"
  case "$FA_OS" in
    Darwin)
      osascript \
        -e "set r to display dialog \"$(_fa_osa_escape "$prompt")\" with title \"$FA_TITLE\" default answer \"$(_fa_osa_escape "$def")\"" \
        -e 'text returned of r' 2>/dev/null || return 1
      ;;
    Linux)
      case "$(_fa_linux_gui)" in
        zenity)  zenity  --entry --title="$FA_TITLE" --text="$prompt" --entry-text="$def" 2>/dev/null || return 1 ;;
        kdialog) kdialog --title "$FA_TITLE" --inputbox "$prompt" "$def" 2>/dev/null || return 1 ;;
        *) return 2 ;;
      esac ;;
    *) return 2 ;;
  esac
}

gui_password() {  # TITLE PROMPT -> prints entered secret; 1 on cancel
  local prompt="$2"
  case "$FA_OS" in
    Darwin)
      osascript \
        -e "set r to display dialog \"$(_fa_osa_escape "$prompt")\" with title \"$FA_TITLE\" default answer \"\" with hidden answer" \
        -e 'text returned of r' 2>/dev/null || return 1
      ;;
    Linux)
      case "$(_fa_linux_gui)" in
        zenity)  zenity  --password --title="$FA_TITLE" 2>/dev/null || return 1 ;;
        kdialog) kdialog --title "$FA_TITLE" --password "$prompt" 2>/dev/null || return 1 ;;
        *) return 2 ;;
      esac ;;
    *) return 2 ;;
  esac
}

# ---- three-way button prompt (the run-time gate) ----------------------------
# gui_three_buttons PROMPT LEFT MIDDLE RIGHT -> prints the chosen label.
# RIGHT is the default/affirmative button. Returns 1 only on a hard failure.
gui_three_buttons() {
  local prompt="$1" left="$2" middle="$3" right="$4"
  case "$FA_OS" in
    Darwin)
      osascript \
        -e "set r to display dialog \"$(_fa_osa_escape "$prompt")\" with title \"$FA_TITLE\" buttons {\"$(_fa_osa_escape "$left")\", \"$(_fa_osa_escape "$middle")\", \"$(_fa_osa_escape "$right")\"} default button \"$(_fa_osa_escape "$right")\"" \
        -e 'button returned of r' 2>/dev/null || { printf '%s' "$middle"; return 0; }
      ;;
    Linux)
      case "$(_fa_linux_gui)" in
        zenity)
          # OK=right(0), extra=left(prints label,1), cancel=middle(empty,1).
          local out rc
          out="$(zenity --question --title="$FA_TITLE" --text="$prompt" \
                        --ok-label="$right" --cancel-label="$middle" \
                        --extra-button="$left" 2>/dev/null)"; rc=$?
          if [ -n "$out" ]; then printf '%s' "$left"
          elif [ "$rc" -eq 0 ]; then printf '%s' "$right"
          else printf '%s' "$middle"; fi
          ;;
        kdialog)
          # yes=right(0), no=left(1), cancel=middle(2).
          kdialog --title "$FA_TITLE" --yesnocancel "$prompt" \
                  --yes-label "$right" --no-label "$left" --cancel-label "$middle" >/dev/null 2>&1
          case $? in 0) printf '%s' "$right" ;; 1) printf '%s' "$left" ;; *) printf '%s' "$middle" ;; esac
          ;;
        *) printf '%s' "$middle" ;;
      esac ;;
    *) printf '%s' "$middle" ;;
  esac
}

# ---- open a terminal and run a command --------------------------------------
# Both ./FreeAgent's install progress and the OpenCode TUI need a real TTY, so
# GUI-launched flows must hand off to a terminal emulator rather than run
# headless. open_terminal_run DIR "command" starts `command` inside DIR in a new
# terminal window and returns immediately.
open_terminal_run() {  # DIR COMMAND
  local dir="$1" cmd="$2"
  case "$FA_OS" in
    Darwin)
      osascript \
        -e "tell application \"Terminal\" to do script \"cd $(_fa_osa_escape "$(printf '%q' "$dir")") && $cmd\"" \
        -e 'tell application "Terminal" to activate' >/dev/null 2>&1
      ;;
    Linux)
      local run="cd $(printf '%q' "$dir") && $cmd; echo; echo '[FreeAgent] session ended — press Enter to close.'; read _"
      local t
      for t in x-terminal-emulator gnome-terminal konsole xfce4-terminal mate-terminal kitty alacritty xterm; do
        if command -v "$t" >/dev/null 2>&1; then
          case "$t" in
            gnome-terminal|xfce4-terminal|mate-terminal) "$t" -- bash -lc "$run" >/dev/null 2>&1 & ;;
            konsole)                                     "$t" -e bash -lc "$run" >/dev/null 2>&1 & ;;
            *)                                           "$t" -e bash -lc "$run" >/dev/null 2>&1 & ;;
          esac
          return 0
        fi
      done
      gui_error "No terminal emulator found. Open a terminal and run:  cd '$dir' && $cmd"
      return 1
      ;;
    *) ( cd "$dir" && eval "$cmd" ) ;;
  esac
}

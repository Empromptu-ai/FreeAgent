#!/usr/bin/env bash
#
# generate-installer.sh — produce a personalized FreeAgent installer.
#
# This is exactly what the web wizard does server-side: take the user's chosen
# .env settings and bake them into the installer template (freeagent-install.sh),
# emitting a single downloadable file. Provided here so the whole flow is
# testable locally, with no web app.
#
# Usage:
#   ./generate-installer.sh [--env FILE] [--os mac|linux] [--out FILE]
#
#   --env FILE   file of KEY=value overrides to bake in (default: just
#                FA_LAUNCH_CLI=1). Comments/blank lines are ignored; only
#                KEY=value lines are kept. FA_LAUNCH_CLI=1 is added if absent.
#   --os         only affects the output filename/extension (the installer
#                itself detects the OS at runtime):
#                  mac   -> FreeAgent-Installer.command   (double-click → Terminal)
#                  linux -> freeagent-install.sh
#                Default: this machine's OS (uname).
#   --out FILE   explicit output path (overrides the --os default name).
#
# Example:
#   printf 'FA_MAIN_PROVIDER=openai\nOPENAI_API_KEY=sk-...\n' > /tmp/my.env
#   ./generate-installer.sh --env /tmp/my.env --os mac --out ~/Downloads/FreeAgent-Installer.command

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="$HERE/freeagent-install.sh"
[ -f "$TEMPLATE" ] || { echo "template not found: $TEMPLATE" >&2; exit 1; }

ENV_SRC=""
OS_SEL=""
OUT=""

while [ $# -gt 0 ]; do
  case "$1" in
    --env) ENV_SRC="${2:-}"; shift 2 ;;
    --os)  OS_SEL="${2:-}";  shift 2 ;;
    --out) OUT="${2:-}";     shift 2 ;;
    -h|--help) sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1 (try --help)" >&2; exit 1 ;;
  esac
done

# Default OS selection from this machine.
if [ -z "$OS_SEL" ]; then
  case "$(uname -s)" in
    Darwin) OS_SEL="mac" ;;
    *)      OS_SEL="linux" ;;
  esac
fi

# Default output filename per OS.
if [ -z "$OUT" ]; then
  case "$OS_SEL" in
    mac)   OUT="FreeAgent-Installer.command" ;;
    linux) OUT="freeagent-install.sh" ;;
    *)     echo "unknown --os '$OS_SEL' (use mac or linux)" >&2; exit 1 ;;
  esac
fi

# Collect the KEY=value override lines to bake in.
ENV_LINES=""
if [ -n "$ENV_SRC" ]; then
  [ -f "$ENV_SRC" ] || { echo "env file not found: $ENV_SRC" >&2; exit 1; }
  # Keep only real assignments (KEY=value); drop comments and blanks.
  ENV_LINES="$(grep -E '^[A-Za-z_][A-Za-z0-9_]*=' "$ENV_SRC" || true)"
fi
# Ensure the launcher drops into the CLI unless the user explicitly set it.
if ! printf '%s\n' "$ENV_LINES" | grep -qE '^FA_LAUNCH_CLI='; then
  ENV_LINES="$(printf 'FA_LAUNCH_CLI=1\n%s' "$ENV_LINES")"
fi

# Stream the template, replacing everything between the FA_ENV_BLOCK markers
# with the collected override lines. Marker lines themselves are preserved.
{
  in_block=0
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in
      *'# >>> FA_ENV_BLOCK'*)
        printf '%s\n' "$line"
        printf '%s\n' "$ENV_LINES"
        in_block=1
        continue
        ;;
      *'# <<< FA_ENV_BLOCK'*)
        printf '%s\n' "$line"
        in_block=0
        continue
        ;;
    esac
    [ "$in_block" -eq 1 ] && continue
    printf '%s\n' "$line"
  done < "$TEMPLATE"
} > "$OUT"

chmod +x "$OUT"
echo "Wrote personalized installer: $OUT"
echo "  target OS naming : $OS_SEL"
echo "  baked-in .env    :"
printf '%s\n' "$ENV_LINES" | sed 's/^/      /'

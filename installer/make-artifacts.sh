#!/usr/bin/env bash
#
# make-artifacts.sh — produce the ready-to-distribute installer files from the
# canonical installer (install.sh). There is only ONE source of truth; these are
# byte-identical copies named so each OS will open them on double-click.
#
#   FreeAgent-Installer.command  -> macOS (Finder opens .command in Terminal)
#   freeagent-install.sh         -> Linux (run, or double-click where supported)
#
# You can also just host install.sh directly for the curl one-liner:
#   curl -fsSL <url>/install.sh | bash
#
# Usage:  ./make-artifacts.sh [OUT_DIR]      (default: ./dist)

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/install.sh"
OUT="${1:-$HERE/dist}"

[ -f "$SRC" ] || { echo "install.sh not found next to this script" >&2; exit 1; }
mkdir -p "$OUT"

cp "$SRC" "$OUT/FreeAgent-Installer.command"
cp "$SRC" "$OUT/freeagent-install.sh"
cp "$SRC" "$OUT/install.sh"
chmod +x "$OUT/FreeAgent-Installer.command" "$OUT/freeagent-install.sh" "$OUT/install.sh"

echo "Wrote distributable installers to: $OUT"
echo "  macOS  : FreeAgent-Installer.command   (double-click; first open: right-click -> Open)"
echo "  Linux  : freeagent-install.sh          (run: bash freeagent-install.sh)"
echo "  curl   : install.sh                    (curl -fsSL <url>/install.sh | bash)"

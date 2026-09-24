#!/bin/zsh
# Install (or update) SmartPinyin for the current user and enable it.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$HOME/Library/Input Methods"
mkdir -p "$DEST"
pkill -x SmartPinyin 2>/dev/null || true
rm -rf "$DEST/SmartPinyin.app"
cp -R "$ROOT/build/SmartPinyin.app" "$DEST/"
"$DEST/SmartPinyin.app/Contents/MacOS/SmartPinyin" --install || true
echo "Installed. If it does not appear yet: System Settings → Keyboard → Input Sources → + → 简体中文 → 智能拼音"
echo "Logs: ~/Library/Logs/SmartPinyin.log"

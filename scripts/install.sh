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
# The menu and the cursor's input-source badge cache artwork across app updates.
pkill -u "$(id -u)" -x TextInputMenuAgent 2>/dev/null || true
# macOS truncates this service's process name, so match its full executable path.
pkill -KILL -u "$(id -u)" -f '/CursorUIViewService\.xpc/Contents/MacOS/CursorUIViewService$' 2>/dev/null || true
echo "Installed. If it does not appear yet: System Settings → Keyboard → Input Sources → + → 简体中文 → 智能拼音"
echo "Logs: ~/Library/Logs/SmartPinyin.log"

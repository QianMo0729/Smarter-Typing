#!/bin/zsh
# Build SmartPinyin.app (input method bundle) into build/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MODEL="${SMARTPINYIN_MODEL:-models/gguf/Qwen3-0.6B-Base-q8_0.gguf}"
swift build -c release --product SmartPinyin
APP=build/SmartPinyin.app
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources/data"
cp .build/release/SmartPinyin "$APP/Contents/MacOS/"
cp resources/Info.plist "$APP/Contents/"
# Always regenerate the icon so updates cannot reuse a stale cached image.
swift scripts/make_icon.swift "$APP/Contents/Resources/input-source.tiff"
for d in zh_CN zh-Hans en; do mkdir -p "$APP/Contents/Resources/$d.lproj"; cp resources/$d.lproj/InfoPlist.strings "$APP/Contents/Resources/$d.lproj/"; done
for f in lex.bin lm.bin weights.json; do cp "data/build/$f" "$APP/Contents/Resources/data/"; done
if [ -f "$MODEL" ]; then cp "$MODEL" "$APP/Contents/Resources/data/model.gguf"; else echo "warning: no model at $MODEL (n-gram only)"; fi
xattr -cr "$APP"
codesign --force --sign - --identifier com.smarttyping.inputmethod.SmartPinyin "$APP"
du -sh "$APP"
echo "built $APP"

#!/bin/zsh
# Build the distributable: SmartPinyin.app signed with a Developer ID, notarized when a notarytool
# profile is given, zipped with the install notes and licences into build/dist/.
#   SMARTPINYIN_SIGN_ID   signing identity (default: the keychain's Developer ID Application)
#   SMARTPINYIN_NOTARY    notarytool keychain profile (xcrun notarytool store-credentials <name>)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
ID="${SMARTPINYIN_SIGN_ID:-$(security find-identity -v -p codesigning | sed -n '/Developer ID Application/{s/.*"\(.*\)".*/\1/p;q;}')}"
[ -n "$ID" ] || { echo "error: no Developer ID Application identity in the keychain"; exit 1; }

./scripts/build_app.sh
APP=build/SmartPinyin.app
BIN="$APP/Contents/MacOS/SmartPinyin"
VERSION="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$APP/Contents/Info.plist")"
cp LICENSE THIRD_PARTY_NOTICES.md "$APP/Contents/Resources/"
cp -R licenses "$APP/Contents/Resources/"
strip -S "$BIN"
# A path of the build machine in the binary means llama.cpp was not built by scripts/build_llama.sh.
LEAKS="$(strings -a "$BIN" | grep -c '/Users/' || true)"
[ "$LEAKS" = 0 ] || { echo "error: $LEAKS build machine paths in $BIN; run scripts/build_llama.sh first"; exit 1; }
# Notarization requires the hardened runtime and a secure timestamp.
codesign --force --options runtime --timestamp --sign "$ID" --identifier com.smarttyping.inputmethod.SmartPinyin "$APP"
codesign --verify --strict "$APP"

mkdir -p build/dist
if [ -n "${SMARTPINYIN_NOTARY:-}" ]; then
  ditto -c -k --keepParent "$APP" build/dist/notarize.zip
  xcrun notarytool submit build/dist/notarize.zip --keychain-profile "$SMARTPINYIN_NOTARY" --wait
  rm build/dist/notarize.zip
  xcrun stapler staple "$APP"
else
  echo "warning: not notarized (SMARTPINYIN_NOTARY is not set); Gatekeeper blocks this build on other Macs"
fi

STAGE="build/dist/SmartPinyin-$VERSION"
rm -rf "$STAGE" "$STAGE.zip"
mkdir -p "$STAGE"
ditto "$APP" "$STAGE/SmartPinyin.app"
cp resources/安装说明.txt LICENSE THIRD_PARTY_NOTICES.md "$STAGE/"
cp -R licenses "$STAGE/"
ditto -c -k --keepParent "$STAGE" "$STAGE.zip"
rm -rf "$STAGE"
du -h "$STAGE.zip"
shasum -a 256 "$STAGE.zip"

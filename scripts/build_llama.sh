#!/bin/zsh
# Build third_party/llama.cpp into the static libraries Package.swift links.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/third_party/llama.cpp"
# Keep the build machine's directory out of the binary (asserts carry __FILE__).
MAP="-ffile-prefix-map=$SRC=llama.cpp -ffile-prefix-map=$(cd "$SRC" && pwd -P)=llama.cpp"
# GGML_NATIVE=OFF: -mcpu=native would use instructions of this Mac's chip that older Apple Silicon lacks.
cmake -S "$SRC" -B "$SRC/build" \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 \
  -DGGML_NATIVE=OFF -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_OPENMP=OFF \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF \
  -DCMAKE_C_FLAGS="$MAP" -DCMAKE_CXX_FLAGS="$MAP" -DCMAKE_OBJC_FLAGS="$MAP"
cmake --build "$SRC/build" --target llama -j
echo "built llama.cpp libraries in $SRC/build"

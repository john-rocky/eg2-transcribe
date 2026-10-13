#!/usr/bin/env bash
# Puts the Qualcomm NPU runtime that LiteRT 2.2.0 compiles with on the phone (JIT) into the app's jniLibs.
#
#   tools/fetch_npu_libs.sh [v81 v79 v75 v73]          # download (LiteRT release zip + QAIRT, about 2.4 GB)
#   tools/fetch_npu_libs.sh --from <dir> [v81]          # copy from a directory that already holds the files
#
# The Hexagon version follows the SoC: SM8550 v73, SM8650 v75, SM8750 v79, SM8850 v81 (default v81).
# Sources, both pinned to what LiteRT 2.2.0 was released with:
#   - google-ai-edge/LiteRT release v2.2.0, litert_npu_runtime_libraries_jit.zip:
#     libLiteRtDispatch_Qualcomm.so and libLiteRtCompilerPlugin_Qualcomm.so
#   - Qualcomm AI Runtime (QAIRT) 2.47.0.260601, the version that release's fetch script pins: libQnnHtp.so
#     libQnnSystem.so libQnnHtpPrepare.so libQnnIr.so libQnnSaver.so libQnnHtpV<NN>Stub.so
#     libQnnHtpV<NN>CalculatorStub.so libQnnHtpV<NN>Skel.so
# QAIRT_ZIP=<path to v2.47.0.260601.zip> reuses a downloaded QAIRT archive instead of fetching it.
# Qualcomm's license lets these files ship only inside an application: never commit them or publish them alone
# (.gitignore excludes app/src/main/jniLibs/**/*.so). app/build.gradle.kts extracts them at install
# (jniLibs.useLegacyPackaging) and keeps them byte for byte (keepDebugSymbols).
set -euo pipefail

HERE=$(cd "$(dirname "$0")/.." && pwd)
DEST="$HERE/app/src/main/jniLibs/arm64-v8a"
FROM=""
if [ "${1:-}" = "--from" ]; then
  FROM=${2:?usage: tools/fetch_npu_libs.sh --from <dir> [v81 ...]}
  shift 2
fi
if [ $# -eq 0 ]; then set -- v81; fi
for v in "$@"; do
  case "$v" in
    v73|v75|v79|v81) ;;
    *) echo "unknown Hexagon version '$v' (LiteRT 2.2.0 lists v73 v75 v79 v81)" >&2; exit 2 ;;
  esac
done
mkdir -p "$DEST"

NAMES=(libLiteRtDispatch_Qualcomm.so libLiteRtCompilerPlugin_Qualcomm.so libQnnHtp.so libQnnSystem.so libQnnHtpPrepare.so libQnnIr.so libQnnSaver.so)
for v in "$@"; do
  n=${v#v}
  NAMES+=("libQnnHtpV${n}Stub.so" "libQnnHtpV${n}CalculatorStub.so" "libQnnHtpV${n}Skel.so")
done

if [ -n "$FROM" ]; then
  for name in "${NAMES[@]}"; do
    [ -f "$FROM/$name" ] || { echo "missing $FROM/$name" >&2; exit 1; }
    cp "$FROM/$name" "$DEST/$name"
  done
else
  LITERT_URL=https://github.com/google-ai-edge/LiteRT/releases/download/v2.2.0/litert_npu_runtime_libraries_jit.zip
  QAIRT_VERSION=2.47.0.260601
  QAIRT_URL=https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/$QAIRT_VERSION/v$QAIRT_VERSION.zip
  TMP=$(mktemp -d)
  trap 'rm -rf "$TMP"' EXIT
  curl -fL --retry 3 -o "$TMP/litert_jit.zip" "$LITERT_URL"
  for name in libLiteRtDispatch_Qualcomm.so libLiteRtCompilerPlugin_Qualcomm.so; do
    unzip -p "$TMP/litert_jit.zip" "qualcomm_runtime_$1/src/main/jni/arm64-v8a/$name" > "$DEST/$name"
  done
  QZIP=${QAIRT_ZIP:-$TMP/qairt.zip}
  if [ -z "${QAIRT_ZIP:-}" ]; then
    curl -fL --retry 3 -o "$QZIP" "$QAIRT_URL"
  fi
  ROOT=qairt/$QAIRT_VERSION/lib
  PATHS=("$ROOT/aarch64-android/libQnnHtp.so" "$ROOT/aarch64-android/libQnnSystem.so" "$ROOT/aarch64-android/libQnnHtpPrepare.so"
         "$ROOT/aarch64-android/libQnnIr.so" "$ROOT/aarch64-android/libQnnSaver.so")
  for v in "$@"; do
    n=${v#v}
    PATHS+=("$ROOT/aarch64-android/libQnnHtpV${n}Stub.so" "$ROOT/aarch64-android/libQnnHtpV${n}CalculatorStub.so" "$ROOT/hexagon-v${n}/unsigned/libQnnHtpV${n}Skel.so")
  done
  unzip -o -j -q "$QZIP" "${PATHS[@]}" -d "$DEST"
fi

echo "Qualcomm NPU runtime for $* in $DEST:"
(cd "$DEST" && shasum -a 256 "${NAMES[@]}")

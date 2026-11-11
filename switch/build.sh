#!/bin/bash
# Builds the Switch NRO in WSL (as root).
#   sources $TMNT_ROOT/src (rsync'd here from the git checkout), SDK $TMNT_ROOT/rexglue-sdk (ReXGlue v0.10.0
#   + patches/rexglue-sdk common and .linux patches, committed there, + switch/sdk-edits.py + sdk-edits-tmnt.py),
#   host CLI built from that SDK (out/linux-amd64/Release/rexglue), the game's files in
#   $TMNT_ROOT/src/game/extracted (never committed), NVK /root/cbfd/nvk-sdk-real/lib/libvulkan.a, devkitPro /opt/devkitpro.
#   switch/build.sh [ninja targets...]
# Every run reconfigures with the version from git ("0.<commit count>.<sha7>", the NACP's display
# version, hbmenu shows it) and writes tmnt_version.h (the same string + the build time, the first
# log line), so a log and an NRO can always be matched to a commit.
set -eu
ROOT=${TMNT_ROOT:-/root/tmntu}
SRC=$ROOT/src
SDK=${REXSDK_DIR:-$ROOT/rexglue-sdk}
NVK_SDK=${NVK_SDK:-/root/cbfd/nvk-sdk-real}
GIT_DIR=${TMNT_GIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}
export DEVKITPRO=${DEVKITPRO:-/opt/devkitpro}
export PATH=$DEVKITPRO/tools/bin:$DEVKITPRO/devkitA64/bin:$PATH
BUILD=$SRC/out/build/switch
MANIFEST=tmnt_turtles_in_time_reshelled_manifest.toml

[ -f "$NVK_SDK/lib/libvulkan.a" ] || { echo "no NVK: $NVK_SDK/lib/libvulkan.a (cbfd-switch switch/build-nvk.sh)"; exit 1; }
CLI=$SDK/out/linux-amd64/Release/rexglue
[ -x "$CLI" ] || { echo "no host CLI: $CLI"; exit 1; }

mkdir -p "$SRC"
rsync -rlc --delete --exclude /out --exclude /game --exclude /generated --exclude /.git --exclude /graphify-out "$GIT_DIR/" "$SRC/"
[ -f "$SRC/game/extracted/default.xex" ] || { echo "no game files: $SRC/game/extracted/default.xex"; exit 1; }
# Switch only: skip_msr (the SDK's one process-wide mtmsrd mutex cost the Castlevania port a third of its
# worker time). The flag goes into the copy's [entrypoint]; the PC manifest stays as it is.
sed -i '/^\[entrypoint\]$/a skip_msr = true' "$SRC/$MANIFEST"
# The edits go onto a clean SDK whenever they changed: a newer sdk-edits.py on an SDK that holds an older
# one's inserts keeps stale code (hcu-switch 0.106: the game hung after its intro). The SDK's HEAD is the
# commit with patches/rexglue-sdk applied.
EDITS_STAMP=$(cat "$SRC/switch/sdk-edits.py" "$SRC/switch/sdk-edits-tmnt.py" | sha1sum | cut -c1-40)
if [ "$(cat "$SDK/.tmnt-edits-stamp" 2>/dev/null)" != "$EDITS_STAMP" ]; then
    echo "sdk edits changed: resetting $SDK to $(git -C "$SDK" rev-parse --short HEAD)"
    git -C "$SDK" checkout -q -- . && git -C "$SDK" clean -fdq -e /out
    git -C "$SDK" submodule foreach -q 'git checkout -q -- . && git clean -fdq'
    rm -f "$SDK/.tmnt-edits-stamp"
fi
python3 "$SRC/switch/sdk-edits.py" "$SDK"
# A second pass: some sdk-edits.py inserts anchor on text an earlier insert of the same run writes.
python3 "$SRC/switch/sdk-edits.py" "$SDK" > /dev/null
python3 "$SRC/switch/sdk-edits-tmnt.py" "$SDK"
echo "$EDITS_STAMP" > "$SDK/.tmnt-edits-stamp"
# The CLI embeds the codegen templates: rebuild it after the SDK edits, and regenerate when the manifest,
# its config includes or the codegen sources changed.
cmake --build "$SDK/out/build/linux-amd64" --config Release --target rexglue > "$ROOT/cli-build.log" 2>&1 || { tail -20 "$ROOT/cli-build.log"; exit 1; }
CG_STAMP=$(cat "$SRC/$MANIFEST" "$SRC"/config/*.toml "$SDK/resources/templates/codegen/pch_h.inja" $(find "$SDK/src/codegen" "$SDK/include/rex/codegen" -type f | sort) | sha1sum | cut -c1-40)
if [ ! -f "$SRC/generated/default/sources.cmake" ] || [ "$(cat "$SRC/generated/.tmnt-codegen-stamp" 2>/dev/null)" != "$CG_STAMP" ]; then
    (cd "$SRC" && "$CLI" codegen --ignore-stamp $MANIFEST > "$ROOT/codegen.log" 2>&1) || { tail -20 "$ROOT/codegen.log"; exit 1; }
    echo "$CG_STAMP" > "$SRC/generated/.tmnt-codegen-stamp"
fi

COUNT=$(git -c safe.directory="*" -C "$GIT_DIR" rev-list --count HEAD 2>/dev/null || echo 0)
SHA=$(git -c safe.directory="*" -C "$GIT_DIR" rev-parse --short=7 HEAD 2>/dev/null || echo unknown)
DIRTY=$(git -c safe.directory="*" -c core.autocrlf=true -C "$GIT_DIR" status --porcelain --untracked-files=no 2>/dev/null | grep -q . && echo "+" || true)
VERSION="0.$COUNT.$SHA$DIRTY"
STAMP=$(date -u +%Y-%m-%dT%H:%MZ)
mkdir -p "$BUILD"
printf '#define TMNT_VERSION "%s"\n#define TMNT_BUILD_UTC "%s"\n' "$VERSION" "$STAMP" > "$BUILD/tmnt_version.h.new"
if ! cmp -s "$BUILD/tmnt_version.h.new" "$BUILD/tmnt_version.h" 2>/dev/null; then
    mv "$BUILD/tmnt_version.h.new" "$BUILD/tmnt_version.h"
else
    rm -f "$BUILD/tmnt_version.h.new"
fi
echo "version $VERSION, built $STAMP"
# NRO icon from the player's game files (banner.png, title_icon.png), game art kept outside git.
ICON=$ROOT/icon/tmnt_icon_256.jpg
mkdir -p "$ROOT/icon"
python3 "$SRC/tools/game_icon.py" "$SRC/game/extracted" "$ICON" || rm -f "$ICON"

# deko3d renderer shaders (shared with cvhd-recomp): the present shaders and the game's shaders as DKSH. They go
# into the NRO (romfs:/shaders/, nx_create_nro ROMFS in switch/switch.cmake), so the NRO is the whole deploy and
# can not run with another build's shaders (console 0.84 ran with the card's 0.78 folder). The game's shaders come
# from the trace backend's --dump_shaders directory ($TMNT_SHADER_DUMPS) through tools/xenos2glsl.py + uam.
ROMFS=$BUILD/romfs
rm -rf "$ROMFS"
mkdir -p "$ROMFS/shaders"
for s in vert frag; do
    uam -s $s -o "$ROMFS/shaders/present.$s.dksh" "$SRC/src/gpu/shaders/present.$s"
done
uam -s frag -o "$ROMFS/shaders/depthcopy.frag.dksh" "$SRC/src/gpu/shaders/depthcopy.frag"
uam -s frag -o "$ROMFS/shaders/depthrestore.frag.dksh" "$SRC/src/gpu/shaders/depthrestore.frag"
SHADER_DUMPS=${TMNT_SHADER_DUMPS:-$ROOT/shaders_trace}
if [ -d "$SHADER_DUMPS" ]; then
    rm -rf "$BUILD/glsl"
    python3 "$SRC/tools/xenos2glsl.py" --all "$SHADER_DUMPS" --out-dir "$BUILD/glsl" --check | tail -1
    for f in "$BUILD"/glsl/*.glsl.dksh; do
        [ -f "$f" ] || continue
        cp "$f" "$ROMFS/shaders/$(basename "$f" .glsl.dksh).dksh"
    done
    # Markers of the vertex shaders whose streams are storage buffers (computed fetch indices). One line each:
    # an empty file inside a romfs is not relied on.
    for f in "$BUILD"/glsl/*.vert.ssbo; do
        [ -f "$f" ] && echo ssbo > "$ROMFS/shaders/$(basename "$f")"
    done
fi

cmake -S "$SRC" -B "$BUILD" -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="$DEVKITPRO/cmake/Switch.cmake" \
    -DCMAKE_BUILD_TYPE=Release \
    -DREXSDK_DIR="$SDK" -DNVK_SDK="$NVK_SDK" -DTMNT_VERSION="$VERSION" -DTMNT_ICON="$ICON" \
    -DTMNT_DEKO3D_LIB="${TMNT_DEKO3D_LIB:-deko3dd}" \
    -DREXGLUE_ENABLE_TRACY=OFF -DREXGLUE_ENABLE_PERF_COUNTERS=ON > "$BUILD/configure.log" 2>&1 \
    || { tail -20 "$BUILD/configure.log"; exit 1; }
# A failed build must not leave the previous NRO behind for deploy.sh.
rm -f "$BUILD/tmnt.nro"
cmake --build "$BUILD" --parallel "${TMNT_JOBS:-12}" ${1:+--target "$@"}
ls -la "$BUILD"/tmnt.nro "$BUILD"/tmnt.elf 2>/dev/null || true
sha1sum "$BUILD"/tmnt.nro 2>/dev/null || true
# What hbmenu will show; the NACP version must be this build's.
python3 "$SRC/tools/nro_nacp.py" "$BUILD/tmnt.nro" "$VERSION"

# Staging next to the NRO in $BUILD/stage/: the card folder sdmc:/switch/tmnt/shaders/ is only the fallback of an
# NRO without shaders; the copy keeps that folder current.
STAGE=$BUILD/stage
rm -rf "$STAGE"
mkdir -p "$STAGE"
cp -r "$ROMFS/shaders" "$STAGE/shaders"
cp "$BUILD/tmnt.nro" "$STAGE/tmnt.nro"
touch "$STAGE/deko3d"
echo "stage: $STAGE ($(ls "$STAGE/shaders" | wc -l) shaders)"

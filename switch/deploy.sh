#!/bin/bash
# Copies the staged Switch build to the console and verifies every file by SHA-1 (run in WSL as root).
#   switch/deploy.sh [push|game|logs]   console: TMNT_FTP=ip:port (default 192.168.50.69:5000)
# push: tmnt.nro (NACP version checked first), the deko3d marker and every staged shader, each through
#       switch/push_verified.sh (.part upload, read-back, SHA-1, rename).
# game: the extracted game ($TMNT_ROOT/src/game/extracted) -> sdmc:/switch/tmnt/game/, file by file, verified.
# logs: tmnt_log_<build id>.txt and tmnt_rex_log.txt -> switch/logs/ (git-ignored).
set -eu
cd "$(dirname "$0")/.."
export CLX_FTP=${TMNT_FTP:-192.168.50.69:5000}
BUILD=${SWITCH_BUILD_DIR:-${TMNT_ROOT:-/root/tmntu}/src/out/build/switch}
STAGE=$BUILD/stage
GAME=${TMNT_GAME_DIR:-${TMNT_ROOT:-/root/tmntu}/src/game/extracted}
P=switch/push_verified.sh
case "${1:-push}" in
    push)
        VERSION=$(grep -o 'TMNT_VERSION "[^"]*"' "$BUILD/tmnt_version.h" | cut -d'"' -f2)
        python3 tools/nro_nacp.py "$STAGE/tmnt.nro" "$VERSION"
        mkdir -p "switch/logs/elf/$VERSION" && cp "$BUILD/tmnt.elf" "switch/logs/elf/$VERSION/"  # symbols for console crash reports
        args=("$STAGE/tmnt.nro" /switch/tmnt/tmnt.nro nro "$STAGE/deko3d" /switch/tmnt/deko3d)
        curl -s --ftp-create-dirs -Q "*MKD /switch/tmnt/shaders" "ftp://$CLX_FTP/switch/tmnt/" -o /dev/null || true
        for f in "$STAGE"/shaders/*; do args+=("$f" "/switch/tmnt/shaders/$(basename "$f")"); done
        bash $P "${args[@]}"
        echo "pushed version $VERSION"
        ;;
    game)
        args=()
        while IFS= read -r f; do args+=("$GAME/$f" "/switch/tmnt/game/$f"); done < <(cd "$GAME" && find . -type f | sed 's#^\./##')
        while IFS= read -r d; do curl -s --ftp-create-dirs "ftp://$CLX_FTP/switch/tmnt/game/$d/" -o /dev/null || true; done < <(cd "$GAME" && find . -type d | sed 's#^\./##;s#^\.$##')
        bash $P "${args[@]}"
        ;;
    logs)
        mkdir -p switch/logs
        # The game's log is tmnt_log_<build id>.txt (switch_nx.cpp open_log): the id of the local build.
        ID=$(grep -o 'TMNT_BUILD_ID "[^"]*"' "$BUILD/tmnt_version.h" 2>/dev/null | cut -d'"' -f2 || true)
        # No local build (no id): the newest tmnt_log_*.txt on the console, by the build time at the end of its name
        # (<version>-<YYYYMMDDTHHMMZ>; a version sort would put 0.100 before 0.99).
        [ -n "$ID" ] || ID=$(curl -sS "ftp://$CLX_FTP/switch/tmnt/" | tr -d '\r' | awk '{print $NF}' |
            sed -n 's/^tmnt_log_\(.*-\([0-9]\{8\}T[0-9]\{4\}Z\)\)\.txt$/\2 \1/p' | sort | tail -n 1 | cut -d' ' -f2 || true)
        for log in "tmnt_log${ID:+_$ID}.txt" tmnt_rex_log.txt; do
            curl -sS -o "switch/logs/$log" "ftp://$CLX_FTP/switch/tmnt/$log" && echo "switch/logs/$log" || true
        done
        ;;
    *) sed -n '2,8p' "$0"; exit 1 ;;
esac

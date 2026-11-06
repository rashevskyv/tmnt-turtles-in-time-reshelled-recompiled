#!/usr/bin/env bash
# push_verified.sh: verified push to the console's FTP (run in WSL as root).
# Each file goes up whole (curl STOR, never REST/resume) to <name>.part, is read
# back and compared by SHA-1 (NROs also need "NRO0" at 0x10), and only then is
# renamed over the final name (DELE + RNFR/RNTO: FAT rename does not replace).
# The final file's size is checked after the rename. Any failure after 3 tries
# is printed as FAILED and the script exits 1.
#   CLX_FTP=ip:port push_verified.sh <local file> <remote path> [nro]   ...repeated as triples
H=ftp://${CLX_FTP:?set CLX_FTP=<console ip>:<port> (ftpd)}
T=${TMPDIR:-/tmp}/clx_vpush; mkdir -p $T
rc=0

one() {
    local src=$1 dst=$2 kind=$3 a b m n try
    a=$(sha1sum < "$src" | cut -c1-40)
    n=$(stat -c %s "$src")
    if [ "$kind" = nro ] && [ "$(dd if="$src" bs=1 skip=16 count=4 2>/dev/null)" != NRO0 ]; then
        echo "FAILED $dst: local file has no NRO0"; return 1
    fi
    for try in 1 2 3; do
        curl -s --max-time 60 -Q "*DELE $dst.part" "$H/" -o /dev/null
        if ! curl -s --fail --max-time 1800 -T "$src" "$H$dst.part"; then
            echo "  $dst try $try: upload failed"; continue
        fi
        rm -f $T/rb
        if ! curl -s --fail --max-time 1800 "$H$dst.part" -o $T/rb; then
            echo "  $dst try $try: read-back failed"; continue
        fi
        b=$(sha1sum < $T/rb | cut -c1-40)
        if [ "$a" != "$b" ]; then
            echo "  $dst try $try: SHA-1 mismatch (local $a, remote $b)"; continue
        fi
        if [ "$kind" = nro ]; then
            m=$(dd if=$T/rb bs=1 skip=16 count=4 2>/dev/null)
            [ "$m" = NRO0 ] || { echo "  $dst try $try: no NRO0 in the read-back"; continue; }
        fi
        curl -s --max-time 60 -Q "*DELE $dst" "$H/" -o /dev/null
        if ! curl -s --fail --max-time 60 -Q "RNFR $dst.part" -Q "RNTO $dst" "$H/" -o /dev/null; then
            echo "  $dst try $try: rename failed"; continue
        fi
        if ! curl -s --max-time 60 "$H${dst%/*}/" | tr -d '\r' | grep -q " $n .* ${dst##*/}\$"; then
            echo "  $dst try $try: final file not listed with $n bytes"; continue
        fi
        echo "OK $dst sha1 $a ($n bytes)"
        return 0
    done
    echo "FAILED $dst"
    return 1
}

while [ $# -ge 2 ]; do
    k=""
    [ "${3:-}" = nro ] && k=nro
    one "$1" "$2" "$k" || rc=1
    shift 2; [ -n "$k" ] && shift
done
exit $rc

// Button glyphs for the Switch's button positions. The pad maps by position (Switch B = 360 A, A = B,
// Y = X, X = Y), so the game's A/B and X/Y glyphs are swapped in memory when the game reads TMNT.bf
// (the XFile::Read bounce copy, switch/sdk-edits.py): the game data on the card stays untouched.
// TMNT.bf files: a 32-byte header (compressed size cs, size, trailer size x, 0, 8 bytes AA), the LZO body (cs
// bytes), then x bytes uncompressed: the list of the keys the file uses. The glyph texture keys (from the game's
// ps3_button_swap.swp) occur only in the trailers of 30 FIRELAB_*.feu files (menus, HUD, pause), 60 keys in all
// (measured 2026-11-18). A 4-byte word of a trailer, aligned to its start, that holds a glyph key is swapped; the
// trailer comes from the header read before it, so the trailer can arrive in one read or in pieces.
// Consoles 0.56-0.100 logged no swap with the scan for "42000000 <key> 00000000" inside one read.
// The setting BUTTON ICONS = XBOX (nx_menu.cpp) leaves the files as they are from the next start.
#include <rex/nx_menu.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <strings.h>

namespace {

struct Pair {
  uint32_t a, b;
};
const Pair kSwap[] = {
    {0x5040201Cu, 0x50401FD9u},  // A <-> B
    {0x504021C8u, 0x504021C5u},  // X <-> Y
};

uint32_t Load(const uint8_t* p) {
  uint32_t v;
  std::memcpy(&v, p, 4);
  return v;  // little-endian on the Switch, as in the file
}

struct Trailer {
  uint64_t header, begin, end;  // file offsets: the header, the trailer [begin, end)
};
// ponytail: the latest 64 trailers only; the game reads a trailer right after its header and body.
Trailer g_trailers[64];
int g_next = 0;
uint64_t g_files[64];  // the headers of the files with swapped keys (30 expected)
std::mutex g_mutex;
int g_headers = 0, g_whole = 0, g_pieces = 0, g_cut = 0, g_file_count = 0, g_total = 0, g_pieces_logged = 0,
    g_swaps_logged = 0;

}  // namespace

extern "C" void rex_nx_patch_file_read(const char* name, uint8_t* data, uint64_t file_offset, size_t length) {
  if (!name || strcasecmp(name, "TMNT.bf") || !rex_nx_setting_button_icons() || !length) return;
  const uint64_t read_end = file_offset + length;
  std::lock_guard<std::mutex> lock(g_mutex);
  // A read that starts at a file header: note the file's trailer (a header read twice is noted once).
  if (length >= 32 && Load(data + 12) == 0 && Load(data + 16) == 0xAAAAAAAAu && Load(data + 20) == 0xAAAAAAAAu) {
    const uint32_t cs = Load(data), x = Load(data + 8);
    bool known = false;
    for (const Trailer& t : g_trailers) known |= t.end && t.header == file_offset;
    if (cs && x && x < 0x10000 && !known) {
      g_trailers[g_next] = {file_offset, file_offset + 32 + cs, file_offset + 32 + cs + x};
      g_next = (g_next + 1) % 64;
      if (++g_headers == 1 || g_headers % 128 == 0)
        std::printf("[nx] glyphs: %d file headers with a trailer, trailer reads: %d whole, %d in pieces, %d keys cut by a read; "
                    "%d keys swapped in %d files\n", g_headers, g_whole, g_pieces, g_cut, g_total, g_file_count);
    }
  }
  for (Trailer& t : g_trailers) {
    if (!t.end || t.end <= file_offset || t.begin >= read_end) continue;
    const bool whole = t.begin >= file_offset && t.end <= read_end;
    whole ? g_whole++ : g_pieces++;
    if (!whole && g_pieces_logged < 16) {
      g_pieces_logged++;
      std::printf("[nx] glyphs: trailer of the file at %llu read in pieces: %zu bytes at %llu\n",
                  (unsigned long long)t.header, length, (unsigned long long)file_offset);
    }
    int swapped = 0;
    for (uint64_t w = t.begin; w + 4 <= t.end; w += 4) {
      if (w + 4 <= file_offset || w >= read_end) continue;
      if (w < file_offset || w + 4 > read_end) {  // a word cut by the edge of this read stays as it is
        g_cut++;
        continue;
      }
      uint8_t* p = data + (w - file_offset);
      const uint32_t k = Load(p);
      for (const Pair& pr : kSwap) {
        const uint32_t to = k == pr.a ? pr.b : k == pr.b ? pr.a : 0;
        if (!to) continue;
        std::memcpy(p, &to, 4);
        swapped++;
        break;
      }
    }
    if (!swapped) continue;
    g_total += swapped;
    bool counted = false;
    for (int i = 0; i < g_file_count; i++) counted |= g_files[i] == t.header;
    if (!counted && g_file_count < 64) g_files[g_file_count++] = t.header;
    if (g_swaps_logged++ >= 128) continue;  // menus load again and again: the first 128 swaps are logged
    std::printf("[nx] glyphs: %d A/B X/Y texture keys swapped in the trailer of the file at %llu (%llu bytes); "
                "%d keys in %d files so far (a full set: 60 keys in 30 FIRELAB .feu files)\n",
                swapped, (unsigned long long)t.header, (unsigned long long)(t.end - t.begin), g_total, g_file_count);
    std::fflush(stdout);
  }
}

// Button glyphs for the Switch's button positions. The pad maps by position (Switch B = 360 A, A = B,
// Y = X, X = Y), so the game's A/B and X/Y glyphs are swapped in memory when the game reads TMNT.bf
// (the XFile::Read bounce copy, switch/sdk-edits.py): the game data on the card stays untouched.
// TMNT's front-end files (FIRELAB_*.feu, menus and the HUD) list the textures they use as
// 42000000 <key LE> 00000000; the glyph keys come from the game's own ps3_button_swap.swp (Xbox key ->
// PS3 key pairs, measured 2026-11-07). Only those 30 .feu files hold the keys with that tag. The
// setting BUTTON ICONS = XBOX (nx_menu.cpp) leaves the files as they are from the next start.
#include <rex/nx_menu.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
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

bool g_logged = false;

}  // namespace

extern "C" void rex_nx_patch_file_read(const char* name, uint8_t* data, uint64_t file_offset, size_t length) {
  (void)file_offset;
  static int names_logged = 0;  // console 0.56 logged no swap: which files come through here
  if (name && names_logged < 4 && (names_logged == 0 || file_offset == 0)) {
    names_logged++;
    std::printf("[nx] glyphs: read of %s at %llu, %zu bytes\n", name, (unsigned long long)file_offset, length);
  }
  if (!name || strcasecmp(name, "TMNT.bf") || !rex_nx_setting_button_icons() || length < 12) return;
  // ponytail: an entry cut by the edge of a read stays as it is; the game reads the .feu files whole.
  int swapped = 0;
  for (size_t i = 4; i + 8 <= length; i++) {  // the entries are not 4-byte aligned in the archive
    if (data[i - 4] != 0x42 || Load(data + i - 4) != 0x42u || Load(data + i + 4) != 0) continue;
    const uint32_t k = Load(data + i);
    for (const Pair& p : kSwap) {
      const uint32_t to = k == p.a ? p.b : k == p.b ? p.a : 0;
      if (!to) continue;
      std::memcpy(data + i, &to, 4);
      swapped++;
      break;
    }
  }
  if (swapped && !g_logged) {
    g_logged = true;
    std::printf("[nx] glyphs: %d A/B X/Y texture keys swapped for the Switch's button positions\n", swapped);
    std::fflush(stdout);
  }
}

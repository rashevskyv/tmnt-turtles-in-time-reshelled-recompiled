// Button glyphs for the Switch's button positions. The pad maps by position (Switch B = 360 A, A = B,
// Y = X, X = Y), so the game's A/B and X/Y glyphs are swapped in the menu bundles as the game unpacks them
// (fire_menu_hook.cpp): the game data on the card stays untouched.
// The glyph texture keys (from the game's ps3_button_swap.swp) occur only in the menu bundles f88d0000.bin and
// f88d0002.bin of TMNT.bf, inside their LZO data: no read of TMNT.bf holds them (consoles 0.56-0.104 swapped nothing
// at the file read). Each FEU movie in a bundle has a texture list: a count, then (key, 0, type) triples, type 0x42 =
// texture; 47 glyph triples in f88d0000 and 28 in f88d0002 (measured 2026-11-18). Only these references are swapped:
// the bundle's object table holds the same keys as the textures' own names, and swapping both would cancel out.
// disabled: breaks the menu bundle key scan, see commit e5ddce2, to be redone. rex_nx_setting_button_icons()
// (nx_menu.cpp) is always 0 and the menu has no item for it, so the swap below never runs.
#include <rex/nx_menu.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

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
  return v;  // little-endian on the Switch, as in the bundle
}

}  // namespace

extern "C" void rex_nx_swap_menu_glyphs(uint8_t* data, size_t size) {
  if (!rex_nx_setting_button_icons()) {  // disabled: breaks the menu bundle key scan, see commit e5ddce2, to be redone
    std::printf("[nx] glyphs: the swap is off, the menu bundle of %zu bytes is left as it is\n", size);
    return;
  }
  int swapped = 0;
  for (size_t i = 0; i + 12 <= size; i++) {  // the triples are not 4-aligned in the bundle
    if (Load(data + i + 8) != 0x42u || Load(data + i + 4) != 0) continue;
    const uint32_t k = Load(data + i);
    for (const Pair& pr : kSwap) {
      const uint32_t to = k == pr.a ? pr.b : k == pr.b ? pr.a : 0;
      if (!to) continue;
      std::memcpy(data + i, &to, 4);
      swapped++;
      break;
    }
  }
  std::printf("[nx] glyphs: %d A/B X/Y texture references swapped in the menu bundle of %zu bytes "
              "(expected 47 in 19784989 bytes, 28 in 15331229 bytes)\n", swapped, size);
  std::fflush(stdout);
}

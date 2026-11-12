// One draw of an instancing vertex shader (tools/xenos2glsl.py, Translator.__init__; cvhd-recomp, BSD-3 like the SDK).
// The shader computes per vertex index i, in 32-bit floats:
//   x = float(i) + offset;  q = (x + bias) * (1 / N), N = trunc(c19.x);
//   instance record = floor(q);  mesh record = x - trunc(q) * N.
// The game draws one instance per draw (VGT_INDX_OFFSET = instance * N), so the backend resolves the instance here and
// feeds both streams as vertex attributes. Check: tools/instancing_check.cpp.
#pragma once
#include <cmath>
#include <cstdint>

namespace rex::graphics::tmnt {

// True: every index of [min_index, max_index] reads instance record `instance` and index i reads mesh record
// `first + i`. False: the draw spans instances, the numbers are not exact in a float, or the mesh would start before
// its stream. The same float expression as the shader, at both ends (q grows with i), without a tolerance: a tolerance
// picks the neighbouring matrix silently.
// ponytail: first < 0 (an index offset that is not instance * N while the lowest index is above 0) is refused, and so
// is a draw over several instances; a vertex-expanding copy on the CPU would draw them if the game ever has such draws.
inline bool ResolveInstance(float c19x, float bias, int32_t offset, uint32_t min_index, uint32_t max_index,
                            uint32_t& instance, uint32_t& first) {
  const float n = std::trunc(c19x);
  if (!(n >= 1.0f && n < 16777216.0f)) return false;
  const int64_t lo = int64_t(min_index) + offset, hi = int64_t(max_index) + offset;
  if (lo < 0 || hi < lo || hi >= (1 << 24)) return false;  // float(i) + offset is exact below 2^24
  const float rn = 1.0f / n;
  const float q_lo = std::floor((float(lo) + bias) * rn), q_hi = std::floor((float(hi) + bias) * rn);
  if (!(q_lo >= 0.0f) || q_lo != q_hi) return false;
  const int64_t base = int64_t(q_lo) * int64_t(n);  // the instance's first index
  if (lo < base || hi - base >= int64_t(n) || offset < base) return false;
  instance = uint32_t(q_lo);
  first = uint32_t(offset - base);
  return true;
}

}  // namespace rex::graphics::tmnt

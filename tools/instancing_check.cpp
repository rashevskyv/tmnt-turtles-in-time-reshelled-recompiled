// Self-check of tmnt::ResolveInstance against the instancing shaders' own formula, for every index of a draw.
//   g++ -std=c++20 -O2 -o /tmp/instancing_check tools/instancing_check.cpp && /tmp/instancing_check
#include <cassert>
#include <cstdio>
#include <initializer_list>

#include "../src/gpu/instancing.h"

namespace tmnt = rex::graphics::tmnt;

int main() {
  unsigned draws = 0;
  for (float n : {24.0f, 35.0f, 71.0f, 161.0f, 533.0f, 908.0f, 2394.0f})
    for (int k : {0, 1, 2, 17, 100, 1000})
      for (float bias : {0.5f, 0.25f})
        for (float frac : {0.0f, 0.5f}) {  // c19.x need not be whole: the shader truncates it
          const int32_t offset = k * int32_t(n);
          uint32_t instance = ~0u, first = ~0u;
          assert(tmnt::ResolveInstance(n + frac, bias, offset, 0, uint32_t(n) - 1, instance, first));
          for (uint32_t i = 0; i < uint32_t(n); i++) {
            // The shader, statement by statement (shader_F7F9306F5415C989.vert.glsl of 0.86).
            const float x = float(i) + float(offset);
            const float q = (x + bias) * (1.0f / n);
            const float mesh = std::floor((-std::trunc(q)) * n + x);
            assert(std::floor(q) == float(instance));
            assert(mesh == float(first + i));
          }
          // The lowest index above 0, the offset still instance * N: the same instance, mesh from record 0.
          assert(tmnt::ResolveInstance(n + frac, bias, offset, 3, uint32_t(n) - 1, instance, first) && instance == uint32_t(k) && first == 0);
          // One index too many: two instances in one draw.
          assert(!tmnt::ResolveInstance(n + frac, bias, offset, 0, uint32_t(n), instance, first));
          // An offset that is not instance * N while the mesh starts above index 0: refused (first would be -5).
          assert(!tmnt::ResolveInstance(n + frac, bias, offset - 5, 5, uint32_t(n) + 4, instance, first));
          draws++;
        }
  uint32_t a, b;
  assert(!tmnt::ResolveInstance(0.5f, 0.5f, 0, 0, 10, a, b));    // N = 0
  assert(!tmnt::ResolveInstance(24.0f, 0.5f, -30, 0, 10, a, b)); // a negative guest index
  std::printf("instancing_check: %u draws ok\n", draws);
  return 0;
}

// deko3d GPU backend (--gpu_backend=deko3d, Nintendo Switch): the SDK's Xenos command processor
// parses the ring (fences, read-pointer write-back, vblank, bin masks), this backend draws with
// deko3d. No EDRAM model: a host image per render target (keyed by EDRAM base and pitch), resolves
// are image copies into images keyed by guest address, textures bound at such an address alias
// them (no readback), the swap presents the image at the front buffer's address through one
// textured triangle. Shaders are translated offline (tools/xenos2glsl.py, uam) and loaded by the
// ucode hash from sdmc:/switch/tmnt/shaders/. docs/native-renderer-plan.md, M3.
// cvhd-recomp, BSD-3 like the SDK.
// Linux (TMNT_DK_GL): the same code over an OpenGL stand-in for libdeko3d (dk_gl/dk_gl_shim.cpp), a test
// host for the backend logic, with the shaders as translated GLSL instead of DKSH.
#if defined(__SWITCH__) || (defined(__linux__) && __has_include(<EGL/egl.h>))
#ifdef __SWITCH__
#include <switch.h>
#include <deko3d.h>
#else
#define TMNT_DK_GL 1
#include "dk_gl/switch.h"
#include "dk_gl/dk_gl_shim.h"
#endif
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <thread>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <rex/cvar.h>
#include <rex/graphics/command_processor.h>
#include <rex/graphics/flags.h>
#include <rex/graphics/graphics_system.h>
#include <rex/graphics/pipeline/shader/shader.h>
#include <rex/graphics/pipeline/texture/conversion.h>
#include <rex/graphics/pipeline/texture/info.h>
#include <rex/graphics/pipeline/texture/util.h>
#include <rex/graphics/register_file.h>
#include <rex/graphics/registers.h>
#include <rex/graphics/util/draw.h>
#include <rex/graphics/xenos.h>
#include <rex/logging.h>
#include <rex/memory.h>
#include <rex/string/buffer.h>
#include <rex/system/gpu_plugin.h>

#include "ucode_hash.h"
#include "untile_fast.h"

extern "C" volatile unsigned rex_nx_swaps;     // switch_nx.cpp: the busy/frame line divides by it
extern "C" volatile double rex_nx_main_busy_ms;  // switch_nx.cpp: the main thread's busy ms per frame
extern "C" void rex_nx_waited_add(unsigned long long ticks);  // switch_nx.cpp: the busy/frame line
#include "../../switch/host/include/rex/nx_menu.h"  // switch/host/nx_menu.cpp: the settings and the overlay the present pass draws
#ifndef TMNT_DK_GL
#include "../../switch/host/include/rex/frame_recorder.h"  // switch/host/frame_recorder.cpp: RECORD 5 SEC
#endif
extern "C" volatile int rex_nx_vblank_divider;  // SDK graphics_system.cpp (sdk-edits rule 31): vblanks per guest vblank interrupt
#ifdef TMNT_DK_GL
volatile int rex_nx_vblank_divider = 1;
volatile unsigned rex_nx_swaps = 0;  // no switch_nx.cpp on the Linux test host
volatile double rex_nx_main_busy_ms = 0;
extern "C" void rex_nx_waited_add(unsigned long long) {}
extern "C" int rex_nx_setting_fps_mode(void) { return 0; }  // no menu on the Linux test host
extern "C" int rex_nx_setting_fps_counter(void) { return 1; }
extern "C" int rex_nx_debug_record(void) { return 0; }
extern "C" void rex_nx_debug_record_set(int) {}
extern "C" int rex_nx_menu_overlay(NxOverlay* out, const char*, int, int) { std::memset(out, 0, sizeof(*out)); return 0; }
#endif

// Where the DKSH shaders are (tools/xenos2glsl.py + uam, staged by switch/build.sh): a game sets it
// in its tmnt.toml / arguments. Missing shaders are dumped to its missing/ subdirectory. The debug
// markers are files next to the directory: dk_dump (image readback at swaps 120 and 600),
// dk_present_rt (present the widest colour render target instead of the front buffer).
REXCVAR_DEFINE_STRING(deko3d_shader_dir, "sdmc:/switch/tmnt/shaders/", "GPU",
                      "deko3d backend: directory of shader_<ucode hash>.vert|geom|frag.dksh and present.vert|frag.dksh")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
// Xenos rectangle lists (3 vertices, the 4th = v1 + v2 - v0): by default through the geometry shader
// the translator writes next to each vertex shader; the hardware fill-rectangle mode (NV_fill_rectangle,
// the same semantics) is an option for the console: Eden ignores that mode (one triangle is drawn).
REXCVAR_DEFINE_BOOL(deko3d_fill_rectangle, false, "GPU",
                    "deko3d backend: draw rectangle lists with the hardware fill-rectangle mode instead of a geometry shader")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

REXCVAR_DEFINE_INT32(deko3d_cull, 1, "GPU",
                     "deko3d backend: 0 = no face culling, 1 = cull as the guest asks, 2 = the same with the winding inverted")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

// The Switch host layer's dirty-page bytes (switch/host/memory_nx.cpp): one byte per 4 KB page of the
// physical arena, set by every guest store (the codegen pch) and by host writes into guest memory
// (file reads). This backend is their only consumer (there is no SharedMemory here).
extern "C" volatile uint8_t rex_nx_dirty[0x20000];
extern "C" volatile uint8_t rex_nx_dirty_coarse[0x200];
#ifdef TMNT_DK_GL
// No store macros mark pages on Linux: the arrays stay empty and changes are found by hashing the guest
// memory of each texture at its bind (MemoryChanged).
extern "C" volatile uint8_t rex_nx_dirty[0x20000] = {};
extern "C" volatile uint8_t rex_nx_dirty_coarse[0x200] = {};
#endif

namespace rex::graphics::deko {

namespace {

constexpr uint32_t kFrames = 2;
constexpr uint32_t kFrameDataBytes = 24u << 20;  // vertex, index and uniform data of one frame
constexpr uint32_t kFrameCmdBytes = 4u << 20;
// One frame's texture uploads. The first stage's backdrop layers are 4320x2432 RGBA8 textures (41 MB each): a
// texture larger than this is not uploaded at all ("staging ring full") and its draws sample nothing. 96 MB holds
// both layers in the stage's first frame (72 MB held one; the second came a frame later, tiles missing meanwhile).
constexpr uint32_t kStagingBytes = 96u << 20;
// ponytail: a bump pool, nothing is freed (the title 132 MB, the menu 185 MB): a free list with
// eviction of textures not bound for a while is the upgrade when a stage fills it.
constexpr uint32_t kImagePoolBytes = 448u << 20;
constexpr uint32_t kCodePoolBytes = 4u << 20;
constexpr uint32_t kMaxImageDescriptors = 512;
constexpr uint32_t kSamplerCount = 513;  // see SamplerIndex: min, mag, border, clamp x, clamp y; the last is kCubeSampler
constexpr uint32_t kCubeSampler = 512;   // cube textures: linear with linear mips (the shaders pick the level: textureLod)
constexpr uint32_t kPresentWidth = 1280;
constexpr uint32_t kPresentHeight = 720;
constexpr uint32_t kEdramTiles = 2048;  // 10 MB of 5120-byte tiles (80 x 16 x 32 bit)
constexpr uint32_t kMaxTargetHeight = 1088;

void Log(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
void Log(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  std::printf("[dk] ");
  std::vprintf(fmt, ap);
  std::printf("\n");
  va_end(ap);
  std::fflush(stdout);
}

// Set by the backend: writes the GPU checkpoint reports of the frames in flight (the abort follows the callback).
std::function<void(const char*)> g_dk_fault_dump;

void DebugCallback(void*, const char* context, DkResult result, const char* message) {
  // A failing command buffer repeats the error on every call (Eden 0.70: a 539 MB log): 32 messages, one dump.
  static uint32_t count = 0;
  static bool dumped = false;
  if (count++ >= 32) return;
  Log("deko3d %s: result %d: %s", context ? context : "?", int(result), message ? message : "");
  std::fflush(stdout);  // deko3d aborts right after an error (console 0.56: the message was lost)
  if (g_dk_fault_dump && !dumped) { dumped = true; g_dk_fault_dump(context ? context : "deko3d error"); }
}

struct MemBlock {
  DkMemBlock block = nullptr;
  uint8_t* cpu = nullptr;
  DkGpuAddr gpu = 0;
  uint32_t size = 0;
  uint32_t used = 0;
  uint32_t peak = 0;
  bool Create(DkDevice device, uint32_t bytes, uint32_t flags) {
    DkMemBlockMaker maker;
    dkMemBlockMakerDefaults(&maker, device, bytes);
    maker.flags = flags;
    block = dkMemBlockCreate(&maker);
    if (!block) return false;
    size = bytes;
    cpu = (flags & DkMemBlockFlags_CpuUncached) || (flags & DkMemBlockFlags_CpuCached)
              ? static_cast<uint8_t*>(dkMemBlockGetCpuAddr(block))
              : nullptr;
    gpu = dkMemBlockGetGpuAddr(block);
    return true;
  }
  // Bump allocation; UINT32_MAX when full.
  uint32_t Alloc(uint32_t bytes, uint32_t align) {
    uint32_t off = (used + align - 1) & ~(align - 1);
    if (off + bytes > size) return UINT32_MAX;
    used = off + bytes;
    if (used > peak) peak = used;
    return off;
  }
  void Reset() { used = 0; }
};

struct HostImage {
  DkImage image{};
  DkImageLayout layout{};
  DkImageFormat format = DkImageFormat_RGBA8_Unorm;
  uint32_t width = 0, height = 0;
  bool is_depth = false;
  uint32_t flags = 0;         // gh: the layout flags (a recycled image must match)
  bool from_resolve = false;  // host-native channels: the fetch swizzle is not applied
  bool rb_swapped = false;    // resolved with RB_COPY_DEST_INFO swap: the fetch swizzle with red and blue exchanged
  bool depth_resolve = false; // an R32F copy of a depth target (ResolveDepth): the k_24_8 / k_24_8_FLOAT texture
  bool reversed_channels = false;  // the format stores the Xenos channels in reverse order (4-4-4-4)
  uint32_t content_serial = 0;  // the draw serial when the contents were set (upload or resolve); 0 = never
  uint64_t mem_hash = 0;        // Linux (no dirty bytes): hash of the guest memory when the contents were set
  uint32_t mem_bytes = 0;       // ... over this many bytes (the same range at every later check)
  uint32_t checked_serial = 0;  // the draw serial of the last dirty-page check
  uint32_t guest_base = 0;    // textures and resolve destinations
  uint32_t edram_base = 0, edram_pitch = 0;  // render targets
  // Last complete, aligned resolve rectangle, for composing adjacent tiled slices on fetch.
  uint32_t resolve_pitch = 0, resolve_height = 0, resolve_frame = 0, resolve_format = 0, resolve_endian = 0;
  uint32_t composed_child_serial = 0;
};

class DkShaderObj : public Shader {
 public:
  DkShaderObj(xenos::ShaderType type, uint64_t hash, const uint32_t* dwords, uint32_t count)
      : Shader(type, hash, dwords, count) {}
  DkShader shader{};
  bool loaded = false;
  DkShader geom{};           // vertex shaders: the rectangle-list geometry shader
  bool geom_loaded = false;
  bool ssbo = false;         // vertex shaders: computed fetch indices; the streams are storage buffers (xenos2glsl .ssbo)
};

struct SysConstants {
  float ndc_scale[4];
  float ndc_offset[4];
  float alpha[4];
  uint32_t bools[8];
  uint32_t loops[32];  // gh: integer loop constants i0..i31 (count | start << 8 | step << 16)
};

struct Frame {
  DkCmdBuf cmdbuf = nullptr;
  uint32_t cmd_offset = 0;
  MemBlock data;     // vertex/index/uniform ring
  MemBlock staging;  // texture uploads
  uint32_t desc_offset = 0;  // image descriptors of this frame, inside `data`? no: a separate static block
  DkFence fence{};
  bool fenced = false;
  uint32_t next_descriptor = 0;
  std::unordered_map<uint64_t, uint32_t> descriptors;  // image (and swizzle) -> descriptor slot this frame
  // The uniform blocks bound last in this frame: reused while the registers behind them do not change
  // (BindUniforms; the copies were 70% of the command thread's draw time in Eden 0.186).
  uint32_t vs_uni = UINT32_MAX, ps_uni = UINT32_MAX, sys_uni = UINT32_MAX;
  const void* vs_uni_shader = nullptr;
  const void* ps_uni_shader = nullptr;
  SysConstants last_sys{};
  // Texture bindings of this frame by fetch constant (+ shader binding): the image, its descriptor+sampler handle
  // and the memory range to re-check (console 0.200: the tex phase was 3.5 ms per frame at ~1.3 bindings per draw).
  struct TexBind {
    HostImage* tex;
    uint32_t handle, base, bytes, width, height;
    bool from_resolve, split_candidate;
  };
  std::unordered_map<uint64_t, TexBind> texbinds;
  // GPU timestamps: pipeline top at the frame's first command, bottom before the present (16-byte reports).
  uint32_t ts_off = UINT32_MAX;
  MemBlock checkpoint_data;  // optional GPU report storage, separate from the frame ring
  uint32_t checkpoint_swap = 0;
  std::vector<std::string> checkpoints;
  bool checkpoints_full = false;
  bool rec_pending = false;  // RECORD 5 SEC: this frame's shown image waits in rec_ for the fence
  uint32_t rec_swap = 0;
  uint64_t rec_us = 0;
  std::vector<std::pair<uint64_t, uint64_t>> draw_log;  // VS, PS of each draw: printed when the frame hangs the GPU
};

// Xenos CompareFunction is 0..7 (never .. always), DkCompareOp is 1..8 in the same order.
DkCompareOp CompareOp(xenos::CompareFunction f) { return DkCompareOp(uint32_t(f) + 1); }

DkBlendFactor BlendFactor(xenos::BlendFactor f) {
  switch (f) {
    case xenos::BlendFactor::kZero: return DkBlendFactor_Zero;
    case xenos::BlendFactor::kOne: return DkBlendFactor_One;
    case xenos::BlendFactor::kSrcColor: return DkBlendFactor_SrcColor;
    case xenos::BlendFactor::kOneMinusSrcColor: return DkBlendFactor_InvSrcColor;
    case xenos::BlendFactor::kSrcAlpha: return DkBlendFactor_SrcAlpha;
    case xenos::BlendFactor::kOneMinusSrcAlpha: return DkBlendFactor_InvSrcAlpha;
    case xenos::BlendFactor::kDstColor: return DkBlendFactor_DstColor;
    case xenos::BlendFactor::kOneMinusDstColor: return DkBlendFactor_InvDstColor;
    case xenos::BlendFactor::kDstAlpha: return DkBlendFactor_DstAlpha;
    case xenos::BlendFactor::kOneMinusDstAlpha: return DkBlendFactor_InvDstAlpha;
    case xenos::BlendFactor::kConstantColor: return DkBlendFactor_ConstColor;
    case xenos::BlendFactor::kOneMinusConstantColor: return DkBlendFactor_InvConstColor;
    case xenos::BlendFactor::kConstantAlpha: return DkBlendFactor_ConstAlpha;
    case xenos::BlendFactor::kOneMinusConstantAlpha: return DkBlendFactor_InvConstAlpha;
    case xenos::BlendFactor::kSrcAlphaSaturate: return DkBlendFactor_SrcAlphaSaturate;
    default: return DkBlendFactor_One;
  }
}

DkBlendOp BlendOp(xenos::BlendOp op) {
  switch (op) {
    case xenos::BlendOp::kAdd: return DkBlendOp_Add;
    case xenos::BlendOp::kSubtract: return DkBlendOp_Sub;
    case xenos::BlendOp::kMin: return DkBlendOp_Min;
    case xenos::BlendOp::kMax: return DkBlendOp_Max;
    case xenos::BlendOp::kRevSubtract: return DkBlendOp_RevSub;
    default: return DkBlendOp_Add;
  }
}

bool PrimitiveOf(xenos::PrimitiveType t, DkPrimitive& out, bool& rects) {
  rects = false;
  switch (t) {
    case xenos::PrimitiveType::kPointList: out = DkPrimitive_Points; return true;
    case xenos::PrimitiveType::kLineList: out = DkPrimitive_Lines; return true;
    case xenos::PrimitiveType::kLineStrip: out = DkPrimitive_LineStrip; return true;
    case xenos::PrimitiveType::kLineLoop: out = DkPrimitive_LineLoop; return true;
    case xenos::PrimitiveType::kTriangleList: out = DkPrimitive_Triangles; return true;
    case xenos::PrimitiveType::kTriangleFan: out = DkPrimitive_TriangleFan; return true;
    case xenos::PrimitiveType::kTriangleStrip: out = DkPrimitive_TriangleStrip; return true;
    case xenos::PrimitiveType::kRectangleList: out = DkPrimitive_Triangles; rects = true; return true;
    case xenos::PrimitiveType::kQuadList: out = DkPrimitive_Quads; return true;
    case xenos::PrimitiveType::kQuadStrip: out = DkPrimitive_QuadStrip; return true;
    case xenos::PrimitiveType::kPolygon: out = DkPrimitive_Polygon; return true;
    default: return false;
  }
}

bool SupportsPrimitiveReset(xenos::PrimitiveType t) {
  switch (t) {
    case xenos::PrimitiveType::kLineStrip: case xenos::PrimitiveType::kTriangleFan:
    case xenos::PrimitiveType::kTriangleStrip: case xenos::PrimitiveType::kLineLoop:
    case xenos::PrimitiveType::kQuadStrip: case xenos::PrimitiveType::kPolygon: return true;
    default: return false;
  }
}

bool VertexAttribOf(const ParsedVertexFetchInstruction::Attributes& a, DkVtxAttribSize& size, DkVtxAttribType& type) {
  using VF = xenos::VertexFormat;
  DkVtxAttribType norm = a.is_integer ? (a.is_signed ? DkVtxAttribType_Sscaled : DkVtxAttribType_Uscaled)
                                      : (a.is_signed ? DkVtxAttribType_Snorm : DkVtxAttribType_Unorm);
  type = norm;
  switch (a.data_format) {
    case VF::k_8_8_8_8: size = DkVtxAttribSize_4x8; return true;
    case VF::k_2_10_10_10: size = DkVtxAttribSize_10_10_10_2; return true;
    case VF::k_16_16: size = DkVtxAttribSize_2x16; return true;
    case VF::k_16_16_16_16: size = DkVtxAttribSize_4x16; return true;
    case VF::k_16_16_FLOAT: size = DkVtxAttribSize_2x16; type = DkVtxAttribType_Float; return true;
    case VF::k_16_16_16_16_FLOAT: size = DkVtxAttribSize_4x16; type = DkVtxAttribType_Float; return true;
    case VF::k_32: size = DkVtxAttribSize_1x32; type = a.is_signed ? DkVtxAttribType_Sscaled : DkVtxAttribType_Uscaled; return true;
    case VF::k_32_32: size = DkVtxAttribSize_2x32; type = a.is_signed ? DkVtxAttribType_Sscaled : DkVtxAttribType_Uscaled; return true;
    case VF::k_32_32_32_32: size = DkVtxAttribSize_4x32; type = a.is_signed ? DkVtxAttribType_Sscaled : DkVtxAttribType_Uscaled; return true;
    case VF::k_32_FLOAT: size = DkVtxAttribSize_1x32; type = DkVtxAttribType_Float; return true;
    case VF::k_32_32_FLOAT: size = DkVtxAttribSize_2x32; type = DkVtxAttribType_Float; return true;
    case VF::k_32_32_32_FLOAT: size = DkVtxAttribSize_3x32; type = DkVtxAttribType_Float; return true;
    case VF::k_32_32_32_32_FLOAT: size = DkVtxAttribSize_4x32; type = DkVtxAttribType_Float; return true;
    case VF::k_10_11_11: case VF::k_11_11_10: size = DkVtxAttribSize_11_11_10; type = DkVtxAttribType_Float; return true;
    default: return false;
  }
}

// Guest texture format -> deko3d format. The packed 16-bit formats need care: deko3d names its formats
// by memory bit order, most significant field first (format_traits.inc in libdeko3d): RGBA4, RGB565 and
// RGB5A1 keep R in the low bits; the Xenos packs R in the high bits (R5G6B5, A1R5G5B5 with R at 14:10) and
// that is the BGR565 / BGR5A1 pair. Xenos 4-4-4-4 is R:G:B:A from the top nibble; deko3d's only 4-bit
// format has the reverse order, so its channels are reversed in the view swizzle (`reversed`).
bool TextureFormatOf(xenos::TextureFormat f, DkImageFormat& out, bool& reversed) {
  using TF = xenos::TextureFormat;
  reversed = false;
  switch (f) {
    case TF::k_8_8_8_8: case TF::k_8_8_8_8_A: out = DkImageFormat_RGBA8_Unorm; return true;
    case TF::k_2_10_10_10: case TF::k_2_10_10_10_AS_16_16_16_16: out = DkImageFormat_RGB10A2_Unorm; return true;
    case TF::k_8: case TF::k_8_A: case TF::k_8_B: out = DkImageFormat_R8_Unorm; return true;
    case TF::k_8_8: out = DkImageFormat_RG8_Unorm; return true;
    case TF::k_DXT1: out = DkImageFormat_RGBA_BC1; return true;
    case TF::k_DXT2_3: out = DkImageFormat_RGBA_BC2; return true;
    case TF::k_DXT4_5: out = DkImageFormat_RGBA_BC3; return true;
    case TF::k_DXN: out = DkImageFormat_RG_BC5_Unorm; return true;
    case TF::k_DXT5A: out = DkImageFormat_R_BC4_Unorm; return true;
    case TF::k_4_4_4_4: out = DkImageFormat_RGBA4_Unorm; reversed = true; return true;
    case TF::k_5_6_5: out = DkImageFormat_BGR565_Unorm; return true;
    case TF::k_1_5_5_5: out = DkImageFormat_BGR5A1_Unorm; return true;
    case TF::k_16_16_16_16_FLOAT: out = DkImageFormat_RGBA16_Float; return true;
    case TF::k_16_16_16_16: out = DkImageFormat_RGBA16_Unorm; return true;
    case TF::k_32_FLOAT: out = DkImageFormat_R32_Float; return true;
    case TF::k_32_32_FLOAT: out = DkImageFormat_RG32_Float; return true;
    case TF::k_32_32_32_32_FLOAT: out = DkImageFormat_RGBA32_Float; return true;
    case TF::k_16_FLOAT: out = DkImageFormat_R16_Float; return true;
    case TF::k_16_16_FLOAT: out = DkImageFormat_RG16_Float; return true;
    case TF::k_16: out = DkImageFormat_R16_Unorm; return true;
    case TF::k_16_16: out = DkImageFormat_RG16_Unorm; return true;
    default: return false;
  }
}

DkImageFormat ColorTargetFormatOf(xenos::ColorRenderTargetFormat f) {
  switch (f) {
    case xenos::ColorRenderTargetFormat::k_8_8_8_8_GAMMA: return DkImageFormat_RGBA8_Unorm_sRGB;
    case xenos::ColorRenderTargetFormat::k_2_10_10_10: return DkImageFormat_RGB10A2_Unorm;
    case xenos::ColorRenderTargetFormat::k_16_16_16_16_FLOAT: return DkImageFormat_RGBA16_Float;
    default: return DkImageFormat_RGBA8_Unorm;
  }
}

// Xenos clamp modes as the Vulkan backend maps them (texture_cache.cpp kAddressModeMap): the half-way
// modes have no host equivalent and use the edge modes, the mirror-clamp ones need MirrorClampToEdge.
DkWrapMode WrapModeOf(xenos::ClampMode m) {
  switch (m) {
    case xenos::ClampMode::kRepeat: return DkWrapMode_Repeat;
    case xenos::ClampMode::kMirroredRepeat: return DkWrapMode_MirroredRepeat;
    case xenos::ClampMode::kClampToEdge: return DkWrapMode_ClampToEdge;
    case xenos::ClampMode::kMirrorClampToEdge: return DkWrapMode_MirrorClampToEdge;
    case xenos::ClampMode::kClampToHalfway: return DkWrapMode_ClampToEdge;
    case xenos::ClampMode::kMirrorClampToHalfway: return DkWrapMode_MirrorClampToEdge;
    case xenos::ClampMode::kClampToBorder: return DkWrapMode_ClampToBorder;
    default: return DkWrapMode_MirrorClampToEdge;
  }
}

// Sampler slot of a fetch: filters as the Vulkan backend resolves them (GetSamplerParameters: the fetch
// instruction's filter unless it says "use the fetch constant"), the clamp modes for the texture's
// dimension, the border colour only where a clamp mode uses it.
uint32_t SamplerIndex(const xenos::xe_gpu_texture_fetch_t& fetch, const ParsedTextureFetchInstruction& instr) {
  xenos::TextureFilter mag = instr.attributes.mag_filter == xenos::TextureFilter::kUseFetchConst ? fetch.mag_filter : instr.attributes.mag_filter;
  xenos::TextureFilter min = instr.attributes.min_filter == xenos::TextureFilter::kUseFetchConst ? fetch.min_filter : instr.attributes.min_filter;
  xenos::ClampMode cx, cy, cz;
  texture_util::GetClampModesForDimension(fetch, cx, cy, cz);
  bool border_white = (xenos::ClampModeUsesBorder(cx) || xenos::ClampModeUsesBorder(cy)) && fetch.border_color == xenos::BorderColor::k_ABGR_White;
  return (min == xenos::TextureFilter::kLinear ? 1u : 0u) | (mag == xenos::TextureFilter::kLinear ? 2u : 0u) | (border_white ? 4u : 0u) |
         (uint32_t(cx) << 3) | (uint32_t(cy) << 6);
}

DkImageSwizzle SwizzleOf(uint32_t s) {
  switch (s & 7) {
    case 0: return DkImageSwizzle_Red;
    case 1: return DkImageSwizzle_Green;
    case 2: return DkImageSwizzle_Blue;
    case 3: return DkImageSwizzle_Alpha;
    case 4: return DkImageSwizzle_Zero;
    default: return DkImageSwizzle_One;
  }
}

// The view swizzle of a resolve image. The image holds the render target's channels. A resolve with
// RB_COPY_DEST_INFO swap writes red and blue exchanged, so the fetch swizzle with its red and blue sources
// exchanged reads them back: 60A gives the identity (0), as the SDK's SwapRedBlueSwizzle does. Without the swap
// (only the depth resolves in this game): the identity.
// ponytail: a colour resolve without the swap also gets the identity, not its fetch swizzle; apply the fetch
// swizzle unchanged there when a game resolves colour that way.
uint32_t ResolveSwizzle(bool rb_swapped, uint32_t fetch_swizzle) {
  if (!rb_swapped) return 0;
  uint32_t r = 0;
  for (uint32_t i = 0; i < 4; i++) {
    uint32_t c = (fetch_swizzle >> (3 * i)) & 7;
    r |= (c == 0 ? 2 : c == 2 ? 0 : c) << (3 * i);
  }
  return r == 0x688 ? 0 : r;
}

class Deko3dCommandProcessor : public CommandProcessor {
 public:
  using CommandProcessor::CommandProcessor;

 protected:
  bool SetupContext() override {
    DkDeviceMaker dm;
    dkDeviceMakerDefaults(&dm);
    dm.cbDebug = DebugCallback;
    device_ = dkDeviceCreate(&dm);
    if (!device_) { Log("dkDeviceCreate failed"); return false; }
    DkQueueMaker qm;
    dkQueueMakerDefaults(&qm, device_);
    qm.flags = DkQueueFlags_Graphics;
    queue_ = dkQueueCreate(&qm);
    if (!queue_) { Log("dkQueueCreate failed"); return false; }
    // The CPU rewrites the descriptors, the frame data ring (uniforms, vertices, indices) and the staging ring at the same
    // addresses every second frame. With GpuCached the GPU's L2 can still hold the old lines (the CPU's uncached writes go
    // to DRAM, nothing invalidates the L2): stale texture headers, constants and upload bytes = the console-only row
    // streaks, wrong glyphs and flicker (Eden has no such cache: the same frames are clean there). GpuUncached reads
    // them from DRAM. The file sdmc:/switch/tmnt/gpucached restores the 0.258 flags for an A/B.
    const bool gpu_cached = access("sdmc:/switch/tmnt/gpucached", F_OK) == 0;
    const uint32_t ring_flags = DkMemBlockFlags_CpuUncached | (gpu_cached ? DkMemBlockFlags_GpuCached : DkMemBlockFlags_GpuUncached);
    std::printf("[nx] gpu: descriptors, frame data and staging %s (file sdmc:/switch/tmnt/gpucached)\n",
                gpu_cached ? "GPU-cached (0.258 behaviour)" : "GPU-uncached");
    if (!code_.Create(device_, kCodePoolBytes, DkMemBlockFlags_CpuUncached | DkMemBlockFlags_GpuCached | DkMemBlockFlags_Code) ||
        !images_.Create(device_, kImagePoolBytes, DkMemBlockFlags_GpuCached | DkMemBlockFlags_Image) ||
        !descriptors_.Create(device_, DK_MEMBLOCK_ALIGNMENT * 4 * kFrames + DK_MEMBLOCK_ALIGNMENT * 2 +
                             ((kSamplerCount * sizeof(DkSamplerDescriptor) + DK_MEMBLOCK_ALIGNMENT - 1) & ~size_t(DK_MEMBLOCK_ALIGNMENT - 1)), ring_flags) ||
        !cmdmem_.Create(device_, kFrameCmdBytes * kFrames, DkMemBlockFlags_CpuUncached | DkMemBlockFlags_GpuCached)) {
      Log("memory block creation failed");
      return false;
    }
    for (uint32_t i = 0; i < kFrames; i++) {
      Frame& f = frames_[i];
      if (!f.data.Create(device_, kFrameDataBytes, ring_flags) ||
          !f.staging.Create(device_, kStagingBytes, ring_flags)) {
        Log("frame %u memory failed", i);
        return false;
      }
      DkCmdBufMaker cm;
      dkCmdBufMakerDefaults(&cm, device_);
      f.cmdbuf = dkCmdBufCreate(&cm);
      f.cmd_offset = i * kFrameCmdBytes;
      f.desc_offset = i * kMaxImageDescriptors * sizeof(DkImageDescriptor);
    }
    sampler_desc_offset_ = kFrames * kMaxImageDescriptors * sizeof(DkImageDescriptor);
    // Samplers, indexed by SamplerIndex: the min and mag filters (the fetch instruction may override the
    // fetch constant's), the border colour (black / white) and every Xenos clamp mode per axis.
    for (uint32_t i = 0; i < kSamplerCount; i++) {
      DkSampler s;
      dkSamplerDefaults(&s);
      s.minFilter = (i & 1) ? DkFilter_Linear : DkFilter_Nearest;
      s.magFilter = (i & 2) ? DkFilter_Linear : DkFilter_Nearest;
      s.mipFilter = DkMipFilter_None;
      s.wrapMode[0] = WrapModeOf(xenos::ClampMode((i >> 3) & 7));
      s.wrapMode[1] = WrapModeOf(xenos::ClampMode((i >> 6) & 7));
      s.wrapMode[2] = DkWrapMode_ClampToEdge;
      if (i == kCubeSampler) {
        s.minFilter = s.magFilter = DkFilter_Linear;
        s.mipFilter = DkMipFilter_Linear;
        s.wrapMode[0] = s.wrapMode[1] = DkWrapMode_ClampToEdge;
      }
      float border = (i & 4) ? 1.0f : 0.0f;
      for (int c = 0; c < 4; c++) s.borderColor[c].value_f = border;
      DkSamplerDescriptor sd;
      dkSamplerDescriptorInitialize(&sd, &s);
      std::memcpy(descriptors_.cpu + sampler_desc_offset_ + i * sizeof(DkSamplerDescriptor), &sd, sizeof(sd));
    }
    // The swapchain on the default window.
    for (uint32_t i = 0; i < kFrames; i++) {
      HostImage* img = CreateImage(kPresentWidth, kPresentHeight, DkImageFormat_RGBA8_Unorm,
                                   DkImageFlags_UsageRender | DkImageFlags_UsagePresent | DkImageFlags_HwCompression, false);
      if (!img) { Log("swapchain image %u failed", i); return false; }
      swap_images_[i] = img;
      swap_image_ptrs_[i] = &img->image;
    }
    DkSwapchainMaker sm;
    dkSwapchainMakerDefaults(&sm, device_, nwindowGetDefault(), swap_image_ptrs_, kFrames);
    swapchain_ = dkSwapchainCreate(&sm);
    if (!swapchain_) { Log("dkSwapchainCreate failed"); return false; }
    shader_dir_ = REXCVAR_GET(deko3d_shader_dir);
    if (shader_dir_.empty() || shader_dir_.back() != '/') shader_dir_ += '/';
    size_t cut = shader_dir_.size() >= 2 ? shader_dir_.rfind('/', shader_dir_.size() - 2) : std::string::npos;
    std::string debug_dir = cut == std::string::npos ? "" : shader_dir_.substr(0, cut + 1);
    present_target_ = access((debug_dir + "dk_present_rt").c_str(), F_OK) == 0;
    dump_images_ = access((debug_dir + "dk_dump").c_str(), F_OK) == 0;  // readback dumps stall the GPU: debug only
    debug_dir_ = debug_dir;
    gpu_checkpoints_ = access((debug_dir + "dk_gpu_checkpoints").c_str(), F_OK) == 0;
#ifdef TMNT_DEKO3D_DEBUG_LIB
    Log("deko3d debug library (deko3dd): errors are reported through the debug callback before the abort");
#endif
    g_dk_fault_dump = [this](const char* why) { DumpFencedFrames(why); };
    if (gpu_checkpoints_) {
      Log("GPU checkpoints enabled; dk_skip is ignored");
      for (Frame& f : frames_) {
        if (!f.checkpoint_data.Create(device_, kMaxCheckpoints * 16, DkMemBlockFlags_CpuUncached | DkMemBlockFlags_GpuUncached)) {
          Log("GPU checkpoint buffer allocation failed");
          return false;
        }
      }
    }
    if (FILE* sf = gpu_checkpoints_ ? nullptr : std::fopen((debug_dir + "dk_skip").c_str(), "r")) {
      unsigned long long v, p;
      while (std::fscanf(sf, "%llx %llx", &v, &p) == 2) skip_pairs_.insert(PairKey({v, p}));
      std::fclose(sf);
      Log("dk_skip: the draws of %zu shader pairs are skipped (suspects of a GPU fault)", skip_pairs_.size());
    }
    ReadFpsMode();
    trace_start_ms_ = NowMs();
    trace_enabled_ = access((debug_dir + "dk_trace").c_str(), F_OK) == 0;
    if (trace_enabled_) {
      if (FILE* tf = std::fopen((debug_dir + "dk_trace").c_str(), "r")) {  // optional: the first swap to trace
        unsigned n = 0;
        char word[8] = {};
        if (std::fscanf(tf, "%7[a-z]", word) == 1) { trace_on_mip_ = std::strcmp(word, "mip") == 0; trace_any_ = std::strcmp(word, "any") == 0; }  // mip: the first mip-only frame; any: no draw-count gate (menus)
        if (std::fscanf(tf, "%u", &n) == 1) trace_min_swap_ = n;
        if (std::fscanf(tf, "%u", &n) == 1) trace_second_ = n;  // optional: a second frame to trace, seconds (or, with a first time of 0, swaps after the first ring-full event)
        std::fclose(tf);
        if (trace_min_swap_ == 0 && trace_second_) { trace_after_full_ = trace_second_; trace_second_ = 0; }
      }
    }
    present_vs_ = LoadDksh(shader_dir_ + "present.vert.dksh", present_vs_shader_);
    present_fs_ = LoadDksh(shader_dir_ + "present.frag.dksh", present_fs_shader_);
    depth_fs_ = LoadDksh(shader_dir_ + "depthcopy.frag.dksh", depth_fs_shader_);
    if (!depth_fs_) Log("no depthcopy.frag.dksh: depth resolves are ignored");
    restore_fs_ = LoadDksh(shader_dir_ + "depthrestore.frag.dksh", restore_fs_shader_);
    if (!restore_fs_) Log("no depthrestore.frag.dksh: depth restore draws are skipped");
    Log("deko3d backend up: shaders from %s (%s present shaders), %u MB images, %u MB data per frame",
        shader_dir_.c_str(), present_vs_ && present_fs_ ? "with" : "WITHOUT", kImagePoolBytes >> 20, kFrameDataBytes >> 20);
    BeginFrame();
    return true;
  }

  void ShutdownContext() override {
    if (queue_) dkQueueWaitIdle(queue_);
    Log("deko3d backend down: %u swaps, %u draws, %u skipped, %u without shaders, %u clears, %u resolves, %u images (%u MB), %u uploads (%u KB, %u re-uploads), %llu dirty pages",
        swaps_, draws_, skipped_, missing_draws_, clears_, resolves_, image_count_, images_.peak >> 20, uploads_,
        uint32_t(upload_bytes_ >> 10), reuploads_, (unsigned long long)dirty_pages_);
  }

  Shader* LoadShader(xenos::ShaderType shader_type, uint32_t guest_address, const uint32_t* host_address,
                     uint32_t dword_count) override {
    uint64_t hash = tmnt::UcodeHash(host_address, dword_count);
    auto it = shaders_.find(hash);
    if (it != shaders_.end()) return it->second;
    DkShaderObj* shader = new DkShaderObj(shader_type, hash, host_address, dword_count);
    string::StringBuffer disasm;
    shader->AnalyzeUcode(disasm);
    shaders_.emplace(hash, shader);
    char name[512];
    std::snprintf(name, sizeof(name), "%sshader_%016llX.%s.dksh", shader_dir_.c_str(), (unsigned long long)hash,
                  shader_type == xenos::ShaderType::kVertex ? "vert" : "frag");
    shader->loaded = LoadDksh(name, shader->shader);
    if (shader->loaded && shader_type == xenos::ShaderType::kVertex) {
      std::snprintf(name, sizeof(name), "%sshader_%016llX.geom.dksh", shader_dir_.c_str(), (unsigned long long)hash);
      shader->geom_loaded = LoadDksh(name, shader->geom);
      if (!shader->geom_loaded) LogOnce("geom", "no rectangle-list geometry shader for VS %016llX (%s)", (unsigned long long)hash, name);
      std::snprintf(name, sizeof(name), "%sshader_%016llX.vert.ssbo", shader_dir_.c_str(), (unsigned long long)hash);
      shader->ssbo = access(name, F_OK) == 0;
      if (shader->ssbo) Log("VS %016llX fetches by computed indices: storage buffers", (unsigned long long)hash);
    }
    if (!shader->loaded) {
      missing_shaders_++;
      Log("no DKSH for %s %016llX (%u dwords at %08X): dumping the ucode to %smissing/",
          shader_type == xenos::ShaderType::kVertex ? "VS" : "PS", (unsigned long long)hash, dword_count, guest_address,
          shader_dir_.c_str());
      DumpMissing(*shader);
    } else {
      REXLOG_INFO("[dk] shader {} {:016X} loaded ({} dwords)", shader_type == xenos::ShaderType::kVertex ? "VS" : "PS",
                  hash, dword_count);
    }
    return shader;
  }

  // Time spent in the backend per second (printed with the fps window): draws and resolves, the swap, and
  // the two waits that tell a GPU-bound frame from a CPU-bound one.
  static double NowMs() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
  }
  bool IssueDraw(xenos::PrimitiveType prim_type, uint32_t index_count, IndexBufferInfo* index_buffer_info,
                 bool major_mode_explicit) override {
    const double t0 = NowMs();
    const bool ok = IssueDrawImpl(prim_type, index_count, index_buffer_info, major_mode_explicit);
    t_draw_ms_ += NowMs() - t0;
    return ok;
  }
  bool IssueDrawImpl(xenos::PrimitiveType prim_type, uint32_t index_count, IndexBufferInfo* index_buffer_info,
                     bool major_mode_explicit) {
    if (Skipping()) return true;
    // Phase timers (the dk time/s line): sync, setup, textures, targets+state, vertices, uniforms, draw.
    double phase_t = NowMs();
    auto phase = [&](int i) { const double n = NowMs(); t_phase_ms_[i] += n - phase_t; phase_t = n; };
    SyncDirtyPages();
    phase(0);
    const RegisterFile& regs = *register_file_;
    auto mode = regs.Get<reg::RB_MODECONTROL>().edram_mode;
    if (mode == xenos::EdramMode::kCopy) return Resolve();
    auto* vs = static_cast<DkShaderObj*>(active_vertex_shader());
    auto* ps = static_cast<DkShaderObj*>(active_pixel_shader());
    if (!vs || !ps) { skipped_++; return true; }
    auto mask = regs.Get<reg::RB_COLOR_MASK>();
    auto dc = draw_util::GetNormalizedDepthControl(regs);
    const uint32_t color_mask = draw_util::GetNormalizedColorMask(regs, ps->writes_color_targets());
    bool writes_color = color_mask != 0;
    bool writes_depth = dc.z_enable && dc.z_write_enable;
    if (!writes_color && !writes_depth && !dc.stencil_enable) { skipped_++; return true; }
    if (IsClearDraw(vs, ps, prim_type)) return ClearDraw(index_count, mode);
    if (!vs->loaded || !ps->loaded) { missing_draws_++; frame_missing_++; return true; }
    DkPrimitive prim;
    bool rects;
    if (!PrimitiveOf(prim_type, prim, rects)) { LogOnce("prim", "unsupported primitive type %u", uint32_t(prim_type)); skipped_++; return true; }
    Frame& f = frames_[frame_];
    DkCmdBuf cmd = f.cmdbuf;
    const uint64_t pair_key = vs->ucode_data_hash() * 0x9E3779B97F4A7C15ull ^ ps->ucode_data_hash();
    if (FirstTime(pair_key)) {
      // Each shader pair's first draw, once: the state a new kind of draw comes with.
      auto blend = regs.Get<reg::RB_BLENDCONTROL>();
      {
        Log("draw VS %016llX PS %016llX: prim %u n=%u%s, blend %u/%u op %u, mask %04x shader targets %x, z %u/%u func %u, cull f%u b%u face %u, %zu vfetch %zu tfetch",
                (unsigned long long)vs->ucode_data_hash(), (unsigned long long)ps->ucode_data_hash(), uint32_t(prim_type), index_count,
                index_buffer_info ? " indexed" : "", uint32_t(blend.color_srcblend), uint32_t(blend.color_destblend),
                uint32_t(blend.color_comb_fcn), mask.value & 0xFFFF, ps->writes_color_targets(), dc.z_enable, dc.z_write_enable,
                uint32_t(dc.zfunc), regs.Get<reg::PA_SU_SC_MODE_CNTL>().cull_front, regs.Get<reg::PA_SU_SC_MODE_CNTL>().cull_back, regs.Get<reg::PA_SU_SC_MODE_CNTL>().face, vs->vertex_bindings().size(), ps->texture_bindings().size());
        for (uint32_t i = 0; i < 4; i++) {
          if (!((color_mask >> (i * 4)) & 0xF)) continue;
          auto ci = regs.Get<reg::RB_COLOR_INFO>(reg::RB_COLOR_INFO::rt_register_indices[i]);
          Log("  rt%u base %u format %u mask %x", i, ci.color_base, uint32_t(ci.color_format), (color_mask >> (i * 4)) & 0xF);
        }
        // Its vertex layout and the first vertices (byte-swapped words as hex and as floats).
        for (const auto& vb : vs->vertex_bindings()) {
          auto fetch = regs.GetVertexFetch(vb.fetch_constant);
          std::string attrs;
          for (const auto& a : vb.attributes) {
            char s[64];
            std::snprintf(s, sizeof(s), " [fmt %u off %u%s%s]", uint32_t(a.fetch_instr.attributes.data_format), a.fetch_instr.attributes.offset,
                          a.fetch_instr.attributes.is_signed ? " signed" : "", a.fetch_instr.attributes.is_integer ? " integer" : "");
            attrs += s;
          }
          const uint32_t* w = reinterpret_cast<const uint32_t*>(memory_->TranslatePhysical(fetch.address << 2));
          uint32_t n = std::min<uint32_t>(vb.stride_words * 3, 24);
          std::string words;
          for (uint32_t i = 0; i < n && i < fetch.size; i++) {
            uint32_t v = fetch.endian == xenos::Endian::k8in32 ? __builtin_bswap32(w[i]) : w[i];
            float fl;
            std::memcpy(&fl, &v, 4);
            char s[48];
            std::snprintf(s, sizeof(s), " %08X(%g)", v, fl);
            words += s;
          }
          Log("  vf%u stride %u words, %u words at %08X endian %u:%s; data%s", vb.fetch_constant, vb.stride_words, fetch.size, fetch.address << 2,
              uint32_t(fetch.endian), attrs.c_str(), words.c_str());
        }
      }
    }

    phase(1);
    // Textures of the pixel shader (before the render targets: a new image is cleared on creation, which
    // rebinds the targets).
    std::string bound_textures;  // for the draw-type log below (built only when the trace or a first draw needs it)
    struct TexName { uint32_t fc, base, w, h; } tnames[32];
    uint32_t tcount = 0;
    uint64_t texset_key = pair_key;
    bool textures_ok = true;
    bool depth_alias = false;  // a depth resolve image fetched as a colour texture: the XDK depth restore
    uint32_t tb_index = 0;
    for (const auto& tb : ps->texture_bindings()) {
      auto fetch = regs.GetTextureFetch(tb.fetch_constant);
      const uint32_t* fw = reinterpret_cast<const uint32_t*>(&fetch);
      uint64_t bind_key = ps->ucode_data_hash() ^ (uint64_t(tb_index++) * 0x9E3779B97F4A7C15ull);
      for (int i = 0; i < 6; i++) bind_key = (bind_key ^ fw[i]) * 0x100000001B3ull;
      if (fetch.dimension == xenos::DataDimension::kCube) {
        // Not through the per-frame bind cache: CubeImage checks its 6 x levels sources itself.
        TextureInfo info;
        HostImage* tex = TextureInfo::Prepare(fetch, &info) ? CubeImage(f, info, fetch) : nullptr;
        uint32_t slot = tex ? DescriptorFor(f, tex, tex->from_resolve ? ResolveSwizzle(tex->rb_swapped, fetch.swizzle) : fetch.swizzle) : UINT32_MAX;
        if (slot == UINT32_MAX) { textures_ok = false; continue; }
        DkResHandle handle = dkMakeTextureHandle(slot, kCubeSampler);
        dkCmdBufBindTextures(cmd, DkStage_Fragment, tb.fetch_constant, &handle, 1);
        texset_key = (texset_key ^ info.memory.base_address) * 0x100000001B3ull + tb.fetch_constant;
        if (tcount < 32) tnames[tcount++] = TexName{tb.fetch_constant, info.memory.base_address, tex->width, tex->height};
        continue;
      }
      if (auto it = f.texbinds.find(bind_key); it != f.texbinds.end()) {
        Frame::TexBind& c = it->second;
        HostImage* tex = c.tex;
        bool valid = tex->guest_base == c.base && tex->width == c.width && tex->height == c.height && tex->from_resolve == c.from_resolve;
        if (valid && tex->checked_serial != serial_) {
          if (MemoryChanged(*tex, c.base, c.split_candidate ? tex->mem_bytes : c.bytes)) valid = false; else tex->checked_serial = serial_;
        }
        if (valid) {
          if (tex->depth_resolve && !IsDepthTexture(fetch.format)) depth_alias = true;
          if (c.split_candidate) ComposeAdjacentResolve(f, *tex);
          dkCmdBufBindTextures(cmd, DkStage_Fragment, tb.fetch_constant, &c.handle, 1);
          texset_key = (texset_key ^ c.base) * 0x100000001B3ull + tb.fetch_constant;
          if (tcount < 32) tnames[tcount++] = TexName{tb.fetch_constant, c.base, tex->width, tex->height};
          texbind_hits_++;
          continue;
        }
        f.texbinds.erase(it);
      }
      TextureInfo info;
      // 1D textures are 2D textures of height 1 on the Xenos (the shaders sample them so).
      if (!TextureInfo::Prepare(fetch, &info) ||
          (info.dimension != xenos::DataDimension::k2DOrStacked && info.dimension != xenos::DataDimension::k1D)) {
        LogOnce("tex", "texture fetch %u not preparable or not 1D/2D (dim %u)", tb.fetch_constant, uint32_t(fetch.dimension));
        textures_ok = false;
        continue;
      }
      HostImage* tex = TextureImage(f, info, fetch);
      if (!tex) { textures_ok = false; continue; }  // e.g. the staging ring is full this frame: draw next frame
      if (tex->depth_resolve && !IsDepthTexture(fetch.format)) depth_alias = true;
      // A depth resolve image holds host values (never guest bytes): the fetch's tiling, format and byte order
      // say nothing about it, so the two slices of the scene depth compose whatever the fetch declares.
      const bool split_candidate = tex->from_resolve && tex->width > 32 && tex->height > 32 &&
                                   info.dimension == xenos::DataDimension::k2DOrStacked && !info.is_stacked &&
                                   info.memory.base_address == tex->guest_base && info.pitch == tex->width &&
                                   info.format_info()->bytes_per_block() == 4 &&
                                   (tex->depth_resolve || (info.is_tiled && ResolveFetchFormatMatches(tex->resolve_format, info.format) &&
                                                           uint32_t(info.endianness) == tex->resolve_endian));
      if (tex->resolve_height && tex->resolve_height < tex->height) {
        char pkey[40];
        std::snprintf(pkey, sizeof(pkey), "resolvepartscheck %08X", tex->guest_base);
        LogOnce(pkey, "split fetch %08X %ux%u tiled %u pitch %u format %u endian %u; resolve pitch %u rows %u format %u endian %u eligible %u",
                tex->guest_base, tex->width, tex->height, info.is_tiled, info.pitch, uint32_t(info.format), uint32_t(info.endianness),
                tex->resolve_pitch, tex->resolve_height, tex->resolve_format, tex->resolve_endian, split_candidate);
      }
      if (split_candidate) ComposeAdjacentResolve(f, *tex);
      uint32_t swizzle = tex->from_resolve ? ResolveSwizzle(tex->rb_swapped, fetch.swizzle) : fetch.swizzle;
      if (swizzle && tex->reversed_channels) {
        uint32_t r = 0;
        for (uint32_t i = 0; i < 4; i++) {
          uint32_t c = (swizzle >> (3 * i)) & 7;
          r |= (c < 4 ? 3 - c : c) << (3 * i);
        }
        swizzle = r;
      }
      uint32_t slot = DescriptorFor(f, tex, swizzle);
      if (slot == UINT32_MAX) { textures_ok = false; continue; }
      uint32_t sampler = SamplerIndex(fetch, tb.fetch_instr);
      if (FirstTime((uint64_t(info.memory.mip_address) << 32) ^ info.memory.base_address ^ 0x7465780000000000ull)) {
        // Each texture's first bind, once: what the game draws with (correlates with the uploads).
        Log("texture at %08X: %s %ux%u %s endian %u swizzle %03X filters min %u mag %u (instr %u/%u) clamp %u/%u (%u KB) mips %u-%u at %08X%s",
                info.memory.base_address, info.format_info()->name, info.width + 1, info.height + 1, info.is_tiled ? "tiled" : "linear",
                uint32_t(fetch.endianness), fetch.swizzle, uint32_t(fetch.min_filter), uint32_t(fetch.mag_filter),
                uint32_t(tb.fetch_instr.attributes.min_filter), uint32_t(tb.fetch_instr.attributes.mag_filter), uint32_t(fetch.clamp_x),
                uint32_t(fetch.clamp_y), info.memory.base_size >> 10, info.mip_min_level, info.mip_max_level, info.memory.mip_address,
                tex->from_resolve ? " (a resolve image)" : "");
      }
      DkResHandle handle = dkMakeTextureHandle(slot, sampler);
      dkCmdBufBindTextures(cmd, DkStage_Fragment, tb.fetch_constant, &handle, 1);
      f.texbinds[bind_key] = Frame::TexBind{tex, handle, tex->guest_base, tex->mem_bytes, tex->width, tex->height, tex->from_resolve, split_candidate};
      texset_key = (texset_key ^ info.memory.base_address) * 0x100000001B3ull + tb.fetch_constant;
      if (tcount < 32) tnames[tcount++] = TexName{tb.fetch_constant, info.memory.base_address, tex->width, tex->height};
    }
    const bool first_texset = FirstTime(texset_key ^ 0x7061697274657800ull);
    if (first_texset || (trace_enabled_ && !traced_)) {
      for (uint32_t i = 0; i < tcount; i++) {
        char t[48];
        std::snprintf(t, sizeof(t), " tf%u=%08X(%ux%u)", tnames[i].fc, tnames[i].base, tnames[i].w, tnames[i].h);
        bound_textures += t;
      }
    }
    if (first_texset) {
      // Each shader pair's first draw with each texture set, once (the texture bases name the layer).
      Log("draw VS %016llX PS %016llX prim %u n=%u:%s", (unsigned long long)vs->ucode_data_hash(),
          (unsigned long long)ps->ucode_data_hash(), uint32_t(prim_type), index_count, bound_textures.empty() ? " no textures" : bound_textures.c_str());
    }
    // A draw with a missing texture would sample whatever the slot held before (noise): skip it.
    if (!textures_ok) { skipped_++; return true; }
    // The XDK depth restore (Eden 0.75: "rt0 base 0 format 0" = the depth buffer's EDRAM base, tf0 = the saved depth
    // 1F360000 fetched as k_8_8_8_8, between the two tiles and before the lighting passes): the guest writes the
    // saved depth's raw bytes through a colour alias of the depth buffer. The host writes the R32F copy's values
    // into the depth target at that base (depthrestore.frag in place of the guest pixel shader). Another shader
    // sampling a depth image as colour would draw noise: skipped.
    const bool depth_restore = depth_alias;
    if (depth_restore && (!restore_fs_ || ps->ucode_data_hash() != 0x894498C723463852ull || mode == xenos::EdramMode::kDepthOnly)) {
      char key[48];
      std::snprintf(key, sizeof(key), "depthalias %016llX", (unsigned long long)ps->ucode_data_hash());
      LogOnce(key, "draw PS %016llX samples a depth resolve image as colour: skipped (no depth restore for it)",
              (unsigned long long)ps->ucode_data_hash());
      skipped_++;
      return true;
    }
    phase(2);

    // Render targets.
    auto surf = regs.Get<reg::RB_SURFACE_INFO>();
    auto depth_info = regs.Get<reg::RB_DEPTH_INFO>();
    uint32_t pitch = EffectivePitch(surf);
    HostImage* colors[4] = {};
    HostImage* color = nullptr;
    if (mode != xenos::EdramMode::kDepthOnly && !depth_restore) {
      for (uint32_t i = 0; i < 4; i++) {
        if (!((color_mask >> (i * 4)) & 0xF)) continue;
        auto ci = regs.Get<reg::RB_COLOR_INFO>(reg::RB_COLOR_INFO::rt_register_indices[i]);
        colors[i] = TargetImage(ci.color_base, pitch, ColorTargetFormatOf(ci.color_format), false);
        if (!colors[i]) { skipped_++; return true; }
        if (!color) color = colors[i];
      }
    }
    HostImage* depth = (dc.z_enable || dc.stencil_enable) ? TargetImage(depth_info.depth_base, pitch, DkImageFormat_Z24S8, true) : nullptr;
    if (depth_restore) {
      auto ci = regs.Get<reg::RB_COLOR_INFO>(reg::RB_COLOR_INFO::rt_register_indices[0]);
      depth = TargetImage(ci.color_base, pitch, DkImageFormat_Z24S8, true);
      char key[64];
      std::snprintf(key, sizeof(key), "depthrestore %u %u", uint32_t(ci.color_base), pitch);
      LogOnce(key, "depth restore: the depth image drawn into the depth target at EDRAM %u pitch %u (guest depth base %u)",
              uint32_t(ci.color_base), pitch, uint32_t(depth_info.depth_base));
      // The constants of the guest shaders, once per texel offset: position c240, address c241, the half swap c255.
      auto cf = [&](uint32_t reg, uint32_t k) { return regs.Get<float>(XE_GPU_REG_SHADER_CONSTANT_000_X + reg * 4 + k); };
      std::snprintf(key, sizeof(key), "depthrestore consts %g %g", cf(241, 2), cf(241, 3));
      LogOnce(key, "depth restore constants: vs c240 %g %g %g %g, c241 %g %g %g %g, c255 %g %g; ps c255 %g %g %g", cf(240, 0), cf(240, 1),
              cf(240, 2), cf(240, 3), cf(241, 0), cf(241, 1), cf(241, 2), cf(241, 3), cf(255, 0), cf(255, 1), cf(511, 0), cf(511, 1), cf(511, 2));
    }
    if (!color && !depth) { skipped_++; return true; }
    if (!BindTargetSet(cmd, colors, depth)) { skipped_++; return true; }
    uint32_t target_w = color ? color->width : depth->width;
    uint32_t target_h = color ? color->height : depth->height;

    // Viewport and scissor (the window offset included, as the SDK computes them).
    // deko3d (upper-left origin) maps NDC y = +1 to the bottom row like Vulkan; the Xenos clip space is
    // y-up like Direct3D, so the SDK's "origin_bottom_left" negation of the NDC y scale/offset applies
    // (the first Eden picture was upside down without it).
    draw_util::ViewportInfo vi;
    draw_util::GetHostViewportInfo(regs, 1, 1, true, target_w, target_h, false, dc, false, false, false, vi);
    DkViewport vp{float(vi.xy_offset[0]), float(vi.xy_offset[1]), float(vi.xy_extent[0]), float(vi.xy_extent[1]), vi.z_min, vi.z_max};
    dkCmdBufSetViewports(cmd, 0, &vp, 1);
    draw_util::Scissor sc;
    draw_util::GetScissor(regs, sc, true);
    DkScissor scissor{sc.offset[0], sc.offset[1], sc.extent[0], sc.extent[1]};
    if (scissor.x + scissor.width > target_w) scissor.width = scissor.x < target_w ? target_w - scissor.x : 0;
    if (scissor.y + scissor.height > target_h) scissor.height = scissor.y < target_h ? target_h - scissor.y : 0;
    dkCmdBufSetScissors(cmd, 0, &scissor, 1);

    // Fixed-function state.
    auto su = regs.Get<reg::PA_SU_SC_MODE_CNTL>();
    // Rectangle lists: the geometry shader makes the 4th vertex, or the hardware mode (cvar).
    const bool use_gs = rects && vs->geom_loaded && !REXCVAR_GET(deko3d_fill_rectangle);
    DkRasterizerState rs;
    dkRasterizerStateDefaults(&rs);
    rs.cullMode = DkFace_None;
    // Face culling for triangles: the book's page turn draws the page mesh twice (front faces with the
    // front texture, back faces with the mirrored one); without culling the left page came out mirrored
    // (console 0.141). Xenos face 0 = front is CCW as seen on screen; the cvar inverts it if the picture says so.
    if (REXCVAR_GET(deko3d_cull) != 0 && (prim_type == xenos::PrimitiveType::kTriangleList ||
                                          prim_type == xenos::PrimitiveType::kTriangleStrip ||
                                          prim_type == xenos::PrimitiveType::kTriangleFan ||
                                          prim_type == xenos::PrimitiveType::kQuadList || prim_type == xenos::PrimitiveType::kQuadStrip)) {
      rs.cullMode = su.cull_front ? (su.cull_back ? DkFace_FrontAndBack : DkFace_Front)
                                  : (su.cull_back ? DkFace_Back : DkFace_None);
      const bool front_cw = (su.face != 0) != (REXCVAR_GET(deko3d_cull) == 2);
      rs.frontFace = front_cw ? DkFrontFace_CW : DkFrontFace_CCW;
    }
    rs.fillRectangleEnable = rects && !use_gs;
    dkCmdBufBindRasterizerState(cmd, &rs);
    DkColorState cs;
    dkColorStateDefaults(&cs);
    DkColorWriteState cw;
    dkColorWriteStateDefaults(&cw);
    cw.masks = mode == xenos::EdramMode::kDepthOnly ? 0 : color_mask;
    DkBlendState bs[4];
    for (uint32_t i = 0; i < 4; i++) {
      auto b = regs.Get<reg::RB_BLENDCONTROL>(reg::RB_BLENDCONTROL::rt_register_indices[i]);
      bool enabled = !(b.color_srcblend == xenos::BlendFactor::kOne && b.color_destblend == xenos::BlendFactor::kZero &&
                       b.color_comb_fcn == xenos::BlendOp::kAdd && b.alpha_srcblend == xenos::BlendFactor::kOne &&
                       b.alpha_destblend == xenos::BlendFactor::kZero && b.alpha_comb_fcn == xenos::BlendOp::kAdd);
      if (enabled) cs.blendEnableMask |= 1u << i;
      bs[i] = DkBlendState{BlendOp(b.color_comb_fcn), BlendFactor(b.color_srcblend), BlendFactor(b.color_destblend),
                           BlendOp(b.alpha_comb_fcn), BlendFactor(b.alpha_srcblend), BlendFactor(b.alpha_destblend)};
    }
    dkCmdBufBindColorState(cmd, &cs);
    dkCmdBufBindColorWriteState(cmd, &cw);
    dkCmdBufBindBlendStates(cmd, 0, bs, 4);
    DkDepthStencilState ds;
    dkDepthStencilStateDefaults(&ds);
    ds.depthTestEnable = depth && dc.z_enable;
    ds.depthWriteEnable = depth && dc.z_write_enable;
    ds.depthCompareOp = CompareOp(dc.zfunc);
    if (depth_restore) {
      ds.depthTestEnable = 1;  // depth writes need the test on
      ds.depthWriteEnable = 1;
      ds.depthCompareOp = DkCompareOp_Always;
    }
    ds.stencilTestEnable = 0;
    dkCmdBufBindDepthStencilState(cmd, &ds);

    if (!skip_pairs_.empty() && skip_pairs_.count(PairKey({vs->ucode_data_hash(), ps->ucode_data_hash()}))) { skipped_++; return true; }
    // Shaders (the whole graphics mask: a stage without a shader here is unbound).
    DkShader const* shaders[3] = {&vs->shader, depth_restore ? &restore_fs_shader_ : &ps->shader, &vs->geom};
    dkCmdBufBindShaders(cmd, DkStageFlag_GraphicsMask, shaders, use_gs ? 3 : 2);
    phase(3);

    // Vertex layout and data: one buffer per fetch constant the vertex shader uses.
    DkVtxAttribState attribs[DK_MAX_VERTEX_ATTRIBS];
    DkVtxBufferState buffers[DK_MAX_VERTEX_BUFFERS];
    uint32_t attrib_count = 0, buffer_count = 0, ssbo_count = 0;
    uint32_t vertex_count = index_count;
    uint32_t min_index = 0;  // the lowest vertex index the draw reads
    uint32_t index_offset = UINT32_MAX;
    uint32_t restart_index = regs.Get<reg::VGT_MULTI_PRIM_IB_RESET_INDX>().reset_indx;
    bool restart = index_buffer_info && su.multi_prim_ib_ena && SupportsPrimitiveReset(prim_type) &&
                   (index_buffer_info->format != xenos::IndexFormat::kInt16 || restart_index <= UINT16_MAX);
    if (index_buffer_info) {
      uint32_t restart_hits = 0;
      if (!CopyIndices(f, *index_buffer_info, restart, restart_index, index_offset, vertex_count, min_index, restart_hits)) { skipped_++; return true; }
      if (restart_hits) LogOnce("primrestart", "primitive restart VS %016llX PS %016llX prim %u index %06X: %u of %u indices",
                                (unsigned long long)vs->ucode_data_hash(), (unsigned long long)ps->ucode_data_hash(),
                                uint32_t(prim_type), restart_index, restart_hits, index_buffer_info->count);
      if (!vertex_count) { skipped_++; return true; }
    }
    // VGT_INDX_OFFSET (Direct3D BaseVertexIndex): the guest fetches vertex (index + offset), in 24 bits.
    // ponytail: VGT_MIN_VTX_INDX / VGT_MAX_VTX_INDX (the SDK clamps the sum to them) are not read.
    const int32_t indx_offset = int32_t(regs.Get<reg::VGT_INDX_OFFSET>().indx_offset << 8) >> 8;
    if (indx_offset) LogOnce("indxoffset", "VGT_INDX_OFFSET %d: VS %016llX n=%u indices %u-%u", indx_offset,
                             (unsigned long long)vs->ucode_data_hash(), index_count, min_index, vertex_count - 1);
    // An attribute stream holds only the vertices the draw reads, [first_vertex, first_vertex + count): the draw's
    // vertex offset is -min_index. A copy from vertex 0 grows with the highest index, whatever the draw reads (Eden
    // 0.78, level 1: the vertex phase grew to 2-4 s a frame, the 24 MB frame ring filled, draws were dropped; the
    // index values themselves were not logged then: the "vertex copy" line below measures them).
    // A storage-buffer shader computes its record addresses from gl_VertexID, which must stay the guest index: its
    // streams still start at vertex 0. ponytail: (max index + 1) records per stream there; bound the instance
    // stream by the shader's own divisor (c19.x) if the frame ring still fills.
    const uint32_t first_vertex = vs->ssbo ? 0 : uint32_t(std::max<int64_t>(0, int64_t(min_index) + indx_offset));
    const uint32_t stream_vertices = vs->ssbo ? uint32_t(std::max<int64_t>(0, int64_t(vertex_count) + indx_offset))
                                              : vertex_count - min_index;
    uint32_t copied_bytes = 0;
    for (const auto& vb : vs->vertex_bindings()) {
      if (buffer_count >= DK_MAX_VERTEX_BUFFERS) break;
      auto fetch = regs.GetVertexFetch(vb.fetch_constant);
      uint32_t stride = vb.stride_words * 4;
      if (stride && stream_vertices > UINT32_MAX / stride) {
        LogOnce("vbsize", "vertex data size overflow: %u vertices x %u bytes", stream_vertices, stride);
        skipped_++; return true;
      }
      uint32_t bytes = stream_vertices * stride;
      const uint32_t skip = uint32_t(std::min<uint64_t>(uint64_t(first_vertex) * stride, uint64_t(fetch.size) * 4));
      uint32_t avail = fetch.size * 4 - skip;
      if (!stride) {
        // A constant stream still needs one record, regardless of the vertex index.
        uint32_t extent_words = 0;
        for (const auto& a : vb.attributes) {
          const auto& attributes = a.fetch_instr.attributes;
          DkVtxAttribSize size;
          DkVtxAttribType type;
          if (!VertexAttribOf(attributes, size, type)) {
            LogOnce("vfmt", "unsupported vertex format %u", uint32_t(attributes.data_format));
            skipped_++; return true;
          }
          if (attributes.offset < 0 || attributes.offset > 0x3FFF / 4) {
            LogOnce("vbzero-offset", "zero-stride vertex fetch %u has invalid offset %d words",
                    vb.fetch_constant, attributes.offset);
            skipped_++; return true;
          }
          const uint32_t words = xenos::GetVertexFormatNeededWords(attributes.data_format, 0b1111);
          if (!words) { skipped_++; return true; }
          extent_words = std::max(extent_words, uint32_t(attributes.offset) + uint32_t(std::bit_width(words)));
        }
        if (!extent_words || extent_words > fetch.size) {
          LogOnce("vbzero", "zero-stride vertex fetch %u has invalid extent %u words (available %u)", vb.fetch_constant,
                  extent_words, fetch.size);
          skipped_++; return true;
        }
        bytes = extent_words * 4;
      }
      if (bytes > avail) {
        LogOnce("vbclamp", "vertex data clamped: %u vertices from %u x %u bytes = %u, fetch constant %u holds %u bytes (VS %016llX)", stream_vertices,
                first_vertex, stride, bytes, vb.fetch_constant, avail, (unsigned long long)vs->ucode_data_hash());
        bytes = avail;
      }
      uint32_t off = f.data.Alloc(bytes ? bytes : 16, 256);
      if (off == UINT32_MAX) { RingFull(f, "vertices", bytes, vs->ucode_data_hash()); skipped_++; return true; }
      const uint8_t* src = memory_->TranslatePhysical(fetch.address << 2) + skip;
      texture_conversion::CopySwapBlock(fetch.endian, f.data.cpu + off, src, bytes);
      copied_bytes += bytes;
      if (vs->ssbo) {
        // The shader computes its record addresses (floor(index) * stride + offset) and bounds-checks them against
        // this size; every index it derives from the vertex id is at most the vertex id, so the per-vertex extent
        // bounds the stream. Binding = the SDK's vertex binding index, the order xenos2glsl numbers the buffers.
        dkCmdBufBindStorageBuffer(cmd, DkStage_Vertex, ssbo_count++, f.data.gpu + off, bytes);
        continue;
      }
      dkCmdBufBindVtxBuffer(cmd, buffer_count, f.data.gpu + off, bytes);
      buffers[buffer_count] = DkVtxBufferState{stride, 0};
      for (const auto& a : vb.attributes) {
        if (attrib_count >= DK_MAX_VERTEX_ATTRIBS) break;
        DkVtxAttribSize size;
        DkVtxAttribType type;
        if (!VertexAttribOf(a.fetch_instr.attributes, size, type)) {
          LogOnce("vfmt", "unsupported vertex format %u", uint32_t(a.fetch_instr.attributes.data_format));
          size = DkVtxAttribSize_4x32;
          type = DkVtxAttribType_Float;
        }
        DkVtxAttribState st{};
        st.bufferId = buffer_count;
        st.isFixed = 0;
        st.offset = uint32_t(a.fetch_instr.attributes.offset) * 4;
        st.size = size;
        st.type = type;
        st.isBgra = 0;
        attribs[attrib_count++] = st;
      }
      buffer_count++;
    }
    if (copied_bytes > (1u << 20) && big_copy_logs_ < 12) {
      big_copy_logs_++;
      Log("vertex copy %u KB: VS %016llX n=%u indices %u-%u offset %d%s", copied_bytes >> 10, (unsigned long long)vs->ucode_data_hash(),
          index_count, min_index, vertex_count - 1, indx_offset, vs->ssbo ? " (storage buffers)" : "");
    }
    dkCmdBufBindVtxAttribState(cmd, attribs, attrib_count);
    dkCmdBufBindVtxBufferState(cmd, buffers, buffer_count);
    phase(4);

    // Uniforms: the ALU constants and the system constants, per stage.
    if (!BindUniforms(f, vs, ps, vi, regs)) { skipped_++; return true; }
    phase(5);

    dkCmdBufSetPrimitiveRestart(cmd, restart, restart_index);
    if (index_buffer_info) {
      dkCmdBufBindIdxBuffer(cmd, index_buffer_info->format == xenos::IndexFormat::kInt32 ? DkIdxFormat_Uint32 : DkIdxFormat_Uint16,
                            f.data.gpu + index_offset);
      dkCmdBufDrawIndexed(cmd, prim, index_count, 1, 0, vs->ssbo ? indx_offset : -int32_t(min_index), 0);
    } else {
      dkCmdBufDraw(cmd, prim, index_count, 1, vs->ssbo ? uint32_t(std::max(indx_offset, 0)) : 0, 0);
    }
    Checkpoint(f, "draw VS %016llX PS %016llX n=%u target=%ux%u first_texture=%08X %ux%u",
               (unsigned long long)vs->ucode_data_hash(), (unsigned long long)ps->ucode_data_hash(), index_count,
               color ? color->width : 0, color ? color->height : 0,
               tcount ? tnames[0].base : 0, tcount ? tnames[0].w : 0, tcount ? tnames[0].h : 0);
    if (trace_enabled_ && !traced_) {
      // One line per draw of a gameplay frame (file sdmc:/switch/tmnt/dk_trace): what the HP/MP bar and the
      // scaled sprites are drawn with.
      std::string verts;
      if (!vs->vertex_bindings().empty()) {
        const auto& vb0 = vs->vertex_bindings().front();
        auto vf = regs.GetVertexFetch(vb0.fetch_constant);
        const uint32_t* w = reinterpret_cast<const uint32_t*>(memory_->TranslatePhysical(vf.address << 2));
        auto half = [](uint16_t h) -> float {
          const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
          float v = e == 0 ? std::ldexp(float(m), -24) : e == 31 ? (m ? NAN : INFINITY) : std::ldexp(float(m | 1024), int(e) - 25);
          return s ? -v : v;
        };
        auto word = [&](uint32_t i) { return vf.endian == xenos::Endian::k8in32 ? __builtin_bswap32(w[i]) : w[i]; };
        if (vs->ucode_data_hash() == 0x9DC6ADB3E4AA3763ull && vb0.stride_words >= 5 && vf.size >= vb0.stride_words * 3) {
          // The sprite vertex shader: 4 halves position, 2 halves uv, RGBA8 colour. Screen position of corners 0 and 2.
          float c[16];
          for (int k = 0; k < 16; k++) c[k] = regs.Get<float>(XE_GPU_REG_SHADER_CONSTANT_000_X + 18 * 4 + k);
          for (uint32_t corner = 0; corner < 4 && corner * vb0.stride_words + 5 <= vf.size; corner += 2) {
            const uint32_t b = corner * vb0.stride_words;
            const float px = half(uint16_t(word(b))), py = half(uint16_t(word(b) >> 16)), pz = half(uint16_t(word(b + 1))),
                        pw = half(uint16_t(word(b + 1) >> 16));
            const float r3x = py, r3y = px, r3z = pw, r3w = pz;
            const float* c18 = c; const float* c19 = c + 4; const float* c20 = c + 8; const float* c21 = c + 12;
            float r2[4] = {r3w * c21[0], r3w * c21[3], r3w * c21[2], r3w * c21[1]};
            float t[4] = {r3z * c20[0] + r2[0], r3z * c20[3] + r2[1], r3z * c20[2] + r2[2], r3z * c20[1] + r2[3]};
            float u[4] = {t[0], t[2], t[3], t[1]};
            float r2n[4] = {r3y * c19[0] + u[0], r3y * c19[2] + u[1], r3y * c19[1] + u[2], r3y * c19[3] + u[3]};
            float o[4] = {r3x * c18[0] + r2n[0], r3x * c18[1] + r2n[2], r3x * c18[2] + r2n[1], r3x * c18[3] + r2n[3]};
            const float xn = o[0] * vi.ndc_scale[0] + vi.ndc_offset[0] * o[3], yn = o[1] * vi.ndc_scale[1] + vi.ndc_offset[1] * o[3];
            char s[160];
            std::snprintf(s, sizeof(s), " corner%u (%.0f,%.0f) uv(%g,%g) col %08X", corner,
                          (xn / o[3] * 0.5f + 0.5f) * vp.width + vp.x, (yn / o[3] * 0.5f + 0.5f) * vp.height + vp.y,
                          half(uint16_t(word(b + 4))), half(uint16_t(word(b + 4) >> 16)), word(b + 3));
            verts += s;
          }
        } else {
          for (uint32_t i = 0; i < std::min<uint32_t>(vb0.stride_words * 3, 15) && i < vf.size; i++) {
            char s[24];
            std::snprintf(s, sizeof(s), " %08X", word(i));
            verts += s;
          }
        }
      }
      char line[1400];
      std::snprintf(line, sizeof(line),
                    "#%u prim %u n=%u VS %08X PS %08X target %s%ux%u vp %g,%g %gx%g sc %d,%d %ux%u blend %u/%u/%u cull f%u b%u z %u/%u rects %d off %d idx %u-%u copy %u |%s | v:%s",
                    frame_draws_, uint32_t(prim_type), index_count, uint32_t(vs->ucode_data_hash()), uint32_t(ps->ucode_data_hash()),
                    color ? "" : "none ", color ? color->width : 0, color ? color->height : 0, vp.x, vp.y, vp.width, vp.height,
                    scissor.x, scissor.y, scissor.width, scissor.height,
                    uint32_t(regs.Get<reg::RB_BLENDCONTROL>().color_srcblend), uint32_t(regs.Get<reg::RB_BLENDCONTROL>().color_destblend),
                    uint32_t(regs.Get<reg::RB_BLENDCONTROL>().color_comb_fcn), su.cull_front, su.cull_back, dc.z_enable, dc.z_write_enable,
                    rects ? 1 : 0, indx_offset, min_index, vertex_count - 1, copied_bytes, bound_textures.c_str(), verts.c_str());
      trace_lines_.emplace_back(line);
    }
    if (ps->ucode_data_hash() == 0xB3EC630B32280C8Cull) frame_sprite_draws_++;
    f.draw_log.emplace_back(vs->ucode_data_hash(), ps->ucode_data_hash());
    draws_++;
    frame_draws_++;
    phase(6);
    return true;
  }

  bool IssueCopy() override {
    if (Skipping()) return true;
    SyncDirtyPages();
    return Resolve();
  }

  void IssueSwap(uint32_t frontbuffer_ptr, uint32_t frontbuffer_width, uint32_t frontbuffer_height) override {
    if (skip_odd_) {
      // "30": no present waits for a vblank in a skipped frame, so every swap is held to 60 a second here
      // (console 0.53: the menus ran at 120 swaps a second; 0.56's swap interval 2 lost frames to the vblank pairs).
      const auto now = std::chrono::steady_clock::now();
      if (now < next_swap_) std::this_thread::sleep_until(next_swap_);
      next_swap_ = std::max(now, next_swap_) + std::chrono::microseconds(16667);
    }
    if (Skipping()) {
      swaps_++;  // nothing was recorded: the frame is just counted
      if (swaps_ % 60 == 0) {
        std::printf("[nx] gpu: swap #%u%s\n", swaps_, " (skipped)");  // the Eden gate's counter
        ReadFpsMode();
      }
      NextFrame(false);
      return;
    }
    Frame& f = frames_[frame_];
    DkCmdBuf cmd = f.cmdbuf;
    HostImage* front = nullptr;
    if (guest_images_.count(frontbuffer_ptr) && frontbuffer_width && frontbuffer_height) {
      // The resolve created it with the aligned height: the first swap trims it to the front buffer's size.
      front = GuestImage(f, frontbuffer_ptr, frontbuffer_width, frontbuffer_height, DkImageFormat_RGBA8_Unorm, false);
    }
    if (present_target_) {
      // Debug (the file sdmc:/switch/tmnt/dk_present_rt): show the widest colour render target itself,
      // bypassing the resolve copies and the front buffer image.
      HostImage* widest = nullptr;
      for (auto& it : targets_) if (!it.second->is_depth && (!widest || it.second->width > widest->width)) widest = it.second;
      if (widest) front = widest;
    }
    // Debug readback at two swaps: what the images hold (uniform colour = a broken stage).
    std::vector<std::pair<std::string, uint32_t>> dumps;
    const bool trace_frame = trace_enabled_ && !traced_ &&
        (trace_on_mip_ ? frame_mip_only_ : trace_min_swap_ == 0 ? (ring_full_swap_ && swaps_ + 1 >= ring_full_swap_ + trace_after_full_ && frame_big_tex_ && frame_draws_ > 250)
                              : (NowMs() - trace_start_ms_ >= trace_min_swap_ * 1000.0 && (trace_any_ || frame_draws_ > 440)));
    if (trace_frame) {
      traced_ = true;
      if (trace_second_) { traced_after_ = true; trace_min_swap_ = trace_second_; trace_second_ = 0; traced_ = false; }
      Log("trace of swap #%u: %zu draws", swaps_ + 1, trace_lines_.size());
      for (const std::string& l : trace_lines_) Log("  %s", l.c_str());
    }
    if (!trace_frame) trace_lines_.clear();
    if (trace_frame || (dump_images_ && (swaps_ + 1 == 120 || swaps_ + 1 == 600 || swaps_ + 1 == 2400))) {
      auto queue_dump = [&](const char* what, HostImage* img) {
        if (!img) return;
        const bool is16 = img->format == DkImageFormat_BGR5A1_Unorm || img->format == DkImageFormat_R16_Unorm || img->format == DkImageFormat_BGR565_Unorm;
        uint32_t bytes = img->width * img->height * (is16 ? 2 : 4);
        uint32_t off = f.staging.Alloc(bytes, 0x1000);
        if (off == UINT32_MAX) return;
        DkImageView v;
        dkImageViewDefaults(&v, &img->image);
        DkImageRect r{0, 0, 0, img->width, img->height, 1};
        DkCopyBuf cb{f.staging.gpu + off, 0, 0};
        dkCmdBufBarrier(cmd, DkBarrier_Full, DkInvalidateFlags_Image);
        dkCmdBufCopyImageToBuffer(cmd, &v, &r, &cb, 0);
        char name[96];
        const char* tag = img->format == DkImageFormat_BGR5A1_Unorm ? " [1555]" : img->format == DkImageFormat_R16_Unorm ? " [r16]" : img->format == DkImageFormat_BGR565_Unorm ? " [565]" : img->format == DkImageFormat_R32_Float ? " [r32f]" : "";
        std::snprintf(name, sizeof(name), "%s%s %ux%u", what, tag, img->width, img->height);
        dumps.emplace_back(name, off);
      };
      queue_dump("front", front);
      for (auto& it : targets_) if (!it.second->is_depth) { char n[48]; std::snprintf(n, sizeof(n), "target base %u pitch %u", it.second->edram_base, it.second->edram_pitch); queue_dump(n, it.second); }
      for (int pass = 0; pass < 2; pass++)  // resolve images first: the staging buffer does not hold every image
        for (auto& it : guest_images_) if ((trace_frame || it.second->width >= 600) && it.second != front && it.second->from_resolve == (pass == 0)) { char n[48]; std::snprintf(n, sizeof(n), "guest %08X%s", it.first, it.second->from_resolve ? " (resolve)" : " (upload)"); queue_dump(n, it.second); }
      dkCmdBufBarrier(cmd, DkBarrier_Full, 0);
    }
    // Inspect completed submissions before acquire can abort on a faulted queue, without waiting.
    if (gpu_checkpoints_) {
      for (Frame& submitted : frames_) {
        if (!submitted.fenced) continue;
        DkResult result = dkFenceWait(&submitted.fence, 0);
        const uint64_t* ts = submitted.ts_off != UINT32_MAX ? reinterpret_cast<const uint64_t*>(submitted.data.cpu + submitted.ts_off) : nullptr;
        if (result == DkResult_Success && ts && !ts[2] && !ts[3]) ReportCheckpoints(submitted);
      }
    }
    if (swaps_ < 5) Log("swap #%u: acquiring a swapchain image", swaps_ + 1);
    const double acq0 = NowMs();
    int slot = dkQueueAcquireImage(queue_, swapchain_);
    t_acquire_ms_ += NowMs() - acq0;
    rex_nx_waited_add(uint64_t((NowMs() - acq0) * 19200.0));
    if (swaps_ < 5) Log("swap #%u: slot %d", swaps_ + 1, slot);
    HostImage* target = swap_images_[slot];
    dkCmdBufBarrier(cmd, DkBarrier_Full, DkInvalidateFlags_Image);
    DkImageView tv;
    dkImageViewDefaults(&tv, &target->image);
    DkImageView const* targets[1] = {&tv};
    dkCmdBufBindRenderTargets(cmd, targets, 1, nullptr);
    DkViewport vp{0, 0, float(kPresentWidth), float(kPresentHeight), 0.0f, 1.0f};
    dkCmdBufSetViewports(cmd, 0, &vp, 1);
    DkScissor sc{0, 0, kPresentWidth, kPresentHeight};
    dkCmdBufSetScissors(cmd, 0, &sc, 1);
    // Black (it was a debug navy: the user saw a blue flash at start, before the first front buffer exists).
    const float black[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    dkCmdBufClearColor(cmd, 0, DkColorMask_RGBA, black);
    Checkpoint(f, "present clear %ux%u", kPresentWidth, kPresentHeight);
    if (front && present_vs_ && present_fs_) {
      DkRasterizerState rs;
      dkRasterizerStateDefaults(&rs);
      rs.cullMode = DkFace_None;
      dkCmdBufBindRasterizerState(cmd, &rs);
      DkColorState cs;
      dkColorStateDefaults(&cs);
      dkCmdBufBindColorState(cmd, &cs);
      DkColorWriteState cw;
      dkColorWriteStateDefaults(&cw);
      dkCmdBufBindColorWriteState(cmd, &cw);
      DkDepthStencilState ds;
      dkDepthStencilStateDefaults(&ds);
      ds.depthTestEnable = 0;
      ds.depthWriteEnable = 0;
      dkCmdBufBindDepthStencilState(cmd, &ds);
      DkShader const* shaders[2] = {&present_vs_shader_, &present_fs_shader_};
      dkCmdBufBindShaders(cmd, DkStageFlag_GraphicsMask, shaders, 2);
      dkCmdBufBindVtxAttribState(cmd, nullptr, 0);
      dkCmdBufBindVtxBufferState(cmd, nullptr, 0);
      // The overlay (present.frag "Hud"): the fps counter or the settings menu as a text grid from
      // nx_menu.cpp (size, grid, style, 16 row colours, two floats per cell). A full ring skips the
      // bind: one frame with a stale overlay is harmless.
      static NxOverlay ov;  // 6 KB: not on this thread's stack
      const int draw = rex_nx_menu_overlay(&ov, rex_nx_setting_fps_counter() ? hud_text_ : "", int(kPresentWidth), int(kPresentHeight));
      constexpr uint32_t kHudBytes = 3 * 16 + 16 * 16 + sizeof(ov.cells);
      uint32_t hud_off = f.data.Alloc(kHudBytes, DK_UNIFORM_BUF_ALIGNMENT);
      if (hud_off != UINT32_MAX) {
        float* u = reinterpret_cast<float*>(f.data.cpu + hud_off);
        u[0] = float(kPresentWidth); u[1] = float(kPresentHeight); u[2] = draw ? float(ov.cols) : 0.0f; u[3] = float(ov.rows);
        u[4] = float(ov.x); u[5] = float(ov.y); u[6] = float(6 * ov.scale); u[7] = float(8 * ov.scale);
        u[8] = float(ov.scale ? ov.scale : 1); u[9] = 0.85f; u[10] = ov.box ? 1.0f : 0.0f; u[11] = 1.0f;  // area average
        std::memcpy(u + 12, ov.row_rgb, sizeof(ov.row_rgb));
        std::memcpy(u + 12 + 64, ov.cells, sizeof(ov.cells));
        DkBufExtents hud_buf{f.data.gpu + hud_off, kHudBytes};
        dkCmdBufBindUniformBuffers(cmd, DkStage_Fragment, 0, &hud_buf, 1);
      }
      uint32_t dslot = DescriptorFor(f, front, 0);
      if (dslot != UINT32_MAX) {
        // The front buffer is 1920x1080 and the swapchain 1280x720 (x1.5). The sampler was index 4 = POINT (bit 4 is
        // the white border, not linear): every third row and column was dropped, so thin text strokes vanished and
        // fine detail shimmered on motion (0.141-0.265). Now linear with clamp-to-edge; present.frag averages four
        // linear taps over the output pixel's footprint (style.w). The DBG PRESENT toggle (AREA / LINEAR / POINT)
        // was removed for the release (0.280); git history before it has the A/B.
        const uint32_t clamp = (uint32_t(xenos::ClampMode::kClampToEdge) << 3) | (uint32_t(xenos::ClampMode::kClampToEdge) << 6);
        DkResHandle handle = dkMakeTextureHandle(dslot, clamp | 3u);
        dkCmdBufBindTextures(cmd, DkStage_Fragment, 0, &handle, 1);
        dkCmdBufDraw(cmd, DkPrimitive_Triangles, 3, 1, 0, 0);
        Checkpoint(f, "present draw front=%08X %ux%u", front->guest_base, front->width, front->height);
      }
    }
    if (!gpu_checkpoints_ && f.ts_off != UINT32_MAX)
      dkCmdBufReportCounter(cmd, DkCounter_Timestamp, f.data.gpu + f.ts_off + 16);
#ifndef TMNT_DK_GL
    RecordFrame(f, cmd, front);
#endif
    dkCmdBufBarrier(cmd, DkBarrier_Full, 0);
    if (gpu_checkpoints_ && f.ts_off != UINT32_MAX)
      dkCmdBufReportCounter(cmd, DkCounter_Timestamp, f.data.gpu + f.ts_off + 16);
    DkCmdList list = dkCmdBufFinishList(cmd);
    // Console 0.67 crash report: DkResult_Fail raised by the queue flush inside dkQueuePresentImage (the channel was
    // already in error from this frame's earlier work) aborted the process before any checkpoint report. Dump first.
    if (dkQueueIsInErrorState(queue_)) DumpFencedFrames("queue in error state before submit");
    dkQueueSubmitCommands(queue_, list);
    dkQueueSignalFence(queue_, &f.fence, false);
    f.checkpoint_swap = swaps_ + 1;
    f.fenced = true;
    if (dkQueueIsInErrorState(queue_)) DumpFencedFrames("queue in error state before present");
    dkQueuePresentImage(queue_, swapchain_, slot);
    // Shown-frame intervals: what the screen gets (the [nx] fps line counts guest swaps).
    const double shown_now = NowMs();
    if (shown_last_ms_) shown_ms_.push_back(shown_now - shown_last_ms_);
    shown_last_ms_ = shown_now;
#ifdef TMNT_DK_GL
    FlushTextureDumps();
#endif
    if (!dumps.empty()) {
      dkQueueWaitIdle(queue_);
      for (auto& d : dumps) {
        // The image's dims are in the name; recover them from the staging layout: sample a 16x16 grid.
        uint32_t w = 0, h = 0;
        std::sscanf(d.first.c_str() + d.first.rfind(' ') + 1, "%ux%u", &w, &h);
        const uint32_t* px = reinterpret_cast<const uint32_t*>(f.staging.cpu + d.second);
        std::unordered_set<uint32_t> distinct;
        for (uint32_t y = 0; y < 16 && h; y++)
          for (uint32_t x = 0; x < 16 && w; x++) distinct.insert(px[(y * h / 16) * w + x * w / 16]);
        Log("dump at swap #%u: %s: %zu distinct colours in a 16x16 grid; pixels %08X %08X %08X %08X %08X", swaps_ + 1,
            d.first.c_str(), distinct.size(), w && h ? px[0] : 0, w && h ? px[(h / 4) * w + w / 4] : 0,
            w && h ? px[(h / 2) * w + w / 2] : 0, w && h ? px[(3 * h / 4) * w + 3 * w / 4] : 0, w && h ? px[(h - 1) * w + w - 1] : 0);
        if (trace_frame && w && h) {
          // The traced frame's images as PPM files next to the shader directory: dump/<n>.ppm (names in the log).
          static uint32_t dump_index = 0;
          const std::string dir = debug_dir_ + "dump";
          mkdir(dir.c_str(), 0777);
          char path[160];
          std::snprintf(path, sizeof(path), "%s/%02u.ppm", dir.c_str(), dump_index);
          Log("  image file %s = %s", path, d.first.c_str());
          dump_index++;
          if (FILE* pf = std::fopen(path, "wb")) {
            std::fprintf(pf, "P6\n%u %u\n255\n", w, h);
            std::vector<uint8_t> row(w * 3);
            for (uint32_t y = 0; y < h; y++) {
              for (uint32_t x = 0; x < w; x++) {
                uint32_t r, g, b;
                if (d.first.find("[1555]") != std::string::npos) {  // A1 R5 G5 B5 (Xenos 1_5_5_5)
                  const uint16_t p16 = reinterpret_cast<const uint16_t*>(px)[y * w + x];
                  r = ((p16 >> 10) & 31) * 255 / 31; g = ((p16 >> 5) & 31) * 255 / 31; b = (p16 & 31) * 255 / 31;
                } else if (d.first.find("[565]") != std::string::npos) {
                  const uint16_t p16 = reinterpret_cast<const uint16_t*>(px)[y * w + x];
                  r = ((p16 >> 11) & 31) * 255 / 31; g = ((p16 >> 5) & 63) * 255 / 63; b = (p16 & 31) * 255 / 31;
                } else if (d.first.find("[r32f]") != std::string::npos) {  // depth: red = the value, green and blue = its finer digits
                  const float v = reinterpret_cast<const float*>(px)[y * w + x];
                  const float c = v < 0.0f ? 0.0f : v > 1.0f ? 1.0f : v;
                  r = uint32_t(c * 255.0f); g = uint32_t(c * 65280.0f) & 255; b = uint32_t(c * 16711680.0f) & 255;
                } else if (d.first.find("[r16]") != std::string::npos) {  // index texture: the high byte as grey, the low byte in blue
                  const uint16_t p16 = reinterpret_cast<const uint16_t*>(px)[y * w + x];
                  r = g = p16 >> 8; b = p16 & 255;
                } else {
                  const uint32_t p = px[y * w + x];  // RGBA8 bytes R,G,B,A in memory order
                  r = p & 255; g = (p >> 8) & 255; b = (p >> 16) & 255;
                }
                row[x * 3 + 0] = uint8_t(r);
                row[x * 3 + 1] = uint8_t(g);
                row[x * 3 + 2] = uint8_t(b);
              }
              std::fwrite(row.data(), 1, row.size(), pf);
            }
            std::fclose(pf);
          }
        }
      }
    }
    swaps_++;
    rex_nx_swaps = swaps_;
    if (swaps_ % 60 == 0) ReadFpsMode();
    // The fps counter: swaps per second over a one-second window, drawn by the next present passes.
    const auto now = std::chrono::steady_clock::now();
    if (hud_window_start_ == std::chrono::steady_clock::time_point{}) hud_window_start_ = now;
    hud_window_swaps_++;
    const double elapsed = std::chrono::duration<double>(now - hud_window_start_).count();
    if (elapsed >= 1.0) {
      // fps, then the main thread's busy ms per frame (switch_nx.cpp's busy line): "36 9.8".
      std::snprintf(hud_text_, sizeof(hud_text_), "%.0f %.1f", hud_window_swaps_ / elapsed, rex_nx_main_busy_ms);
      std::sort(shown_ms_.begin(), shown_ms_.end());
      const size_t sn = shown_ms_.size();
      const size_t le34 = std::upper_bound(shown_ms_.begin(), shown_ms_.end(), 34.0) - shown_ms_.begin();
      std::printf("[nx] dk time/s: draws+resolves %.0f ms, acquire %.0f ms, fence wait %.0f ms, gpu %.0f ms/%u frames, over %u frames (%.0f ms window); "
                  "draw phases ms: sync %.0f setup %.0f tex %.0f state %.0f vtx %.0f uni %.0f draw %.0f; uniform uploads vs %u ps %u sys %u; texture bind hits %u; late writes %u; "
                  "shown %zu: interval p50 %.1f p95 %.1f ms, %zu%% <= 34 ms\n",
                  t_draw_ms_, t_acquire_ms_, t_fence_ms_, t_gpu_ms_, gpu_frames_, hud_window_swaps_, elapsed * 1000.0, t_phase_ms_[0], t_phase_ms_[1],
                  t_phase_ms_[2], t_phase_ms_[3], t_phase_ms_[4], t_phase_ms_[5], t_phase_ms_[6], uni_uploads_[0], uni_uploads_[1],
                  uni_uploads_[2], texbind_hits_, late_writes_, sn, sn ? shown_ms_[sn / 2] : 0.0, sn ? shown_ms_[sn * 95 / 100] : 0.0, sn ? le34 * 100 / sn : 0);
      shown_ms_.clear();
      texbind_hits_ = late_writes_ = 0;
      t_draw_ms_ = t_acquire_ms_ = t_fence_ms_ = t_gpu_ms_ = 0;
      gpu_frames_ = 0;
      for (double& t : t_phase_ms_) t = 0;
      for (uint32_t& u : uni_uploads_) u = 0;
      hud_window_start_ = now;
      hud_window_swaps_ = 0;
    }
    // Textures uploaded in this frame whose guest memory was written again before the swap: the game rewrote a sprite
    // slot after our snapshot while the draw that samples it is still queued (the blink hypothesis REUPLOAD = ALL
    // cannot exclude). The marks of those late writes are consumed here; the next frame re-uploads them anyway.
    SyncDirtyPages();
    for (const Uploaded& u : frame_uploaded_) if (WrittenSince(u.base, u.bytes, u.serial)) late_writes_++;
    frame_uploaded_.clear();
    if (swaps_ <= 3 || swaps_ % 60 == 0) {
      std::printf("[nx] gpu: swap #%u\n", swaps_);  // the Eden gate's frame counter
      Log("swap #%u: front %08X %ux%u %s; frame: %u draws, %u clears, %u resolves, %u skipped, %u no-shader, %u re-uploads, %llu dirty pages; data %u KB staging %u KB; images %u (%u MB)",
          swaps_, frontbuffer_ptr, frontbuffer_width, frontbuffer_height, front ? "shown" : "MISSING", frame_draws_,
          frame_clears_, frame_resolves_, frame_skipped_, frame_missing_, frame_reuploads_, (unsigned long long)(dirty_pages_ - dirty_pages_at_swap_),
          f.data.used >> 10, f.staging.used >> 10, image_count_, images_.used >> 20);
    }
    if (frame_ring_full_ && ring_full_logs_ < 40) {
      ring_full_logs_++;
      Log("frame data ring full: swap #%u dropped %u draws, drew %u (first: %s); data %u KB", swaps_, frame_ring_full_,
          frame_draws_, ring_first_, f.data.used >> 10);
    }
    frame_ring_full_ = 0;
    dirty_pages_at_swap_ = dirty_pages_;
    frame_draws_ = frame_clears_ = frame_resolves_ = frame_skipped_ = frame_missing_ = frame_reuploads_ = 0;
    frame_sprite_draws_ = 0;
    frame_big_tex_ = false;
    frame_mip_only_ = false;
    frame_ = (frame_ + 1) % kFrames;
    begun_ = false;
    NextFrame(true);
  }

 private:
  // Whether the next frame is drawn (hcu-switch's 30skip). 60: every frame. 30: not the frame after a drawn one, and not
  // one the GPU has no free frame slot for yet (its fence is still pending) instead of waiting: the guest keeps its 60
  // frames a second (the command processor still runs every packet: fences, write-backs, the guest's waits), the game
  // keeps its speed, every second frame is shown.
  // The draws of a frame whose fence did not signal, by shader pair in order of first use (the hang's suspects).
  void LogHungFrame(const Frame& f) {
    std::vector<std::pair<std::pair<uint64_t, uint64_t>, uint32_t>> pairs;
    for (const auto& d : f.draw_log) {
      auto it = std::find_if(pairs.begin(), pairs.end(), [&](const auto& p) { return p.first == d; });
      if (it == pairs.end()) pairs.push_back({d, 1}); else it->second++;
    }
    Log("hung frame: %zu draws, %zu shader pairs (in order of first use):", f.draw_log.size(), pairs.size());
    for (const auto& p : pairs)
      Log("  VS %016llX PS %016llX x%u", (unsigned long long)p.first.first, (unsigned long long)p.first.second, p.second);
    std::fflush(stdout);
  }
  static uint64_t PairKey(const std::pair<uint64_t, uint64_t>& d) { return d.first * 0x9E3779B97F4A7C15ull ^ d.second; }
  // The pairs of a hung or faulted frame that never ran in a finished frame: appended to dk_skip (next to the shaders
  // folder), and the next start skips their draws. ponytail: a fault outside draws (a copy, a resolve) bans innocent
  // pairs; delete dk_skip to undo.
  void NoteFaultedFrame(const Frame& f) {
    LogHungFrame(f);
    if (gpu_checkpoints_) { ReportCheckpoints(f); return; }
    int n = 0;
    if (FILE* o = std::fopen((debug_dir_ + "dk_skip").c_str(), "a")) {
      for (const auto& d : f.draw_log) {
        if (ok_pairs_.count(PairKey(d)) || !skip_pairs_.insert(PairKey(d)).second) continue;
        std::fprintf(o, "%016llX %016llX\n", (unsigned long long)d.first, (unsigned long long)d.second);
        Log("  suspect VS %016llX PS %016llX: its draws are skipped from the next start", (unsigned long long)d.first,
            (unsigned long long)d.second);
        n++;
      }
      std::fclose(o);
    }
    Log("%d suspect pairs added to %sdk_skip", n, debug_dir_.c_str());
    std::fflush(stdout);
  }

  void NextFrame(bool drawn) {
    if (!(skip_odd_ && frames_[frame_].fenced) || dkFenceWait(&frames_[frame_].fence, 0) == DkResult_Success) busy_run_ = 0;
    skip_frame_ = false;
    if (skip_odd_ && drawn) {
      skip_frame_ = true;
    } else if (skip_odd_ && frames_[frame_].fenced && dkFenceWait(&frames_[frame_].fence, 0) != DkResult_Success) {
      skip_frame_ = true;
      if (++busy_run_ == 120) {
        LogHungFrame(frames_[frame_]);
        if (gpu_checkpoints_) {
          // Save the live snapshot before a driver error check can abort the process.
          ReportCheckpoints(frames_[frame_], true);
          const DkResult result = dkFenceWait(&frames_[frame_].fence, 10000000ll);
          Log("pending frame %u: diagnostic fence wait result %d, queue error %d", frame_, int(result),
              int(dkQueueIsInErrorState(queue_)));
        }
      }
      busy_skips_++;
    }
    if (!skip_frame_ && !begun_) {
      BeginFrame();
      begun_ = true;
    }
  }

  // The frame-rate mode (the settings menu, nx_menu.cpp; read at start and every 60 swaps, so it switches during a run).
  // "60" = every frame drawn. "30" = hcu-switch's 30skip (NextFrame): the guest keeps 60 frames a second, every second
  // one is drawn. TMNT counts frames for its game time: a 30 Hz present or 30 Hz vblanks (Castlevania's "30", console
  // 0.50) ran it at half speed. A skipped frame of "30" waits for no vblank,
  // so IssueSwap holds every swap to 60 a second (console 0.53: the menus ran at 120 swaps a second).
  void ReadFpsMode() {
    if (!debug_dir_.empty() && unlink((debug_dir_ + "record_now").c_str()) == 0) rex_nx_debug_record_set(1);
    const bool skip = rex_nx_setting_fps_mode() == 1;
    if (skip == skip_odd_ && swaps_) return;
    skip_odd_ = skip;
    std::printf("[nx] gpu: fps mode %s from swap #%u (%u frames skipped for a busy GPU so far)\n", skip ? "30 (every second frame drawn)" : "60",
                swaps_, busy_skips_);
    std::fflush(stdout);
  }

  // Linux test host: TMNT_DK_START_SWAP=N turns the backend into a null backend until swap N (the guest runs
  // at full speed to a late scene; textures and targets are created lazily when drawing starts).
  // The file <TMNT_DK_DUMP_DIR>/skip_on does the same while it exists (checked every 8 swaps): fast forward
  // through loading screens, then delete it.
  bool Skipping() const {
    if (skip_frame_) return true;  // the 30 mode (NextFrame)
#ifdef TMNT_DK_GL
    static const uint32_t start = std::getenv("TMNT_DK_START_SWAP") ? uint32_t(std::atoi(std::getenv("TMNT_DK_START_SWAP"))) : 0;
    static const char* dir = std::getenv("TMNT_DK_DUMP_DIR");
    static bool file_skip = false;
    if (dir && (swaps_ & 7) == 0 && !skip_checked_[0]) {
      file_skip = access((std::string(dir) + "/skip_on").c_str(), F_OK) == 0;
      skip_checked_[0] = true;
    } else if ((swaps_ & 7) != 0) {
      skip_checked_[0] = false;
    }
    return file_skip || swaps_ + 1 < start;
#else
    return false;
#endif
  }
  mutable bool skip_checked_[1] = {false};

  // ----- frames ---------------------------------------------------------------------------------
  static constexpr uint32_t kMaxCheckpoints = 4096;
  void Checkpoint(Frame& f, const char* fmt, ...) {
    if (!gpu_checkpoints_ || !f.checkpoint_data.block) return;
    if (f.checkpoints.size() == kMaxCheckpoints) { f.checkpoints_full = true; return; }
    char line[384];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(line, sizeof(line), fmt, ap);
    va_end(ap);
    uint32_t index = uint32_t(f.checkpoints.size());
    f.checkpoints.emplace_back(line);
    // A full barrier precedes the graphics report; fault timing can still make the boundary ambiguous.
    dkCmdBufBarrier(f.cmdbuf, DkBarrier_Full, 0);
    dkCmdBufReportCounter(f.cmdbuf, DkCounter_Timestamp, f.checkpoint_data.gpu + index * 16);
  }
  // Every frame in flight, one file each (dk_gpu_fault_<frame>.txt): called from the deko3d error callback and
  // when the queue is found in error state. No deko3d calls here: the library may be in its error path.
  void DumpFencedFrames(const char* why) {
    Log("GPU fault dump: %s", why);
    for (uint32_t i = 0; i < kFrames; i++) {
      if (!frames_[i].fenced) continue;
      char name[48];
      std::snprintf(name, sizeof(name), "dk_gpu_fault_%u.txt", i);
      ReportCheckpoints(frames_[i], true, name);
    }
    std::fflush(stdout);
  }
  void ReportCheckpoints(const Frame& f, bool pending = false, const char* name = nullptr) {
    if (!pending && gpu_reported_) return;
    if (!name) name = pending ? "dk_gpu_pending.txt" : "dk_gpu_fault.txt";
    FILE* out = std::fopen((debug_dir_ + name).c_str(), "w");
    if (!out) { Log("GPU checkpoint report could not be opened"); return; }
    if (pending) std::fprintf(out, "Live snapshot after 120 busy skips; the GPU may still be running. Reports may change while read.\n");
    std::fprintf(out, "frame %u, swap %u: %zu checkpoints%s\n", uint32_t(&f - frames_), f.checkpoint_swap, f.checkpoints.size(),
                 f.checkpoints_full ? " (limit reached)" : "");
    std::fprintf(out, "Reports follow full barriers; missing reports bracket incomplete work, not necessarily the exact fault.\n");
    size_t first_missing = 0;
    while (first_missing < f.checkpoints.size()) {
      const uint64_t* report = reinterpret_cast<const uint64_t*>(f.checkpoint_data.cpu + first_missing * 16);
      if (!report[0] && !report[1]) break;
      first_missing++;
    }
    if (first_missing) std::fprintf(out, "last consecutive completed: %zu %s\n", first_missing - 1, f.checkpoints[first_missing - 1].c_str());
    else std::fprintf(out, "last consecutive completed: none\n");
    if (first_missing < f.checkpoints.size()) std::fprintf(out, "first missing: %zu %s\n", first_missing, f.checkpoints[first_missing].c_str());
    else std::fprintf(out, "first missing: none in recorded checkpoints%s\n", f.checkpoints_full ? " (limit reached)" : "");
    for (size_t i = 0; i < f.checkpoints.size(); i++) {
      const uint64_t* report = reinterpret_cast<const uint64_t*>(f.checkpoint_data.cpu + i * 16);
      std::fprintf(out, "%zu %s %s\n", i, (report[0] || report[1]) ? "done" : "missing", f.checkpoints[i].c_str());
    }
    const bool write_failed = std::ferror(out) != 0;
    if (std::fclose(out) != 0 || write_failed) {
      Log("GPU checkpoint report could not be written completely");
      return;
    }
    if (!pending) gpu_reported_ = true;
    Log("GPU checkpoints written to %s%s", debug_dir_.c_str(), name);
  }

  void BeginFrame() {
    Frame& f = frames_[frame_];
    if (f.fenced) {
      if (swaps_ < 6) Log("frame %u: waiting for its fence", frame_);
      const double fw0 = NowMs();
      DkResult r = dkFenceWait(&f.fence, 2000000000ll);  // 2 s: a hang is logged, not silent
      t_fence_ms_ += NowMs() - fw0;
      rex_nx_waited_add(uint64_t((NowMs() - fw0) * 19200.0));
      // A fence done without the frame's end timestamp: the kernel signalled it after a GPU fault (console 0.57,
      // the first cutscene; the next acquire then aborts). A hung or faulted frame's new shader pairs go to dk_skip.
      const uint64_t* ts = f.ts_off != UINT32_MAX ? reinterpret_cast<const uint64_t*>(f.data.cpu + f.ts_off) : nullptr;
      if (r != DkResult_Success) {
        Log("frame %u: fence wait result %d (GPU hang?)", frame_, int(r));
        NoteFaultedFrame(f);
      } else if (ts && !ts[2] && !ts[3]) {
        Log("frame %u: fence done without its end timestamp (GPU fault)", frame_);
        NoteFaultedFrame(f);
      } else {
        for (size_t i = 0; i < f.draw_log.size(); i++)
          if (!i || f.draw_log[i] != f.draw_log[i - 1]) ok_pairs_.insert(PairKey(f.draw_log[i]));
      }
      if (swaps_ < 6) Log("frame %u: fence done", frame_);
      f.fenced = false;
    }
    f.draw_log.clear();
    f.checkpoints.clear();
    f.checkpoints_full = false;
#ifndef TMNT_DK_GL
    if (f.rec_pending) {
      const uint32_t bytes = kRecW * kRecH * 4;
      dkMemBlockFlushCpuCache(rec_.block, frame_ * bytes, bytes);
      rex_rec_push(rec_.cpu + frame_ * bytes, f.rec_swap, f.rec_us);
      f.rec_pending = false;
    }
#endif
    if (f.ts_off != UINT32_MAX) {
      const uint64_t* r = reinterpret_cast<const uint64_t*>(f.data.cpu + f.ts_off);
      const uint64_t t0 = r[1] ? r[1] : r[0], t1 = r[3] ? r[3] : r[2];
      if (gpu_frames_ < 3) Log("gpu timestamps: top %llu/%llu bottom %llu/%llu", (unsigned long long)r[0], (unsigned long long)r[1],
                               (unsigned long long)r[2], (unsigned long long)r[3]);
      if (t1 > t0) { t_gpu_ms_ += double(dkTimestampToNs(t1 - t0)) / 1e6; gpu_frames_++; }
      f.ts_off = UINT32_MAX;
    }
    RestampRecentPages();
    f.data.Reset();
    f.staging.Reset();
    f.next_descriptor = 0;
    f.descriptors.clear();
    f.texbinds.clear();
    dkCmdBufClear(f.cmdbuf);
    dkCmdBufAddMemory(f.cmdbuf, cmdmem_.block, f.cmd_offset, kFrameCmdBytes);
    f.ts_off = f.data.Alloc(32, 16);
    if (f.ts_off != UINT32_MAX) {
      std::memset(f.data.cpu + f.ts_off, 0, 32);
      dkCmdBufReportCounter(f.cmdbuf, DkCounter_TimestampPipelineTop, f.data.gpu + f.ts_off);
    }
    if (gpu_checkpoints_) std::memset(f.checkpoint_data.cpu, 0, kMaxCheckpoints * 16);
    dkCmdBufBindImageDescriptorSet(f.cmdbuf, descriptors_.gpu + f.desc_offset, kMaxImageDescriptors);
    dkCmdBufBindSamplerDescriptorSet(f.cmdbuf, descriptors_.gpu + sampler_desc_offset_, kSamplerCount);
    std::fill(std::begin(bound_colors_), std::end(bound_colors_), nullptr);
    bound_depth_ = nullptr;
    f.vs_uni = f.ps_uni = f.sys_uni = UINT32_MAX;
    f.vs_uni_shader = f.ps_uni_shader = nullptr;
  }

  // Every register write of the command stream (SET_CONSTANT, LOAD_ALU_CONSTANT, REG_RMW ...) comes through here:
  // the float constant ranges mark their uniform block stale.
  void WriteRegister(uint32_t index, uint32_t value) override {
    CommandProcessor::WriteRegister(index, value);
    if (index < XE_GPU_REG_SHADER_CONSTANT_000_X) return;
    if (index < XE_GPU_REG_SHADER_CONSTANT_256_X) vs_consts_dirty_ = true;
    else if (index <= XE_GPU_REG_SHADER_CONSTANT_511_W) ps_consts_dirty_ = true;
  }
  // SET_CONSTANT / LOAD_ALU_CONSTANT write the shader constants (float 0x4000-0x47FF, fetch 0x4800-0x48FF, bool and
  // loop 0x4900-0x4927) a dword at a time through the virtual WriteRegister (a bounds check, a register-info lookup,
  // the scratch/COHER/LUT switch each): the game writes constants before nearly every draw, ~80,000 dwords a frame
  // (Eden 0.218: the command thread's non-draw work was ~6 ms per frame). None of those registers is special, so a
  // range inside them is copied straight into the register file (byte-swapped) with one dirty mark.
  void WriteRegistersFromMem(uint32_t start_index, uint32_t* base, uint32_t num_registers) override {
    const uint32_t end = start_index + num_registers;
    if (start_index >= XE_GPU_REG_SHADER_CONSTANT_000_X && end <= XE_GPU_REG_SHADER_CONSTANT_LOOP_31 + 1 && num_registers) {
      uint32_t* dst = &register_file_->values[start_index];
      for (uint32_t i = 0; i < num_registers; ++i) dst[i] = rex::memory::load_and_swap<uint32_t>(base + i);
      if (start_index < XE_GPU_REG_SHADER_CONSTANT_256_X) vs_consts_dirty_ = true;
      if (end > XE_GPU_REG_SHADER_CONSTANT_256_X && start_index <= XE_GPU_REG_SHADER_CONSTANT_511_W) ps_consts_dirty_ = true;
      return;
    }
    CommandProcessor::WriteRegistersFromMem(start_index, base, num_registers);
  }

  // ----- dirty pages ----------------------------------------------------------------------------
  // Consumes the dirty-page bytes before every draw and stamps each written page with the draw
  // serial; a texture whose pages were written after the upload or resolve that filled its image
  // is uploaded again at its next bind (TextureImage). The coarse map makes a clean frame cost 512
  // byte reads. ponytail: the guest marks a page after the store without a barrier, so a texture
  // written in the same microsecond as its bind can be uploaded stale once (the next write fixes it).
  void SyncDirtyPages() {
    serial_++;
    // The guest threads mark a page after its store; on the Tegra (ARM) the dirty byte and the data can be seen out of
    // order without a fence: the data read after this point is at least as new as the marks read here.
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
    const uint64_t* words = reinterpret_cast<const uint64_t*>(const_cast<const uint8_t*>(rex_nx_dirty));
    for (uint32_t c = 0; c < 0x200; c++) {
      // A region seen dirty in this or the previous frame is scanned again even when its coarse byte is 0: on ARM the
      // writer's coarse byte can become visible before its fine byte, and clearing the coarse byte then lost the fine
      // one (a texture that updated seconds late or never: the grey portrait, the green flashes; 0.266).
      if (!rex_nx_dirty_coarse[c] && !recent_coarse_[c]) continue;
      if (rex_nx_dirty_coarse[c]) recent_coarse_[c] = 2;
      rex_nx_dirty_coarse[c] = 0;
      for (uint32_t w = c * 32; w < c * 32 + 32; w++) {
        if (!words[w]) continue;
        for (uint32_t i = w * 8; i < w * 8 + 8; i++) {
          if (!rex_nx_dirty[i]) continue;
          rex_nx_dirty[i] = 0;
          page_serial_[i] = serial_;
          if (!page_seen_[i]) { page_seen_[i] = 1; frame_pages_.push_back(i); }
          dirty_pages_++;
        }
      }
    }
  }
  // At a frame start: the pages written in the previous frame are stamped again. A vector store (stvx, dcbz, stwcx)
  // marks its page BEFORE the store (REX_RAW_STORE_ADDR), so an upload between the mark and the store read the old
  // bytes and the image looked current from then on. Re-stamping makes the next bind compare again with landed data.
  void RestampRecentPages() {
    serial_++;
    for (uint32_t p : last_frame_pages_) page_serial_[p] = serial_;
    for (uint32_t p : frame_pages_) page_seen_[p] = 0;
    last_frame_pages_.swap(frame_pages_);
    frame_pages_.clear();
    for (uint32_t c = 0; c < 0x200; c++) if (recent_coarse_[c]) recent_coarse_[c]--;
    frame_start_serial_ = serial_;
  }
  bool WrittenSince(uint32_t physical, uint32_t bytes, uint32_t serial) const {
    uint32_t first = (physical & 0x1FFFFFFFu) >> 12;
    uint32_t last = ((physical & 0x1FFFFFFFu) + (bytes ? bytes - 1 : 0)) >> 12;
    for (uint32_t p = first; p <= last && p < page_serial_.size(); p++) {
      if (page_serial_[p] > serial) return true;
    }
    return false;
  }

  // Whether the guest memory behind an image changed since its contents were set. Switch: the dirty-page
  // serials. Linux: a hash of the range against the hash taken when the contents were set.
  uint64_t HashMemory(uint32_t physical, uint32_t bytes) const {
#ifdef TMNT_DK_GL
    const uint8_t* p = memory_->TranslatePhysical(physical);
    uint64_t h = 1469598103934665603ull;
    uint32_t words = bytes / 8;
    const uint64_t* w = reinterpret_cast<const uint64_t*>(p);
    for (uint32_t i = 0; i < words; i++) {
      h ^= w[i];
      h *= 1099511628211ull;
      h ^= h >> 29;
    }
    return h;
#else
    (void)physical; (void)bytes;
    return 0;
#endif
  }
  bool MemoryChanged(const HostImage& img, uint32_t physical, uint32_t bytes) const {
#ifdef TMNT_DK_GL
    (void)bytes;
    return HashMemory(physical, img.mem_bytes) != img.mem_hash;
#else
    return WrittenSince(physical, bytes, img.content_serial);
#endif
  }

  // ----- images ---------------------------------------------------------------------------------
  // gh: images replaced by a resize (or a format change) wait here for a request of the same shape: the
  // pool is a bump allocator and a per-frame resize ping-pong filled 448 MB in minutes (finding 0.22).
  std::vector<HostImage*> free_images_;
  void RecycleImage(HostImage* img) {
    if (img && !img->is_depth) free_images_.push_back(img);
  }

  static bool BlockCompressed(DkImageFormat format) {
    switch (format) {
      case DkImageFormat_RGBA_BC1: case DkImageFormat_RGBA_BC2: case DkImageFormat_RGBA_BC3: case DkImageFormat_RG_BC5_Unorm:
      case DkImageFormat_R_BC4_Unorm: return true;
      default: return false;
    }
  }

  HostImage* CreateImage(uint32_t width, uint32_t height, DkImageFormat format, uint32_t flags, bool is_depth) {
    // Block-compressed formats cannot be render targets: deko3dd rejects the layout (dk_image.cpp:353 "specified
    // format does not support DkImageFlags_UsageRender", Eden 0.68); the release library built it anyway.
    if (BlockCompressed(format)) flags &= ~uint32_t(DkImageFlags_UsageRender);
    for (size_t i = 0; i < free_images_.size(); i++) {
      HostImage* r = free_images_[i];
      if (r->width == width && r->height == height && r->format == format && r->flags == flags && r->is_depth == is_depth) {
        free_images_.erase(free_images_.begin() + i);
        r->from_resolve = false;
        r->rb_swapped = false;
        r->depth_resolve = false;
        r->content_serial = 0;
        r->checked_serial = 0;
        r->guest_base = 0;
        r->edram_base = r->edram_pitch = 0;
        r->resolve_pitch = r->resolve_height = r->resolve_frame = r->resolve_format = r->resolve_endian = 0;
        r->composed_child_serial = 0;
        return r;
      }
    }
    auto* img = new HostImage();
    img->flags = flags;
    DkImageLayoutMaker lm;
    dkImageLayoutMakerDefaults(&lm, device_);
    lm.flags = flags;
    lm.format = format;
    lm.dimensions[0] = width;
    lm.dimensions[1] = height;
    lm.dimensions[2] = 1;
    dkImageLayoutInitialize(&img->layout, &lm);
    uint64_t size = dkImageLayoutGetSize(&img->layout);
    uint32_t align = dkImageLayoutGetAlignment(&img->layout);
    uint32_t off = images_.Alloc(uint32_t(size), align);
    if (off == UINT32_MAX) {
      Log("image pool full (%u MB used): %ux%u not created", images_.used >> 20, width, height);
      delete img;
      return nullptr;
    }
    dkImageInitialize(&img->image, &img->layout, images_.block, off);
    img->format = format;
    img->width = width;
    img->height = height;
    img->is_depth = is_depth;
    image_count_++;
    return img;
  }

  // Fresh pool memory holds whatever an earlier owner left (noise on screen where the game's clears do
  // not reach). Render targets and resolve destinations start cleared to zero instead; textures are
  // fully uploaded, and block-compressed images cannot be cleared.
  void ClearNewImage(HostImage* img) {
    if (!img || !frames_[frame_].cmdbuf || BlockCompressed(img->format)) return;
    DkCmdBuf cmd = frames_[frame_].cmdbuf;
    DkScissor sc{0, 0, img->width, img->height};
    if (img->is_depth) {
      BindTargets(cmd, nullptr, img);
      dkCmdBufSetScissors(cmd, 0, &sc, 1);
      dkCmdBufClearDepthStencil(cmd, true, 1.0f, 0xFF, 0);
    } else {
      const float zero[4] = {0, 0, 0, 0};
      BindTargets(cmd, img, nullptr);
      dkCmdBufSetScissors(cmd, 0, &sc, 1);
      dkCmdBufClearColor(cmd, 0, DkColorMask_RGBA, zero);
    }
    Checkpoint(frames_[frame_], "clear new image guest=%08X edram=%u %ux%u depth=%u", img->guest_base, img->edram_base, img->width, img->height, img->is_depth);
  }

  static uint32_t EffectivePitch(reg::RB_SURFACE_INFO surf) {
    // The XDK clears through a 4x MSAA alias at half the pitch (2x2 samples per pixel: the same
    // EDRAM bytes as the 1x surface, both axes halved).
    return surf.msaa_samples == xenos::MsaaSamples::k4X ? surf.surface_pitch * 2 : surf.surface_pitch;
  }

  HostImage* TargetImage(uint32_t edram_base, uint32_t pitch, DkImageFormat format, bool is_depth) {
    if (!pitch) return nullptr;
    uint64_t key = (uint64_t(edram_base) << 32) | (uint64_t(pitch) << 1) | (is_depth ? 1 : 0);
    auto it = targets_.find(key);
    if (it != targets_.end()) return it->second;
    uint32_t tiles = edram_base < kEdramTiles ? kEdramTiles - edram_base : 0;
    uint32_t rows = tiles ? uint32_t((uint64_t(tiles) * 80 * 16) / pitch) : 0;
    if (rows > kMaxTargetHeight) rows = kMaxTargetHeight;
    if (rows < 16) rows = 16;
    HostImage* img = CreateImage(pitch, rows, format, DkImageFlags_UsageRender, is_depth);
    if (!img) return nullptr;
    img->edram_base = edram_base;
    img->edram_pitch = pitch;
    targets_.emplace(key, img);
    ClearNewImage(img);
    Log("%s target: EDRAM base %u pitch %u -> %ux%u image (%u MB in images)", is_depth ? "depth" : "color", edram_base, pitch,
        img->width, img->height, images_.used >> 20);
    return img;
  }

  bool BindTargetSet(DkCmdBuf cmd, HostImage* const requested[4], HostImage* depth) {
    HostImage* colors[4] = {requested[0], requested[1], requested[2], requested[3]};
    uint32_t count = 0;
    HostImage* size_ref = nullptr;
    for (uint32_t i = 0; i < 4; i++) {
      if (!colors[i]) continue;
      count = i + 1;
      size_ref = colors[i];
      for (uint32_t j = 0; j < i; j++) {
        if (colors[i] == colors[j]) {
          LogOnce("mrtalias", "overlapping MRT slots %u and %u skipped", j, i);
          return false;
        }
      }
    }
    for (uint32_t i = 0; i < count; i++) {
      if (colors[i]) continue;
      uint64_t key = (uint64_t(size_ref->width) << 32) | (uint64_t(size_ref->height) << 2) | i;
      auto it = dummy_targets_.find(key);
      if (it == dummy_targets_.end()) {
        HostImage* dummy = CreateImage(size_ref->width, size_ref->height, DkImageFormat_RGBA8_Unorm, DkImageFlags_UsageRender, false);
        if (!dummy) return false;
        it = dummy_targets_.emplace(key, dummy).first;
      }
      colors[i] = it->second;
    }
    if (std::equal(std::begin(colors), std::end(colors), std::begin(bound_colors_)) && depth == bound_depth_) return true;
    dkCmdBufBarrier(cmd, DkBarrier_Fragments, 0);
    DkImageView cv[4], dv;
    DkImageView const* views[4];
    for (uint32_t i = 0; i < count; i++) {
      dkImageViewDefaults(&cv[i], &colors[i]->image);
      views[i] = &cv[i];
    }
    if (depth) dkImageViewDefaults(&dv, &depth->image);
    dkCmdBufBindRenderTargets(cmd, views, count, depth ? &dv : nullptr);
    std::copy(std::begin(colors), std::end(colors), std::begin(bound_colors_));
    bound_depth_ = depth;
    return true;
  }

  void BindTargets(DkCmdBuf cmd, HostImage* color, HostImage* depth) {
    HostImage* colors[4] = {color, nullptr, nullptr, nullptr};
    BindTargetSet(cmd, colors, depth);
  }

  // The slices of a depth texture resolved one by one (the scene depth: 384 + 336 rows at 1F360000 / 1F540000; the
  // shadow atlas 512x3072 at 1BD18000: 512-row slices at 1BD18000 + n * 0x100000). Each slice after the first is
  // its own depth resolve image; its rows are copied into the parent when any of them is newer than the last copy.
  void ComposeDepthSlices(Frame& f, HostImage& parent) {
    if (!parent.resolve_height || parent.resolve_pitch != parent.width || parent.resolve_height >= parent.height) return;
    struct Part { HostImage* img; uint32_t row, rows; } parts[16];
    uint32_t count = 0, newest = 0;
    for (uint32_t row = parent.resolve_height; row < parent.height && count < 16;) {
      uint64_t child_base = uint64_t(parent.guest_base) + uint64_t(parent.resolve_pitch) * row * 4;
      if (child_base > UINT32_MAX) break;
      auto it = guest_images_.find(uint32_t(child_base));
      if (it == guest_images_.end()) break;
      HostImage& child = *it->second;
      if (!child.depth_resolve || child.width != parent.width || !child.resolve_height ||
          child.resolve_pitch != parent.resolve_pitch || child.height < child.resolve_height) break;
      const uint32_t rows = std::min(child.resolve_height, parent.height - row);
      // A slice older than the parent's own resolve is the previous frame's (the scene depth: the whole 720 rows
      // are resolved after the depth pass, then the first 384 again; the old second slice must not replace them).
      if (child.content_serial >= parent.content_serial) {
        parts[count++] = Part{&child, row, rows};
        newest = std::max(newest, child.content_serial);
      }
      row += rows;
    }
    if (!count || newest == parent.composed_child_serial) return;
    dkCmdBufBarrier(f.cmdbuf, DkBarrier_Full, DkInvalidateFlags_Image);
    for (uint32_t i = 0; i < count; i++) {
      DkImageView sv, dv;
      dkImageViewDefaults(&sv, &parts[i].img->image);
      dkImageViewDefaults(&dv, &parent.image);
      DkImageRect sr{0, 0, 0, parent.width, parts[i].rows, 1};
      DkImageRect dr{0, parts[i].row, 0, parent.width, parts[i].rows, 1};
      dkCmdBufCopyImage(f.cmdbuf, &sv, &sr, &dv, &dr, 0);
    }
    dkCmdBufBarrier(f.cmdbuf, DkBarrier_Full, DkInvalidateFlags_Image);
    parent.composed_child_serial = newest;
    char key[48];
    std::snprintf(key, sizeof(key), "depthparts %08X %u", parent.guest_base, count);
    LogOnce(key, "composed depth slices at %08X: %ux%u + %u slices (the first %u rows at %08X)", parent.guest_base, parent.width,
            parent.resolve_height, count, parts[0].rows, parts[0].img->guest_base);
  }

  void ComposeAdjacentResolve(Frame& f, HostImage& parent) {
    if (parent.depth_resolve) { ComposeDepthSlices(f, parent); return; }
    if (!parent.from_resolve || parent.resolve_frame != swaps_ || parent.resolve_pitch != parent.width ||
        !parent.resolve_height || parent.resolve_height >= parent.height || (parent.resolve_height & 31) ||
        MemoryChanged(parent, parent.guest_base, parent.mem_bytes)) return;
    uint64_t child_base = uint64_t(parent.guest_base) + uint64_t(parent.resolve_pitch) * parent.resolve_height * 4;
    if (child_base > UINT32_MAX) return;
    auto it = guest_images_.find(uint32_t(child_base));
    if (it == guest_images_.end()) return;
    HostImage& child = *it->second;
    if (!child.from_resolve || child.resolve_frame != swaps_ || child.resolve_pitch != parent.resolve_pitch ||
        child.resolve_height != parent.height - parent.resolve_height || child.width != parent.width ||
        child.height < child.resolve_height || child.format != parent.format || child.depth_resolve != parent.depth_resolve ||
        child.resolve_format != parent.resolve_format ||
        child.resolve_endian != parent.resolve_endian || child.content_serial == parent.composed_child_serial ||
        MemoryChanged(child, child.guest_base, child.mem_bytes)) return;
    DkImageView sv, dv;
    dkImageViewDefaults(&sv, &child.image);
    dkImageViewDefaults(&dv, &parent.image);
    DkImageRect sr{0, 0, 0, parent.width, child.resolve_height, 1};
    DkImageRect dr{0, parent.resolve_height, 0, parent.width, child.resolve_height, 1};
    dkCmdBufBarrier(f.cmdbuf, DkBarrier_Full, DkInvalidateFlags_Image);
    dkCmdBufCopyImage(f.cmdbuf, &sv, &sr, &dv, &dr, 0);
    dkCmdBufBarrier(f.cmdbuf, DkBarrier_Full, DkInvalidateFlags_Image);
    parent.composed_child_serial = child.content_serial;
    parent.mem_bytes = uint32_t(child_base - parent.guest_base) + child.mem_bytes;
    parent.mem_hash = HashMemory(parent.guest_base, parent.mem_bytes);
    char ckey[40];
    std::snprintf(ckey, sizeof(ckey), "resolveparts %08X", parent.guest_base);
    LogOnce(ckey, "composed adjacent resolves %08X and %08X: %ux%u + %u rows%s",
            parent.guest_base, child.guest_base, parent.width, parent.resolve_height, child.resolve_height,
            parent.depth_resolve ? " (depth)" : "");
  }

  // The image at a guest address (resolve destinations, textures). A fetch or a swap asks for the
  // exact size (a resolve created it with the aligned height: resized once, contents copied); a
  // resolve asks for a minimum size (`at_least`) and creates with the aligned size (`create_w/h`).
  HostImage* GuestImage(Frame& f, uint32_t base, uint32_t width, uint32_t height, DkImageFormat format, bool at_least,
                        uint32_t create_w = 0, uint32_t create_h = 0) {
    if (!width || !height) { Log("invalid guest image %08X %ux%u", base, width, height); return nullptr; }
    auto it = guest_images_.find(base);
    if (it != guest_images_.end()) {
      HostImage* img = it->second;
      if (img->format != format) {
        // The memory was reused with another format (a texture over an old resolve destination or the
        // reverse): a new image. ponytail: the old image's pool memory is not reclaimed (bump pool);
        // a free list when the pool fills in a stage.
        HostImage* fresh = CreateImage(std::max(width, create_w), std::max(height, create_h), format, DkImageFlags_UsageRender, false);
        if (!fresh) return nullptr;
        fresh->guest_base = base;
        if (create_w || create_h) ClearNewImage(fresh);
        Log("image at %08X recreated: format %u -> %u, %ux%u -> %ux%u", base, uint32_t(img->format), uint32_t(format), img->width,
            img->height, fresh->width, fresh->height);
        it->second = fresh;
        RecycleImage(img);
        return fresh;
      }
      if (img->width == width && img->height == height) return img;
      if (at_least && width <= img->width && height <= img->height) return img;
      if (at_least) {
        width = std::max(width, img->width);
        height = std::max(height, img->height);
      }
      HostImage* resized = CreateImage(width, height, img->format, DkImageFlags_UsageRender, false);
      if (!resized) return at_least ? img : nullptr;
      if (!img->from_resolve) {
        // An uploaded texture whose memory now holds a texture of another size (the game reuses sprite memory:
        // 20-30 size changes per address in a stage): the tiled layout differs, so the old texels are noise.
        // Upload from guest memory again (content_serial 0) instead of copying. A resolve into it (at_least) writes a
        // sub-rectangle: clear the rest (a recycled image holds old texels).
        if (at_least) ClearNewImage(resized);
        resized->guest_base = base;
        it->second = resized;
        RecycleImage(img);
        return resized;
      }
      ClearNewImage(resized);
      uint32_t cw = std::min(width, img->width), ch = std::min(height, img->height);
      DkImageView sv, dv;
      dkImageViewDefaults(&sv, &img->image);
      dkImageViewDefaults(&dv, &resized->image);
      DkImageRect sr{0, 0, 0, cw, ch, 1}, dr{0, 0, 0, cw, ch, 1};
      dkCmdBufBarrier(f.cmdbuf, DkBarrier_Full, DkInvalidateFlags_Image);
      dkCmdBufCopyImage(f.cmdbuf, &sv, &sr, &dv, &dr, 0);
      Checkpoint(f, "resize %08X %ux%u -> %ux%u copy %ux%u", base, img->width, img->height, resized->width, resized->height, cw, ch);
      dkCmdBufBarrier(f.cmdbuf, DkBarrier_Full, DkInvalidateFlags_Image);
      resized->from_resolve = img->from_resolve;
      resized->rb_swapped = img->rb_swapped;
      resized->depth_resolve = img->depth_resolve;
      resized->content_serial = img->content_serial;
      resized->checked_serial = img->checked_serial;
      if (img->resolve_height && img->resolve_height < height) {
        resized->mem_bytes = img->mem_bytes;
        resized->mem_hash = img->mem_hash;
      }
      resized->resolve_pitch = img->resolve_pitch;
      resized->resolve_height = img->resolve_height;
      resized->resolve_frame = img->resolve_frame;
      resized->resolve_format = img->resolve_format;
      resized->resolve_endian = img->resolve_endian;
      resized->guest_base = base;
      Log("image at %08X resized %ux%u -> %ux%u", base, img->width, img->height, width, height);
      it->second = resized;
      RecycleImage(img);  // gh: its pool memory is reused by the next request of its size
      return resized;
    }
    HostImage* img = CreateImage(std::max(width, create_w), std::max(height, create_h), format,
                                 DkImageFlags_UsageRender, false);
    if (!img) return nullptr;
    img->guest_base = base;
    guest_images_.emplace(base, img);
    if (create_w || create_h) ClearNewImage(img);
    return img;
  }

  static bool IsDepthTexture(xenos::TextureFormat format) {
    return format == xenos::TextureFormat::k_24_8 || format == xenos::TextureFormat::k_24_8_FLOAT;
  }

  static bool ResolveFetchFormatMatches(uint32_t resolved, xenos::TextureFormat fetched) {
    using TF = xenos::TextureFormat;
    return resolved == uint32_t(fetched) ||
           (resolved == uint32_t(TF::k_2_10_10_10) && fetched == TF::k_2_10_10_10_AS_16_16_16_16);
  }

  // ----- cube textures --------------------------------------------------------------------------
  // Eden 0.73: "texture fetch 1 not preparable or not 1D/2D (dim 3)": the ambient lighting draw (PS B1581472F8E5082B)
  // and the probe filter passes sample cube textures and were skipped. The game renders its reflection probes face
  // by face and level by level and resolves each to its own address inside the cube (128x128 faces at
  // 1DEFD000 + n * 0x10000, then 64, 32, 16 and 8 texel levels after them). A cube image is 6 layers with every
  // stored level; each face of each level is copied from the resolve image at its guest address, or, for a cube
  // with no resolve image at its first face, uploaded from guest memory.
  struct CubeTex {
    HostImage* img = nullptr;
    uint32_t size = 0, levels = 0;
    bool from_memory = false;
    uint32_t checked_serial = 0;
    std::vector<uint32_t> part_serial;  // [level * 6 + face]: the source's content serial at the copy; 0 = never filled
  };
  std::unordered_map<uint32_t, CubeTex> cubes_;

  HostImage* CreateCube(uint32_t size, uint32_t levels, DkImageFormat format) {
    auto* img = new HostImage();
    DkImageLayoutMaker lm;
    dkImageLayoutMakerDefaults(&lm, device_);
    lm.type = DkImageType_Cubemap;
    lm.flags = 0;
    lm.format = format;
    lm.dimensions[0] = lm.dimensions[1] = size;
    lm.dimensions[2] = 6;
    lm.mipLevels = levels;
    dkImageLayoutInitialize(&img->layout, &lm);
    uint64_t bytes = dkImageLayoutGetSize(&img->layout);
    uint32_t off = images_.Alloc(uint32_t(bytes), dkImageLayoutGetAlignment(&img->layout));
    if (off == UINT32_MAX) {
      Log("image pool full (%u MB used): cube %ux%u not created", images_.used >> 20, size, size);
      delete img;
      return nullptr;
    }
    dkImageInitialize(&img->image, &img->layout, images_.block, off);
    img->format = format;
    img->width = img->height = size;
    image_count_++;
    return img;
  }

  HostImage* CubeImage(Frame& f, const TextureInfo& info, const xenos::xe_gpu_texture_fetch_t& fetch) {
    const uint32_t size = info.width + 1;
    const uint32_t base = info.memory.base_address;
    if (!base || info.height + 1 != size) {
      LogOnce("cubebase", "cube texture without a base level or not square: base %08X mips %08X %ux%u", base, info.memory.mip_address,
              size, info.height + 1);
      return nullptr;
    }
    uint32_t levels = info.mip_max_level + 1;
    while (levels > 1 && !(size >> (levels - 1))) levels--;
    const auto layout = texture_util::GetGuestTextureLayout(xenos::DataDimension::kCube, info.pitch >> 5, size, size, 6, info.is_tiled,
                                                            info.format, info.has_packed_mips, true, levels - 1);
    auto part_address = [&](uint32_t level, uint32_t face) {
      if (!level) return base + face * layout.base.array_slice_stride_bytes;
      const uint32_t stored = std::min(level, layout.packed_level);
      return info.memory.mip_address + layout.mip_offsets_bytes[stored] + face * layout.mips[stored].array_slice_stride_bytes;
    };
    // The resolve image of a face (not for the levels of the packed tail: they share one address).
    auto resolve_at = [&](uint32_t level, uint32_t face) -> HostImage* {
      if (level >= layout.packed_level) return nullptr;
      auto it = guest_images_.find(part_address(level, face));
      if (it == guest_images_.end()) return nullptr;
      HostImage* r = it->second;
      const uint32_t lw = std::max(1u, size >> level);
      return r->from_resolve && !r->depth_resolve && r->content_serial && r->width >= lw && r->height >= lw ? r : nullptr;
    };
    HostImage* first = resolve_at(0, 0);
    DkImageFormat format;
    bool reversed = false;
    if (first) {
      format = first->format;
    } else if (!TextureFormatOf(info.format, format, reversed)) {
      LogOnce((std::string("tfmt ") + info.format_info()->name).c_str(), "unsupported texture format %s at %08X (cube %ux%u, swizzle %03X)",
              info.format_info()->name, base, size, size, fetch.swizzle);
      return nullptr;
    }
    CubeTex& cube = cubes_[base];
    if (!cube.img || cube.size != size || cube.levels != levels || cube.img->format != format) {
      // ponytail: a replaced cube's pool memory is not reclaimed (bump pool); a free list if a stage ever cycles them.
      HostImage* img = CreateCube(size, levels, format);
      if (!img) return nullptr;
      img->guest_base = base;
      cube = CubeTex{img, size, levels, !first, 0, std::vector<uint32_t>(levels * 6)};
      Log("cube texture at %08X: %s %ux%u %s endian %u swizzle %03X, %u levels (fetch mips %u-%u at %08X, packed %u from level %u), "
          "face stride %u bytes, level 1 at %08X, filled from %s (host format %u)",
          base, info.format_info()->name, size, size, info.is_tiled ? "tiled" : "linear", uint32_t(fetch.endianness), fetch.swizzle, levels,
          info.mip_min_level, info.mip_max_level, info.memory.mip_address, info.has_packed_mips, layout.packed_level,
          layout.base.array_slice_stride_bytes, levels > 1 ? part_address(1, 0) : 0, first ? "resolve images" : "guest memory",
          uint32_t(format));
    }
    HostImage* img = cube.img;
    if (cube.checked_serial == serial_) return img;
    cube.checked_serial = serial_;
    img->reversed_channels = reversed;
    const uint32_t bpb = info.format_info()->bytes_per_block();
    uint32_t missing = 0;
    for (uint32_t level = 0; level < levels; level++) {
      const uint32_t lw = std::max(1u, size >> level);
      for (uint32_t face = 0; face < 6; face++) {
        uint32_t& filled = cube.part_serial[level * 6 + face];
        if (!cube.from_memory) {
          HostImage* r = resolve_at(level, face);
          if (!r || r->format != format) { if (!filled) missing++; continue; }
          if (r->content_serial == filled) continue;
          DkImageView sv, dv;
          dkImageViewDefaults(&sv, &r->image);
          dkImageViewDefaults(&dv, &img->image);
          dv.mipLevelOffset = uint8_t(level);
          DkImageRect sr{0, 0, 0, lw, lw, 1}, dr{0, 0, face, lw, lw, 1};
          dkCmdBufBarrier(f.cmdbuf, DkBarrier_Full, DkInvalidateFlags_Image);
          dkCmdBufCopyImage(f.cmdbuf, &sv, &sr, &dv, &dr, 0);
          Checkpoint(f, "cube %08X level %u face %u from resolve %08X %ux%u", base, level, face, r->guest_base, lw, lw);
          dkCmdBufBarrier(f.cmdbuf, DkBarrier_Full, DkInvalidateFlags_Image);
          filled = r->content_serial;
          img->from_resolve = true;
          img->rb_swapped = r->rb_swapped;
        } else {
          const uint32_t stored = std::min(level, layout.packed_level);
          const auto& ll = level ? layout.mips[stored] : layout.base;
          TexSource s{part_address(level, face), ll.array_slice_data_extent_bytes, lw, lw, ll.row_pitch_bytes / bpb, 0, 0, level};
          if (filled && !WrittenSince(s.address, s.bytes, filled)) continue;
          uint32_t oz = 0;
          if (level >= layout.packed_level) texture_util::GetPackedMipOffset(size, size, 1, info.format, level, s.ox, s.oy, oz);
          if (!UploadTexture(f, *img, info, s, fetch.endianness, face, level)) { if (!filled) missing++; continue; }
          filled = serial_;
        }
      }
    }
    if (missing) {
      char key[40];
      std::snprintf(key, sizeof(key), "cubemissing %08X %u", base, missing);
      LogOnce(key, "cube texture at %08X: %u of %u faces not filled yet", base, missing, levels * 6);
    }
    return img;
  }

  HostImage* TextureImage(Frame& f, const TextureInfo& info, const xenos::xe_gpu_texture_fetch_t& fetch) {
    DkImageFormat format;
    bool reversed;
    auto depth_it = guest_images_.find(info.memory.base_address);
    // The saved depth fetched as a 4-byte colour texture (the XDK depth restore reads its raw bytes): the depth
    // resolve image itself, unless the guest wrote that memory since (then it is a texture again).
    const bool depth_alias = !IsDepthTexture(info.format) && depth_it != guest_images_.end() && depth_it->second->depth_resolve &&
                             info.format_info()->bytes_per_block() == 4 &&
                             !MemoryChanged(*depth_it->second, info.memory.base_address, depth_it->second->mem_bytes);
    if (IsDepthTexture(info.format) || depth_alias) {
      // The scene depth: only as the R32F copy a depth resolve made (guest memory holds no depth: nothing to upload).
      auto it = depth_it;
      char key[40];
      std::snprintf(key, sizeof(key), "depthtex %08X", info.memory.base_address);
      if (it == guest_images_.end() || !it->second->depth_resolve) {
        LogOnce(key, "depth texture %s at %08X (%ux%u, swizzle %03X) has no depth resolve image: draw skipped",
                info.format_info()->name, info.memory.base_address, info.width + 1, info.height + 1, fetch.swizzle);
        return nullptr;
      }
      std::snprintf(key, sizeof(key), "depthtex ok %08X", info.memory.base_address);
      LogOnce(key, "depth texture %s at %08X (%ux%u, swizzle %03X): the depth resolve image %ux%u",
              info.format_info()->name, info.memory.base_address, info.width + 1, info.height + 1, fetch.swizzle,
              it->second->width, it->second->height);
      format = DkImageFormat_R32_Float;
      reversed = false;
    } else if (!TextureFormatOf(info.format, format, reversed)) {
      LogOnce((std::string("tfmt ") + info.format_info()->name).c_str(), "unsupported texture format %s at %08X (%ux%u, swizzle %03X)",
              info.format_info()->name, info.memory.base_address, info.width + 1, info.height + 1, fetch.swizzle);
      return nullptr;
    }
    // TextureInfo keeps the fetch constant's fields: the size minus one.
    if (info.width + 1 >= 4000) frame_big_tex_ = true;
    const TexSource src = SourceOf(info);
    if (src.level) frame_mip_only_ = true;
    const uint32_t base = src.address;
    if (!src.level) {
      auto it = guest_images_.find(base);
      if (it != guest_images_.end()) {
        HostImage* resolved = it->second;
        if (resolved->from_resolve && resolved->resolve_height &&
            ResolveFetchFormatMatches(resolved->resolve_format, info.format) &&
            resolved->resolve_endian == uint32_t(fetch.endianness) &&
            !MemoryChanged(*resolved, base, resolved->mem_bytes)) {
          format = resolved->format;
        }
      }
    }
    HostImage* img = GuestImage(f, base, src.width, src.height, format, false);
    if (!img) return nullptr;
    if (img->depth_resolve) return img;  // never uploaded from guest memory
    if (img->checked_serial == serial_) return img;
    img->reversed_channels = reversed;  // already checked for this draw
    img->checked_serial = serial_;
    // Never filled, or written by the guest (or a file read) after the upload / resolve that filled it.
    if (img->content_serial && !MemoryChanged(*img, base, src.bytes)) return img;
    // A failed re-upload (staging ring full) keeps the previous contents rather than unbinding the slot.
    if (!UploadTexture(f, *img, info, src, fetch.endianness)) return img->content_serial ? img : nullptr;
    if (img->content_serial) {
      reuploads_++;
      frame_reuploads_++;
      if (reuploads_ <= 8) Log("texture at %08X re-uploaded (%s, %ux%u): its memory changed", base,
                               img->from_resolve ? "was a resolve destination" : "was uploaded before", img->width, img->height);
    }
    img->from_resolve = false;
    img->rb_swapped = false;
    img->depth_resolve = false;
    img->resolve_pitch = img->resolve_height = 0;
    img->content_serial = serial_;
    img->mem_bytes = src.bytes;
    img->mem_hash = HashMemory(base, img->mem_bytes);
    frame_uploaded_.push_back({img->guest_base, img->mem_bytes, img->content_serial});
#ifdef TMNT_DK_GL
    QueueTextureDump(img, info);
#endif
    return img;
  }

#ifdef TMNT_DK_GL
  // Linux test host: every uploaded texture (first 400, and re-uploads of the first 60 images) is written
  // as a PPM after the frame's commands ran, named by guest address, size, format and upload count.
  void QueueTextureDump(HostImage* img, const TextureInfo& info) {
    static const char* dir = std::getenv("TMNT_DK_DUMP_DIR");
    if (!dir || dump_count_ >= 400) return;
    uint32_t n = ++upload_counts_[img->guest_base];
    if (n > 3) return;
    char name[256];
    std::snprintf(name, sizeof(name), "%s/tex-%08X-%ux%u-%s-%s%u.ppm", dir, img->guest_base, img->width, img->height, info.format_info()->name,
                  info.is_tiled ? "t" : "l", n);
    pending_dumps_.emplace_back(img, name);
    dump_count_++;
  }
  void FlushTextureDumps() {
    for (auto& d : pending_dumps_) dkglDumpImage(&d.first->image, d.second.c_str());
    pending_dumps_.clear();
    // TMNT_DK_DUMP_AT=2900,3000: at those swaps every render target and resolve/texture image is written
    // (after the swap's commands ran).
    // Or on demand: creating the file <dump dir>/dump_now dumps the next swap (the file is removed).
    static const char* at = std::getenv("TMNT_DK_DUMP_AT");
    static const char* dir = std::getenv("TMNT_DK_DUMP_DIR");
    if (!dir) return;
    std::string trigger = std::string(dir) + "/dump_now";
    bool now = (swaps_ & 7) == 0 && access(trigger.c_str(), F_OK) == 0;
    if (now) unlink(trigger.c_str());
    if (!now) {
      if (!at) return;
      std::string list = std::string(",") + at + ",";
      if (list.find("," + std::to_string(swaps_ + 1) + ",") == std::string::npos) return;
    }
    char name[256];
    for (auto& it : targets_) {
      if (it.second->is_depth) continue;
      std::snprintf(name, sizeof(name), "%s/at%u-target-base%u-pitch%u.ppm", dir, swaps_ + 1, it.second->edram_base, it.second->edram_pitch);
      dkglDumpImage(&it.second->image, name);
    }
    for (auto& it : guest_images_) {
      if (it.second->width < 64) continue;
      std::snprintf(name, sizeof(name), "%s/at%u-guest-%08X-%ux%u%s.ppm", dir, swaps_ + 1, it.first, it.second->width, it.second->height,
                    it.second->from_resolve ? "-resolve" : "");
      dkglDumpImage(&it.second->image, name);
    }
  }
  std::vector<std::pair<HostImage*, std::string>> pending_dumps_;
  std::unordered_map<uint32_t, uint32_t> upload_counts_;
  uint32_t dump_count_ = 0;
#endif

  // Where the largest stored level of a texture lives. A fetch without a base level (base address 0, or a minimum
  // mip level above 0: the game streamed it out) keeps its data under the mip address: the full-screen map and the
  // stage overview draw the 4320x2432 castle layers so, and an upload from address 0 drew nothing (console 0.177).
  // The host image is then that mip at its own size; the shader's normalized coordinates sample it the same.
  struct TexSource {
    uint32_t address, bytes, width, height, pitch_blocks, ox, oy, level;
  };
  static TexSource SourceOf(const TextureInfo& info) {
    const uint32_t w0 = info.width + 1, h0 = info.height + 1;  // TextureInfo keeps size minus one
    // Row pitch in blocks, tile-aligned (TextureInfo::pitch is in texels: 4x too wide for DXT).
    TexSource s{info.memory.base_address, info.memory.base_size, w0, h0, info.extent.block_pitch_h, 0, 0, 0};
    uint32_t oz = 0;
    if (info.memory.base_address || !info.memory.mip_address) {
      // A texture whose shorter side is 16 texels or less keeps its base level in the packed mip tail: at an offset
      // inside its 32x32 tile (x 16 for a 16x24 sprite). Read from 0,0, the HUD digits and 16-wide ledge strips were noise.
      if (info.has_packed_mips) texture_util::GetPackedMipOffset(w0, h0, 1, info.format, 0, s.ox, s.oy, oz);
      return s;
    }
    s.level = std::max(1u, info.mip_min_level);
    const auto layout = texture_util::GetGuestTextureLayout(xenos::DataDimension::k2DOrStacked, info.pitch >> 5, w0, h0, 1,
                                                            info.is_tiled, info.format, info.has_packed_mips, false,
                                                            std::max(s.level, info.mip_max_level));
    const uint32_t stored = std::min(s.level, layout.packed_level);  // levels from the packed one on share its tail
    s.address = info.memory.mip_address + layout.mip_offsets_bytes[stored];
    s.bytes = layout.mips[stored].level_data_extent_bytes;
    s.width = std::max(1u, w0 >> s.level);
    s.height = std::max(1u, h0 >> s.level);
    s.pitch_blocks = layout.mips[stored].row_pitch_bytes / info.format_info()->bytes_per_block();
    if (s.level >= layout.packed_level) texture_util::GetPackedMipOffset(w0, h0, 1, info.format, s.level, s.ox, s.oy, oz);
    return s;
  }

  bool UploadTexture(Frame& f, HostImage& img, const TextureInfo& info, const TexSource& s, xenos::Endian endian,
                     uint32_t layer = 0, uint32_t mip = 0) {
    const FormatInfo* fi = info.format_info();
    const uint32_t width = s.width, height = s.height;
    if (!width || !height || width > img.width || height > img.height) {
      Log("invalid upload %08X %ux%u into %ux%u", s.address, width, height, img.width, img.height);
      return false;
    }
    uint32_t bw = (width + fi->block_width - 1) / fi->block_width;
    uint32_t bh = (height + fi->block_height - 1) / fi->block_height;
    uint32_t bpb = fi->bytes_per_block();
    uint32_t bytes = bw * bh * bpb;
    uint32_t off = f.staging.Alloc(bytes, 0x100);
    if (off == UINT32_MAX) {
      static uint32_t full_logged = 0;
      if (!ring_full_swap_) ring_full_swap_ = swaps_ + 1;
      if (full_logged++ < 60) Log("staging ring full at swap #%u (%u KB used of %u KB): %u KB texture %08X not uploaded", swaps_ + 1, f.staging.used >> 10, kStagingBytes >> 10, bytes >> 10, s.address);
      return false;
    }
    uint8_t* dst = f.staging.cpu + off;
    const uint8_t* src = memory_->TranslatePhysical(s.address);
    if (info.is_tiled) {
      if (!::tmnt::UntileFast(dst, src, bw, bh, s.pitch_blocks, bpb, static_cast<::tmnt::SwapMode>(uint32_t(endian)), s.ox, s.oy)) {
        LogOnce("untile", "no untile for %u-byte blocks (%s)", bpb, fi->name);
        return false;
      }
    } else {
      uint32_t row = bw * bpb;
      uint32_t src_row = s.pitch_blocks * bpb;
      src += s.oy * src_row + s.ox * bpb;
      for (uint32_t y = 0; y < bh; y++) texture_conversion::CopySwapBlock(endian, dst + y * row, src + y * src_row, row);
    }
    DkCopyBuf cb{f.staging.gpu + off, 0, 0};
    DkImageView view;
    dkImageViewDefaults(&view, &img.image);
    view.mipLevelOffset = uint8_t(mip);
    DkImageRect rect{0, 0, layer, width, height, 1};
    // A re-upload (the game reuses sprite memory between draws) or a recycled image: earlier draws of this frame
    // may still be sampling it.
    dkCmdBufBarrier(f.cmdbuf, DkBarrier_Full, 0);
    dkCmdBufCopyBufferToImage(f.cmdbuf, &cb, &view, &rect, 0);
    Checkpoint(f, "upload %08X %ux%u image %ux%u", s.address, width, height, img.width, img.height);
    dkCmdBufBarrier(f.cmdbuf, DkBarrier_Full, DkInvalidateFlags_Image);
    uploads_++;
    upload_bytes_ += bytes;
    if (uploads_ <= 8) Log("texture %s %ux%u %s endian %u at %08X: %u KB uploaded", fi->name, width, height,
                           info.is_tiled ? "tiled" : "linear", uint32_t(endian), s.address, bytes >> 10);
    return true;
  }

  uint32_t DescriptorFor(Frame& f, HostImage* img, uint32_t swizzle) {
    uint64_t key = uint64_t(reinterpret_cast<uintptr_t>(img)) ^ (uint64_t(swizzle) << 48);
    auto it = f.descriptors.find(key);
    if (it != f.descriptors.end()) return it->second;
    if (f.next_descriptor >= kMaxImageDescriptors) { LogOnce("desc", "image descriptors exhausted"); return UINT32_MAX; }
    DkImageView view;
    dkImageViewDefaults(&view, &img->image);
    if (swizzle) {
      view.swizzle[0] = SwizzleOf(swizzle);
      view.swizzle[1] = SwizzleOf(swizzle >> 3);
      view.swizzle[2] = SwizzleOf(swizzle >> 6);
      view.swizzle[3] = SwizzleOf(swizzle >> 9);
    }
    DkImageDescriptor desc;
    dkImageDescriptorInitialize(&desc, &view, false, false);
    uint32_t slot = f.next_descriptor++;
    std::memcpy(descriptors_.cpu + f.desc_offset + slot * sizeof(DkImageDescriptor), &desc, sizeof(desc));
    // The GPU caches texture headers: a slot rewritten by the CPU must be invalidated before the draw.
    dkCmdBufBarrier(f.cmdbuf, DkBarrier_None, DkInvalidateFlags_Descriptors);
    f.descriptors.emplace(key, slot);
    return slot;
  }

  // ----- draws ----------------------------------------------------------------------------------
  bool CopyIndices(Frame& f, const IndexBufferInfo& ib, bool restart, uint32_t restart_index,
                   uint32_t& offset_out, uint32_t& vertex_count_out, uint32_t& min_index_out, uint32_t& restart_hits) {
    const uint32_t index_size = ib.format == xenos::IndexFormat::kInt32 ? 4 : 2;
    if (ib.length > UINT32_MAX || uint64_t(ib.count) * index_size > ib.length) {
      LogOnce("indexrange", "invalid index buffer: %u indices, %zu bytes", ib.count, ib.length);
      return false;
    }
    uint32_t bytes = uint32_t(ib.length);
    uint32_t off = f.data.Alloc(bytes, 64);
    if (off == UINT32_MAX) return RingFull(f, "indices", bytes, 0);
    const uint8_t* src = memory_->TranslatePhysical(ib.guest_base);
    uint32_t max_index = 0, min_index = UINT32_MAX;
    bool has_vertex = false;
    if (ib.format == xenos::IndexFormat::kInt32) {
      const uint32_t* s = reinterpret_cast<const uint32_t*>(src);
      uint32_t* d = reinterpret_cast<uint32_t*>(f.data.cpu + off);
      for (uint32_t i = 0; i < ib.count; i++) {
        uint32_t v = xenos::GpuSwap(s[i], ib.endianness);
        if (restart) v &= xenos::kVertexIndexMask;
        d[i] = v;
        if (restart && v == restart_index) { restart_hits++; continue; }
        has_vertex = true;
        if (v > max_index) max_index = v;
        if (v < min_index) min_index = v;
      }
    } else {
      const uint16_t* s = reinterpret_cast<const uint16_t*>(src);
      uint16_t* d = reinterpret_cast<uint16_t*>(f.data.cpu + off);
      auto endian = ib.endianness == xenos::Endian::k8in32 ? xenos::Endian::k8in16 :
                    ib.endianness == xenos::Endian::k16in32 ? xenos::Endian::kNone : ib.endianness;
      for (uint32_t i = 0; i < ib.count; i++) {
        uint16_t v = xenos::GpuSwap(s[i], endian);
        d[i] = v;
        if (restart && v == restart_index) { restart_hits++; continue; }
        has_vertex = true;
        if (v > max_index) max_index = v;
        if (v < min_index) min_index = v;
      }
    }
    offset_out = off;
    if (has_vertex && max_index == UINT32_MAX) { LogOnce("indexmax", "vertex index is too large"); return false; }
    vertex_count_out = has_vertex ? max_index + 1 : 0;
    min_index_out = has_vertex ? min_index : 0;
    return true;
  }

  static void CopyConstantSpan(uint8_t* dst, const uint32_t* src, const DkShaderObj* shader) {
    if (!shader) { std::memcpy(dst, src, 4096); return; }
    // Each run of consecutive registers the shader reads (the sprite pixel shader reads c3-c4 and c254-c255: one
    // span from the lowest to the highest was still 4 KB).
    const auto& map = shader->constant_register_map();
    int run = -1;
    for (int i = 0; i <= 256; i++) {
      const bool used = i < 256 && (map.float_bitmap[i >> 6] >> (i & 63)) & 1;
      if (used && run < 0) run = i;
      if (!used && run >= 0) {
        std::memcpy(dst + run * 16, src + run * 4, size_t(i - run) * 16);
        run = -1;
      }
    }
  }

  bool BindUniforms(Frame& f, DkShaderObj* vs, DkShaderObj* ps, const draw_util::ViewportInfo& vi, const RegisterFile& regs) {
    DkCmdBuf cmd = f.cmdbuf;
    // A block is copied again only when its registers were written since (WriteRegister) or another shader reads
    // another span of them; only the runs of float constants the shader reads are copied (dynamic indexing sets
    // every bit). The deko3d binding is sticky within the frame's command buffer: rebind on a new block only.
    bool bind_vs = false, bind_ps = false;
    if (vs_consts_dirty_ || f.vs_uni == UINT32_MAX || f.vs_uni_shader != vs) {
      const uint32_t off = f.data.Alloc(4096, DK_UNIFORM_BUF_ALIGNMENT);
      if (off == UINT32_MAX) return RingFull(f, "uniforms", 4096, vs->ucode_data_hash());
      CopyConstantSpan(f.data.cpu + off, &regs.values[XE_GPU_REG_SHADER_CONSTANT_000_X], vs);
      f.vs_uni = off;
      f.vs_uni_shader = vs;
      vs_consts_dirty_ = false;
      bind_vs = true;
      uni_uploads_[0]++;
    }
    if (ps_consts_dirty_ || f.ps_uni == UINT32_MAX || f.ps_uni_shader != ps) {
      const uint32_t off = f.data.Alloc(4096, DK_UNIFORM_BUF_ALIGNMENT);
      if (off == UINT32_MAX) return RingFull(f, "uniforms", 4096, vs->ucode_data_hash());
      CopyConstantSpan(f.data.cpu + off, &regs.values[XE_GPU_REG_SHADER_CONSTANT_256_X], ps);
      f.ps_uni = off;
      f.ps_uni_shader = ps;
      ps_consts_dirty_ = false;
      bind_ps = true;
      uni_uploads_[1]++;
    }
    SysConstants sys{};
    for (int i = 0; i < 3; i++) {
      sys.ndc_scale[i] = vi.ndc_scale[i];
      sys.ndc_offset[i] = vi.ndc_offset[i];
    }
    auto cc = regs.Get<reg::RB_COLORCONTROL>();
    sys.alpha[0] = cc.alpha_test_enable ? 1.0f : 0.0f;
    sys.alpha[1] = float(uint32_t(cc.alpha_func));
    sys.alpha[2] = regs.Get<float>(XE_GPU_REG_RB_ALPHA_REF);
    sys.alpha[3] = 0.0f;  // was the DBG TEX LOD switch (0.258-0.276); the fetches use the computed LOD now
    std::memcpy(sys.bools, &regs.values[XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031], sizeof(sys.bools));
    std::memcpy(sys.loops, &regs.values[XE_GPU_REG_SHADER_CONSTANT_LOOP_00], sizeof(sys.loops));
    if (f.sys_uni == UINT32_MAX || std::memcmp(&sys, &f.last_sys, sizeof(sys)) != 0) {
      const uint32_t off = f.data.Alloc(sizeof(SysConstants), DK_UNIFORM_BUF_ALIGNMENT);
      if (off == UINT32_MAX) return RingFull(f, "uniforms", 4096, vs->ucode_data_hash());
      std::memcpy(f.data.cpu + off, &sys, sizeof(sys));
      f.sys_uni = off;
      f.last_sys = sys;
      bind_vs = bind_ps = true;
      uni_uploads_[2]++;
    }
    if (bind_vs) {
      DkBufExtents vs_bufs[2] = {{f.data.gpu + f.vs_uni, 4096}, {f.data.gpu + f.sys_uni, sizeof(SysConstants)}};
      dkCmdBufBindUniformBuffers(cmd, DkStage_Vertex, 0, vs_bufs, 2);
    }
    if (bind_ps) {
      DkBufExtents ps_bufs[2] = {{f.data.gpu + f.ps_uni, 4096}, {f.data.gpu + f.sys_uni, sizeof(SysConstants)}};
      dkCmdBufBindUniformBuffers(cmd, DkStage_Fragment, 0, ps_bufs, 2);
    }
    return true;
  }

  // The XDK's Clear: rectangles (position xyz + color rgba, stride 7 words) with the depth
  // function "always"; half of them through the 2x MSAA half-pitch alias of the surface.
  bool IsClearDraw(DkShaderObj* vs, DkShaderObj* ps, xenos::PrimitiveType prim) {
    if (prim != xenos::PrimitiveType::kRectangleList || !ps->texture_bindings().empty() || (ps->writes_color_targets() & ~1u)) return false;
    const auto& vbs = vs->vertex_bindings();
    if (vbs.size() != 1 || vbs[0].stride_words != 7 || vbs[0].attributes.size() != 2) return false;
    const auto& a0 = vbs[0].attributes[0].fetch_instr.attributes;
    const auto& a1 = vbs[0].attributes[1].fetch_instr.attributes;
    return a0.data_format == xenos::VertexFormat::k_32_32_32_FLOAT && a0.offset == 0 &&
           a1.data_format == xenos::VertexFormat::k_32_32_32_32_FLOAT && a1.offset == 3;
  }

  bool ClearDraw(uint32_t vertex_count, xenos::EdramMode mode) {
    const RegisterFile& regs = *register_file_;
    Frame& f = frames_[frame_];
    DkCmdBuf cmd = f.cmdbuf;
    auto fetch = regs.GetVertexFetch(0);
    const uint32_t* words = reinterpret_cast<const uint32_t*>(memory_->TranslatePhysical(fetch.address << 2));
    auto surf = regs.Get<reg::RB_SURFACE_INFO>();
    auto color_info = regs.Get<reg::RB_COLOR_INFO>();
    auto depth_info = regs.Get<reg::RB_DEPTH_INFO>();
    uint32_t scale = surf.msaa_samples == xenos::MsaaSamples::k4X ? 2 : 1;
    uint32_t pitch = EffectivePitch(surf);
    auto word = [&](uint32_t i) {
      uint32_t w = words[i];
      if (fetch.endian == xenos::Endian::k8in32) w = __builtin_bswap32(w);
      float v;
      std::memcpy(&v, &w, 4);
      return v;
    };
    for (uint32_t r = 0; r + 3 <= vertex_count && r * 7 + 21 <= fetch.size; r += 3) {
      float x0 = 1e9f, y0 = 1e9f, x1 = -1e9f, y1 = -1e9f;
      for (uint32_t v = 0; v < 3; v++) {
        float x = word((r + v) * 7), y = word((r + v) * 7 + 1);
        x0 = std::min(x0, x); x1 = std::max(x1, x); y0 = std::min(y0, y); y1 = std::max(y1, y);
      }
      float z = word(r * 7 + 2);
      float color[4] = {word(r * 7 + 3), word(r * 7 + 4), word(r * 7 + 5), word(r * 7 + 6)};
      DkScissor sc;
      sc.x = uint32_t(std::max(0.0f, std::floor(x0 + 0.5f))) * scale;
      sc.y = uint32_t(std::max(0.0f, std::floor(y0 + 0.5f))) * scale;
      sc.width = uint32_t(std::max(0.0f, std::floor(x1 + 0.5f) - std::floor(x0 + 0.5f))) * scale;
      sc.height = uint32_t(std::max(0.0f, std::floor(y1 + 0.5f) - std::floor(y0 + 0.5f))) * scale;
      HostImage* color_img = nullptr;
      HostImage* depth_img = nullptr;
      if (mode == xenos::EdramMode::kColorDepth) {
        color_img = TargetImage(color_info.color_base, pitch, ColorTargetFormatOf(color_info.color_format), false);
        depth_img = TargetImage(depth_info.depth_base, pitch, DkImageFormat_Z24S8, true);
      } else if (mode == xenos::EdramMode::kDepthOnly) {
        // Depth-only through the alias: the XDK fills a color surface's EDRAM this way too.
        auto it = targets_.find((uint64_t(depth_info.depth_base) << 32) | (uint64_t(pitch) << 1));
        if (it != targets_.end()) color_img = it->second;
        else depth_img = TargetImage(depth_info.depth_base, pitch, DkImageFormat_Z24S8, true);
      }
      if (!color_img && !depth_img) continue;
      BindTargets(cmd, color_img, depth_img);
      uint32_t w = color_img ? color_img->width : depth_img->width, h = color_img ? color_img->height : depth_img->height;
      if (sc.x >= w || sc.y >= h) continue;
      if (sc.x + sc.width > w) sc.width = w - sc.x;
      if (sc.y + sc.height > h) sc.height = h - sc.y;
      dkCmdBufSetScissors(cmd, 0, &sc, 1);
      if (color_img) dkCmdBufClearColor(cmd, 0, DkColorMask_RGBA, color);
      if (depth_img) dkCmdBufClearDepthStencil(cmd, true, z, 0xFF, 0);
      Checkpoint(f, "clear EDRAM color=%u depth=%u rect %u,%u %ux%u", color_info.color_base, depth_info.depth_base, sc.x, sc.y, sc.width, sc.height);
      {
        char key[96];
        std::snprintf(key, sizeof(key), "clear %u %u %u %u %u %u %u %u", uint32_t(mode), color_info.color_base, depth_info.depth_base, pitch,
                      sc.x, sc.y, sc.width, sc.height);
        LogOnce(key, "clear %s%s at EDRAM %u/%u pitch %u: rect %u,%u %ux%u color %g %g %g %g z %g (alias x%u)", color_img ? "color" : "",
                depth_img ? " depth" : "", color_info.color_base, depth_info.depth_base, pitch, sc.x, sc.y, sc.width, sc.height, color[0],
                color[1], color[2], color[3], z, scale);
      }
      clears_++;
      frame_clears_++;
    }
    return true;
  }

  // A depth resolve. The lighting shaders sample the scene depth as a k_24_8_FLOAT texture (fetch 15; Eden 0.73:
  // every lighting draw was skipped for that texture). The host depth value is the guest's (the viewport keeps the
  // guest depth range), so the texture is an R32F image holding it in red. The copy engine cannot turn depth into
  // colour: a fullscreen draw (present.vert + depthcopy.frag) does. Every draw binds its whole state again, so
  // nothing is restored here.
  void ResolveDepth(Frame& f, const draw_util::ResolveInfo& info) {
    DkCmdBuf cmd = f.cmdbuf;
    const uint32_t src_pitch = info.depth_edram_info.pitch_tiles * 80;
    uint32_t x = info.coordinate_info.edram_offset_x_div_8 * 8, y = info.coordinate_info.edram_offset_y_div_8 * 8;
    uint32_t w = info.coordinate_info.width_div_8 * 8, h = info.height_div_8 * 8;
    const uint32_t dest_pitch = info.copy_dest_coordinate_info.pitch_aligned_div_32 * 32;
    const uint32_t dest_height = info.copy_dest_coordinate_info.height_aligned_div_32 * 32;
    const uint32_t dx = info.copy_dest_coordinate_info.offset_x_div_8 * 8, dy = info.copy_dest_coordinate_info.offset_y_div_8 * 8;
    {
      char key[96];
      std::snprintf(key, sizeof(key), "depthresolve %u %u %u %u %u %u %08X %u %u", info.depth_edram_info.base_tiles, src_pitch, x, y, w, h,
                    info.copy_dest_base, dx, dy);
      LogOnce(key, "depth resolve EDRAM %u format %u pitch %u rect %u,%u %ux%u -> %08X at %u,%u (dest format %u endian %u pitch %u rows %u)%s",
              info.depth_edram_info.base_tiles, uint32_t(info.depth_edram_info.format), src_pitch, x, y, w, h, info.copy_dest_base, dx, dy,
              uint32_t(info.copy_dest_info.copy_dest_format), uint32_t(info.copy_dest_info.copy_dest_endian), dest_pitch, dest_height,
              depth_fs_ && present_vs_ ? "" : " ignored: no depthcopy shader");
    }
    if (!depth_fs_ || !present_vs_) return;
    HostImage* src = TargetImage(info.depth_edram_info.base_tiles, src_pitch, DkImageFormat_Z24S8, true);
    if (!src) return;
    HostImage* dst = GuestImage(f, info.copy_dest_base, dx + w, dy + h, DkImageFormat_R32_Float, true);
    if (!dst) return;
    dst->from_resolve = true;
    dst->rb_swapped = false;
    dst->depth_resolve = true;
    dst->content_serial = serial_;
    dst->resolve_pitch = dst->resolve_height = dst->composed_child_serial = 0;
    dst->mem_bytes = dest_pitch * dest_height * 4;
    dst->mem_hash = HashMemory(info.copy_dest_base, dst->mem_bytes);
    if (x + w > src->width) w = x < src->width ? src->width - x : 0;
    if (y + h > src->height) h = y < src->height ? src->height - y : 0;
    if (dx + w > dst->width) w = dx < dst->width ? dst->width - dx : 0;
    if (dy + h > dst->height) h = dy < dst->height ? dst->height - dy : 0;
    const uint32_t slot = w && h ? DescriptorFor(f, src, 0) : UINT32_MAX;
    if (slot == UINT32_MAX) return;
    dkCmdBufBarrier(cmd, DkBarrier_Full, DkInvalidateFlags_Image);
    BindTargets(cmd, dst, nullptr);
    // The triangle's uv spans the viewport: the whole source image, placed so that source x,y lands on dx,dy.
    DkViewport vp{float(dx) - float(x), float(dy) - float(y), float(src->width), float(src->height), 0.0f, 1.0f};
    dkCmdBufSetViewports(cmd, 0, &vp, 1);
    DkScissor sc{dx, dy, w, h};
    dkCmdBufSetScissors(cmd, 0, &sc, 1);
    DkRasterizerState rs;
    dkRasterizerStateDefaults(&rs);
    rs.cullMode = DkFace_None;
    dkCmdBufBindRasterizerState(cmd, &rs);
    DkColorState cs;
    dkColorStateDefaults(&cs);
    dkCmdBufBindColorState(cmd, &cs);
    DkColorWriteState cw;
    dkColorWriteStateDefaults(&cw);
    dkCmdBufBindColorWriteState(cmd, &cw);
    DkDepthStencilState ds;
    dkDepthStencilStateDefaults(&ds);
    ds.depthTestEnable = 0;
    ds.depthWriteEnable = 0;
    dkCmdBufBindDepthStencilState(cmd, &ds);
    DkShader const* shaders[2] = {&present_vs_shader_, &depth_fs_shader_};
    dkCmdBufBindShaders(cmd, DkStageFlag_GraphicsMask, shaders, 2);
    dkCmdBufBindVtxAttribState(cmd, nullptr, 0);
    dkCmdBufBindVtxBufferState(cmd, nullptr, 0);
    const uint32_t clamp = (uint32_t(xenos::ClampMode::kClampToEdge) << 3) | (uint32_t(xenos::ClampMode::kClampToEdge) << 6);
    DkResHandle handle = dkMakeTextureHandle(slot, clamp);  // nearest
    dkCmdBufBindTextures(cmd, DkStage_Fragment, 0, &handle, 1);
    dkCmdBufDraw(cmd, DkPrimitive_Triangles, 3, 1, 0, 0);
    Checkpoint(f, "depth resolve EDRAM %u %ux%u rect %u,%u %ux%u -> %08X %ux%u at %u,%u", info.depth_edram_info.base_tiles, src->width,
               src->height, x, y, w, h, info.copy_dest_base, dst->width, dst->height, dx, dy);
    dkCmdBufBarrier(cmd, DkBarrier_Full, DkInvalidateFlags_Image);
    if (!dx && !dy && w == dest_pitch && !(w & 31)) {
      dst->resolve_pitch = dest_pitch;
      dst->resolve_height = h;
      dst->resolve_frame = swaps_;
      dst->resolve_format = uint32_t(info.copy_dest_info.copy_dest_format);
      dst->resolve_endian = uint32_t(info.copy_dest_info.copy_dest_endian);
    }
    resolves_++;
    frame_resolves_++;
  }

  bool Resolve() {
    const RegisterFile& regs = *register_file_;
    Frame& f = frames_[frame_];
    DkCmdBuf cmd = f.cmdbuf;
    draw_util::ResolveInfo info;
    if (!draw_util::GetResolveInfo(regs, *memory_, 1, 1, false, false, info)) { LogOnce("resolve", "GetResolveInfo failed"); return true; }
    if (info.IsCopyingDepth()) {
      ResolveDepth(f, info);
      return true;
    }
    if (info.color_edram_info.format_is_64bpp) {
      LogOnce("resolve64", "64-bit color resolve and clear at EDRAM %u are unsupported", info.color_edram_info.base_tiles);
      return true;
    }
    uint32_t src_pitch = info.color_edram_info.pitch_tiles * 80;
    HostImage* src = TargetImage(info.color_edram_info.base_tiles, src_pitch, DkImageFormat_RGBA8_Unorm, false);
    if (!src) return true;
    uint32_t x = info.coordinate_info.edram_offset_x_div_8 * 8, y = info.coordinate_info.edram_offset_y_div_8 * 8;
    uint32_t w = info.coordinate_info.width_div_8 * 8, h = info.height_div_8 * 8;
    auto src_format = xenos::ColorRenderTargetFormat(info.color_edram_info.format);
    const bool packed = src->format == DkImageFormat_RGB10A2_Unorm &&
                        src_format == xenos::ColorRenderTargetFormat::k_2_10_10_10 &&
                        xenos::IsColorResolveFormatBitwiseEquivalent(src_format, info.copy_dest_info.copy_dest_format) &&
                        !info.copy_dest_info.copy_dest_exp_bias;
    if (!packed && src->format != DkImageFormat_RGBA8_Unorm && src->format != DkImageFormat_RGBA8_Unorm_sRGB) {
      LogOnce("resolvefmt", "color resolve EDRAM %u format %u to format %u bias %d swap %u -> %08X is unsupported", info.color_edram_info.base_tiles,
              uint32_t(src_format), uint32_t(info.copy_dest_info.copy_dest_format), info.copy_dest_info.copy_dest_exp_bias,
              info.copy_dest_info.copy_dest_swap, info.copy_dest_base);
      if (info.IsClearingColor()) {
        BindTargets(cmd, src, nullptr);
        DkScissor sc{x, y, x < src->width ? std::min(w, src->width - x) : 0,
                     y < src->height ? std::min(h, src->height - y) : 0};
        dkCmdBufSetScissors(cmd, 0, &sc, 1);
        uint32_t c = info.rb_color_clear;
        float clear[4] = {((c >> 16) & 0xFF) / 255.0f, ((c >> 8) & 0xFF) / 255.0f,
                          (c & 0xFF) / 255.0f, ((c >> 24) & 0xFF) / 255.0f};
        dkCmdBufClearColor(cmd, 0, DkColorMask_RGBA, clear);
      }
      return true;
    }
    uint32_t dest_pitch = info.copy_dest_coordinate_info.pitch_aligned_div_32 * 32;
    uint32_t dest_height = info.copy_dest_coordinate_info.height_aligned_div_32 * 32;
    (void)dest_pitch; (void)dest_height;
    uint32_t dx = info.copy_dest_coordinate_info.offset_x_div_8 * 8, dy = info.copy_dest_coordinate_info.offset_y_div_8 * 8;
    HostImage* dst = GuestImage(f, info.copy_dest_base, dx + w, dy + h,
                                packed ? DkImageFormat_RGB10A2_Unorm : DkImageFormat_RGBA8_Unorm, true);
    if (!dst) return true;
    dst->from_resolve = true;
    dst->depth_resolve = false;
    // RB_COPY_DEST_INFO swap exchanges R and B in the written texels. The image keeps the target's channels: see
    // ResolveSwizzle for the bind.
    dst->rb_swapped = info.copy_dest_info.copy_dest_swap != 0;
    if (dst->rb_swapped) {
      char key[32];
      std::snprintf(key, sizeof key, "resolveswap %08X", info.copy_dest_base);
      LogOnce(key, "resolve EDRAM %u format %u -> %08X with red/blue swap", info.color_edram_info.base_tiles,
              uint32_t(src_format), info.copy_dest_base);
    }
    dst->content_serial = serial_;
    dst->resolve_pitch = dst->resolve_height = dst->composed_child_serial = 0;
    dst->mem_bytes = dest_pitch * dest_height * 4;
    dst->mem_hash = HashMemory(info.copy_dest_base, dst->mem_bytes);
    if (x + w > src->width) w = x < src->width ? src->width - x : 0;
    if (y + h > src->height) h = y < src->height ? src->height - y : 0;
    if (dx + w > dst->width) w = dx < dst->width ? dst->width - dx : 0;
    if (dy + h > dst->height) h = dy < dst->height ? dst->height - dy : 0;
    if (w && h) {
      DkImageView sv, dv;
      dkImageViewDefaults(&sv, &src->image);
      dkImageViewDefaults(&dv, &dst->image);
      DkImageRect sr{x, y, 0, w, h, 1}, dr{dx, dy, 0, w, h, 1};
      dkCmdBufBarrier(cmd, DkBarrier_Full, DkInvalidateFlags_Image);
      dkCmdBufCopyImage(cmd, &sv, &sr, &dv, &dr, 0);
      if (!dx && !dy && w == dest_pitch && !(w & 31)) {
        dst->resolve_pitch = dest_pitch;
        dst->resolve_height = h;
        dst->resolve_frame = swaps_;
        dst->resolve_format = uint32_t(info.copy_dest_info.copy_dest_format);
        dst->resolve_endian = uint32_t(info.copy_dest_info.copy_dest_endian);
      }
      Checkpoint(f, "resolve EDRAM %u %ux%u rect %u,%u %ux%u -> %08X %ux%u at %u,%u", info.color_edram_info.base_tiles, src->width, src->height, x, y, w, h, info.copy_dest_base, dst->width, dst->height, dx, dy);
      dkCmdBufBarrier(cmd, DkBarrier_Full, DkInvalidateFlags_Image);
    }
    if (info.IsClearingColor()) {
      BindTargets(cmd, src, nullptr);
      DkScissor sc{x, y, w, h};
      dkCmdBufSetScissors(cmd, 0, &sc, 1);
      uint32_t c = info.rb_color_clear;
      float color[4] = {((c >> 16) & 0xFF) / 255.0f, ((c >> 8) & 0xFF) / 255.0f, (c & 0xFF) / 255.0f, ((c >> 24) & 0xFF) / 255.0f};
      dkCmdBufClearColor(cmd, 0, DkColorMask_RGBA, color);
      Checkpoint(f, "resolve clear EDRAM %u rect %u,%u %ux%u", info.color_edram_info.base_tiles, x, y, w, h);
    }
    {
      // Every distinct resolve (source rect, destination and offset) once: the frame structure.
      char key[96];
      std::snprintf(key, sizeof(key), "resolve %u %u %u %u %u %u %08X %u %u", info.color_edram_info.base_tiles, src_pitch, x, y, w, h,
                    info.copy_dest_base, dx, dy);
      LogOnce(key, "resolve EDRAM %u pitch %u rect %u,%u %ux%u -> %08X (%ux%u image) at %u,%u%s", info.color_edram_info.base_tiles, src_pitch,
              x, y, w, h, info.copy_dest_base, dst->width, dst->height, dx, dy, info.IsClearingColor() ? ", then cleared" : "");
    }
    resolves_++;
    frame_resolves_++;
    return true;
  }

  // ----- shaders --------------------------------------------------------------------------------
  // The same two files as Shader::DumpUcode (shader_<hash>.ucode.bin.<ext> and the disassembly
  // shader_<hash>.ucode.<ext>, the translator's input), written with stdio: the SDK's version goes
  // through std::filesystem::absolute, which turns sdmc:/x into /sdmc:/x on newlib (finding 0.41).
  void DumpMissing(const DkShaderObj& shader) {
    std::string dir = shader_dir_ + "missing/";
    mkdir(dir.c_str(), 0777);
    const char* ext = shader.type() == xenos::ShaderType::kVertex ? "vert" : "frag";
    char name[512];
    std::snprintf(name, sizeof(name), "%sshader_%016llX.ucode.bin.%s", dir.c_str(), (unsigned long long)shader.ucode_data_hash(), ext);
    if (FILE* fp = std::fopen(name, "wb")) {
      std::fwrite(shader.ucode_data().data(), 4, shader.ucode_data().size(), fp);
      std::fclose(fp);
    } else {
      Log("cannot write %s", name);
    }
    std::snprintf(name, sizeof(name), "%sshader_%016llX.ucode.%s", dir.c_str(), (unsigned long long)shader.ucode_data_hash(), ext);
    if (FILE* fp = std::fopen(name, "w")) {
      std::fwrite(shader.ucode_disassembly().data(), 1, shader.ucode_disassembly().size(), fp);
      std::fclose(fp);
    }
  }

  bool LoadDksh(const std::string& path, DkShader& out) {
#ifdef TMNT_DK_GL
    // Linux: the translated GLSL next to where the DKSH would be (name.dksh -> name.glsl).
    std::string glsl = path.size() > 5 && path.compare(path.size() - 5, 5, ".dksh") == 0 ? path.substr(0, path.size() - 5) + ".glsl" : path;
    FILE* gp = std::fopen(glsl.c_str(), "rb");
    if (!gp) return false;
    std::string text;
    char buf[4096];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), gp)) > 0) text.append(buf, n);
    std::fclose(gp);
    DkStage stage = path.find(".vert") != std::string::npos ? DkStage_Vertex : path.find(".geom") != std::string::npos ? DkStage_Geometry : DkStage_Fragment;
    dkShaderInitializeGlsl(&out, stage, text.c_str());
    return true;
#else
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) return false;
    struct { uint32_t magic, header_sz, control_sz, code_sz, programs_off, num_programs; } hdr;
    if (std::fread(&hdr, sizeof(hdr), 1, fp) != 1 || hdr.control_sz < sizeof(hdr) || hdr.control_sz > (1u << 20) || hdr.code_sz > (4u << 20)) {
      std::fclose(fp);
      return false;
    }
    std::vector<uint8_t> control(hdr.control_sz);
    std::rewind(fp);
    if (std::fread(control.data(), hdr.control_sz, 1, fp) != 1) { std::fclose(fp); return false; }
    uint32_t off = code_.Alloc(hdr.code_sz, DK_SHADER_CODE_ALIGNMENT);
    if (off == UINT32_MAX) { std::fclose(fp); Log("shader code pool full"); return false; }
    if (std::fread(code_.cpu + off, hdr.code_sz, 1, fp) != 1) { std::fclose(fp); return false; }
    std::fclose(fp);
    DkShaderMaker sm;
    dkShaderMakerDefaults(&sm, code_.block, off);
    sm.control = control.data();
    sm.programId = 0;
    dkShaderInitialize(&out, &sm);
    return true;
#endif
  }

  // A draw that does not fit the frame data ring is dropped: counted per frame, the first one described (the swap logs it).
  bool RingFull(const Frame& f, const char* what, uint32_t bytes, uint64_t vs_hash) {
    if (!frame_ring_full_++) {
      std::snprintf(ring_first_, sizeof(ring_first_), "draw #%u VS %016llX: %u bytes of %s at %u KB used", frame_draws_,
                    (unsigned long long)vs_hash, bytes, what, f.data.used >> 10);
    }
    return false;
  }
  // First sighting of a numeric key (the per-draw diagnostics: no string building on the command thread).
  bool FirstTime(uint64_t key) { return seen_.insert(key).second; }
  void LogOnce(const char* key, const char* fmt, ...) {
    if (!logged_.insert(key).second) return;
    va_list ap;
    va_start(ap, fmt);
    std::printf("[dk] ");
    std::vprintf(fmt, ap);
    std::printf("\n");
    va_end(ap);
    std::fflush(stdout);
  }

  DkDevice device_ = nullptr;
  DkQueue queue_ = nullptr;
  DkSwapchain swapchain_ = nullptr;
  MemBlock code_, images_, descriptors_, cmdmem_;
  uint32_t sampler_desc_offset_ = 0;
  Frame frames_[kFrames];
  uint32_t frame_ = 0;
  HostImage* swap_images_[kFrames] = {};
  DkImage const* swap_image_ptrs_[kFrames] = {};
  DkShader present_vs_shader_{}, present_fs_shader_{};
  bool present_vs_ = false, present_fs_ = false;
  DkShader depth_fs_shader_{};  // depthcopy.frag (ResolveDepth)
  bool depth_fs_ = false;
  DkShader restore_fs_shader_{};  // depthrestore.frag (the XDK depth restore draw)
  bool restore_fs_ = false;
  bool present_target_ = false;
#ifndef TMNT_DK_GL
  // Port menu RECORD 5 SEC (nx_menu.cpp, switch/host/frame_recorder.cpp, from hcu-switch): the 1920x1080 guest front buffer of each
  // drawn frame is copied into rec_ (one slot per frame in flight) and handed to the recorder once the frame's fence has
  // passed (BeginFrame): the command thread never waits for it. The file sdmc:/switch/tmnt/record_now does the same
  // (scripted runs; removed when seen, checked with the fps mode every 60 swaps).
  static constexpr uint32_t kRecW = 1920, kRecH = 1080;  // the guest front buffer (no HwCompression: a plain copy)
  MemBlock rec_;
  bool rec_on_ = false;
  uint64_t rec_start_tick_ = 0;
  void RecordFrame(Frame& f, DkCmdBuf cmd, HostImage* shown) {
    if (!rec_on_ && rex_nx_debug_record()) {
      if (!rec_.block && !rec_.Create(device_, kRecW * kRecH * 4 * kFrames, DkMemBlockFlags_CpuCached | DkMemBlockFlags_GpuCached)) {
        Log("RECORD 5 SEC: no memory block");
        rex_nx_debug_record_set(0);
        return;
      }
      if (!rex_rec_start((debug_dir_ + "rec/").c_str(), swaps_ + 1, kRecW, kRecH)) {
        rex_nx_debug_record_set(0);
        return;
      }
      rec_on_ = true;
      rec_start_tick_ = armGetSystemTick();
    }
    if (!rec_on_) return;
    const uint64_t us = armTicksToNs(armGetSystemTick() - rec_start_tick_) / 1000;
    if (us > 5000000) {
      // Done: the frames still in flight are handed over, then the writer finishes the file.
      dkQueueWaitIdle(queue_);
      const uint32_t bytes = kRecW * kRecH * 4;
      for (uint32_t i = 0; i < kFrames; i++) {
        if (!frames_[i].rec_pending) continue;
        dkMemBlockFlushCpuCache(rec_.block, i * bytes, bytes);
        rex_rec_push(rec_.cpu + i * bytes, frames_[i].rec_swap, frames_[i].rec_us);
        frames_[i].rec_pending = false;
      }
      rex_rec_stop();
      rec_on_ = false;
      rex_nx_debug_record_set(0);
      return;
    }
    if (!shown || shown->width != kRecW || shown->height != kRecH || shown->format != DkImageFormat_RGBA8_Unorm) return;
    DkImageView v;
    dkImageViewDefaults(&v, &shown->image);
    DkImageRect r{0, 0, 0, kRecW, kRecH, 1};
    DkCopyBuf cb{rec_.gpu + uint64_t(frame_) * kRecW * kRecH * 4, 0, 0};
    dkCmdBufBarrier(cmd, DkBarrier_Full, DkInvalidateFlags_Image);
    dkCmdBufCopyImageToBuffer(cmd, &v, &r, &cb, 0);
    Checkpoint(f, "record readback %08X %ux%u", shown->guest_base, kRecW, kRecH);
    f.rec_pending = true;
    f.rec_swap = swaps_ + 1;
    f.rec_us = us;
  }
#endif
  bool skip_odd_ = false;   // ReadFpsMode: the 30 mode
  bool skip_frame_ = false; // this frame is not drawn (NextFrame)
  std::unordered_set<uint64_t> ok_pairs_, skip_pairs_;  // PairKey: ran in a finished frame / skipped (dk_skip)
  uint32_t busy_run_ = 0;   // frames skipped in a row for a busy GPU
  std::chrono::steady_clock::time_point next_swap_{};  // the "30" swap limiter (IssueSwap)
  bool begun_ = true;       // frames_[frame_] is ready for recording (BeginFrame ran)
  uint32_t busy_skips_ = 0;
  double shown_last_ms_ = 0;      // the last present (the shown-frame intervals of the dk time/s line)
  std::vector<double> shown_ms_;
  bool gpu_reported_ = false;
  bool gpu_checkpoints_ = false;  // opt-in: sdmc:/switch/tmnt/dk_gpu_checkpoints
  bool dump_images_ = false;
  bool trace_enabled_ = false, traced_ = false, trace_any_ = false;  // file dk_trace: one line per draw of the first frame with 440+ draws
  uint32_t trace_min_swap_ = 0;  // seconds after the backend started
  double trace_start_ms_ = 0;
  uint32_t trace_second_ = 0;
  uint32_t frame_ring_full_ = 0, ring_full_logs_ = 0;  // draws dropped for a full frame data ring: this frame, frames logged
  char ring_first_[160] = "";
  uint32_t big_copy_logs_ = 0;
  uint32_t ring_full_swap_ = 0;  // swap of the first staging-ring-full event (the first stage frame)
  uint32_t trace_after_full_ = 60;
  uint32_t frame_sprite_draws_ = 0;  // draws of the paletted sprite pixel shader in this frame
  bool frame_big_tex_ = false;  // a stage-sized (4000+ wide) texture was fetched in this frame
  bool frame_mip_only_ = false, trace_on_mip_ = false;  // a texture without its base level was fetched in this frame
  bool traced_after_ = false;
  std::string debug_dir_;
  std::vector<std::string> trace_lines_;
  std::string shader_dir_;
  std::vector<uint32_t> page_serial_ = std::vector<uint32_t>(0x20000);  // per 4 KB physical page: the draw serial of its last write
  std::vector<uint8_t> page_seen_ = std::vector<uint8_t>(0x20000);  // in frame_pages_ already
  std::vector<uint32_t> frame_pages_, last_frame_pages_;            // pages seen dirty in this / the previous frame
  uint8_t recent_coarse_[0x200] = {};                               // 1 MB regions seen dirty lately (frames left)
  uint32_t frame_start_serial_ = 0;
  uint32_t serial_ = 1;  // the draw serial (SyncDirtyPages)
  uint64_t dirty_pages_ = 0, dirty_pages_at_swap_ = 0;
  std::unordered_map<uint64_t, DkShaderObj*> shaders_;
  std::unordered_map<uint64_t, HostImage*> targets_;       // EDRAM base/pitch -> render target
  std::unordered_map<uint32_t, HostImage*> guest_images_;  // guest address -> texture / resolve destination
  HostImage* bound_colors_[4] = {};
  std::unordered_map<uint64_t, HostImage*> dummy_targets_;
  HostImage* bound_depth_ = nullptr;
  std::unordered_set<std::string> logged_;
  std::unordered_set<uint64_t> seen_;
  uint32_t swaps_ = 0, draws_ = 0, skipped_ = 0, missing_draws_ = 0, missing_shaders_ = 0, clears_ = 0, resolves_ = 0;
  std::chrono::steady_clock::time_point hud_window_start_{};  // the fps counter's one-second window
  uint32_t hud_window_swaps_ = 0;
  double t_draw_ms_ = 0, t_acquire_ms_ = 0, t_fence_ms_ = 0;  // see IssueDraw
  double t_phase_ms_[7] = {};  // IssueDrawImpl phases
  double t_gpu_ms_ = 0;  // GPU time between the frame's timestamps
  uint32_t gpu_frames_ = 0;
  uint32_t uni_uploads_[3] = {};  // BindUniforms copies per second: vs, ps, sys
  uint32_t texbind_hits_ = 0;  // texture bindings served by the frame's cache
  bool vs_consts_dirty_ = true, ps_consts_dirty_ = true;  // WriteRegister
  char hud_text_[9] = "";  // what the present pass draws top-left (fps, one decimal)
  uint32_t image_count_ = 0, uploads_ = 0, reuploads_ = 0;
  uint64_t upload_bytes_ = 0;
  uint32_t frame_draws_ = 0, frame_clears_ = 0, frame_resolves_ = 0, frame_skipped_ = 0, frame_missing_ = 0, frame_reuploads_ = 0;
  struct Uploaded { uint32_t base, bytes, serial; };  // by value: a recycled image may change hands within the frame
  std::vector<Uploaded> frame_uploaded_;  // uploads of this frame (cleared at the swap)
  uint32_t late_writes_ = 0;               // per second: uploaded textures written again before their frame's swap
};

class Deko3dGraphicsSystem : public GraphicsSystem {
 public:
  std::string name() const override { return "deko3d"; }
  // The backend owns the default window's swapchain; the SDK keeps no presenter.
  X_STATUS SetupPresentation(::rex::ui::WindowedAppContext*) override { return X_STATUS_SUCCESS; }

 protected:
  void CreateProvider(bool) override {}
  std::unique_ptr<CommandProcessor> CreateCommandProcessor() override {
    return std::make_unique<Deko3dCommandProcessor>(this, kernel_state_);
  }
};

}  // namespace

rex::system::IGraphicsSystem* CreateDeko3dGraphicsSystem() { return new Deko3dGraphicsSystem(); }

}  // namespace rex::graphics::deko
#endif  // __SWITCH__ || TMNT_DK_GL

#version 460
// The XDK depth restore, on the host. The guest draws its saved depth texture (fetched as k_8_8_8_8: the raw
// bytes) with pixel shader 894498C723463852 into a colour target at the depth buffer's EDRAM base. The host keeps
// that texture as an R32F depth resolve image, so this shader replaces the guest's: the same texel address
// (copied from the translated guest shader), the value written as depth.
layout(location = 0) in vec4 i_0;
layout(binding = 0) uniform sampler2D tf_0;
layout(std140, binding = 0) uniform Consts { vec4 c[256]; };
void main() {
  float x = fract(i_0.z) >= c[255].z ? i_0.x + c[255].x : i_0.x + c[255].y;
  gl_FragDepth = texture(tf_0, vec2(x, i_0.y) / vec2(textureSize(tf_0, 0))).r;
}

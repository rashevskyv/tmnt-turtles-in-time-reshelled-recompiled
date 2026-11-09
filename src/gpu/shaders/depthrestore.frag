#version 460
// The XDK depth restore, on the host. The guest draws its saved depth texture (fetched as k_8_8_8_8: the raw
// bytes) with pixel shader 894498C723463852 into a colour target at the depth buffer's EDRAM base. The host keeps
// that texture as an R32F depth resolve image, so this shader replaces the guest's and writes the value as depth.
// The guest shader reads texel x + c[255].x or x + c[255].y by fract(i_0.z): it swaps the two 40-sample halves of
// every 80-sample EDRAM tile, because a depth buffer stores them in the other order than a colour target does.
// The host depth image has no such layout: pixel x takes texel x. The interpolated address is a whole number or
// a whole number + 0.5 at a pixel centre (the guest's half-pixel rule): both give the same texel here.
layout(location = 0) in vec4 i_0;
layout(binding = 0) uniform sampler2D tf_0;
void main() {
  gl_FragDepth = texture(tf_0, (floor(i_0.xy + 0.25) + 0.5) / vec2(textureSize(tf_0, 0))).r;
}

#version 460
// Depth resolve pass: writes a depth target's values into the red channel of an R32F guest texture (the game
// samples the scene depth as k_24_8_FLOAT). Drawn with present.vert; the viewport spans the whole source image,
// so uv times the source size is the source texel.
layout(binding = 0) uniform sampler2D depth_tex;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
void main() {
  color = vec4(texelFetch(depth_tex, ivec2(uv * vec2(textureSize(depth_tex, 0))), 0).r);
}

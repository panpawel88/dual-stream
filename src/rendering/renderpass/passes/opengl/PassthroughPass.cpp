#include "PassthroughPass.h"
#include "core/Logger.h"

OpenGLPassthroughPass::OpenGLPassthroughPass() : OpenGLSimpleRenderPass("Passthrough") {
}

std::string OpenGLPassthroughPass::GetFragmentShaderSource() const {
    return R"(
#version 460 core

in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D videoTexture;
uniform bool isYUV;

void main()
{
    // Simple passthrough - copy input to output
    FragColor = texture(videoTexture, TexCoord);
}
)";
}
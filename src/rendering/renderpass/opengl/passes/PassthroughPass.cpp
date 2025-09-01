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
uniform bool flipY;

void main()
{
    // Handle Y-coordinate flipping if needed
    vec2 texCoord = TexCoord;
    if (flipY) {
        texCoord.y = 1.0 - texCoord.y;
    }
    // Simple passthrough - copy input to output
    FragColor = texture(videoTexture, texCoord);
}
)";
}
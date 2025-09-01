#include "YUVToRGBRenderPass.h"
#include "core/Logger.h"

OpenGLYUVToRGBRenderPass::OpenGLYUVToRGBRenderPass() : OpenGLSimpleRenderPass("YUVToRGB") {
}

std::string OpenGLYUVToRGBRenderPass::GetFragmentShaderSource() const {
    return R"(
#version 460 core

in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D videoTexture; // Y plane or RGB texture
uniform sampler2D uvTexture;    // UV plane for NV12 format
uniform bool isYUV;
uniform bool flipY;

void main()
{
    // Handle Y-coordinate flipping if needed
    vec2 texCoord = TexCoord;
    if (flipY) {
        texCoord.y = 1.0 - texCoord.y;
    }
    
    if (isYUV) {
        // YUV to RGB conversion
        // Sample Y component
        float y = texture(videoTexture, texCoord).r;
        
        // Sample UV components (for NV12 format)
        vec2 uv = texture(uvTexture, texCoord).rg;
        float u = uv.r;
        float v = uv.g;
        
        // Convert from [0,1] range to proper YUV range
        y = (y - 0.0625) * 1.164; // (y - 16/255) * 255/(235-16)
        u = u - 0.5; // (u - 128/255)
        v = v - 0.5; // (v - 128/255)
        
        // YUV to RGB conversion matrix (Rec. 709)
        float r = y + 1.793 * v;
        float g = y - 0.213 * u - 0.533 * v;
        float b = y + 2.112 * u;
        
        // Clamp to [0,1] range
        FragColor = vec4(clamp(r, 0.0, 1.0), clamp(g, 0.0, 1.0), clamp(b, 0.0, 1.0), 1.0);
    } else {
        // Direct RGB passthrough
        FragColor = texture(videoTexture, texCoord);
    }
}
)";
}
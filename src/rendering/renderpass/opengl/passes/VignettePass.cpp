#include "VignettePass.h"
#include "core/Logger.h"
#include <cstring>

OpenGLVignettePass::OpenGLVignettePass() : OpenGLSimpleRenderPass("Vignette") {
    m_intensity = 0.5f;
    m_radius = 0.6f;
    m_feather = 0.4f;
    m_centerX = 0.0f;
    m_centerY = 0.0f;
}

void OpenGLVignettePass::UpdateParameters(const std::map<std::string, RenderPassParameter>& parameters) {
    for (const auto& [name, value] : parameters) {
        if (name == "intensity" && std::holds_alternative<float>(value)) {
            m_intensity = std::get<float>(value);
        } else if (name == "radius" && std::holds_alternative<float>(value)) {
            m_radius = std::get<float>(value);
        } else if (name == "feather" && std::holds_alternative<float>(value)) {
            m_feather = std::get<float>(value);
        } else if (name == "center_x" && std::holds_alternative<float>(value)) {
            m_centerX = std::get<float>(value);
        } else if (name == "center_y" && std::holds_alternative<float>(value)) {
            m_centerY = std::get<float>(value);
        }
    }
    
    // Call base class to handle parameter update notification
    OpenGLSimpleRenderPass::UpdateParameters(parameters);
}

std::string OpenGLVignettePass::GetFragmentShaderSource() const {
    return R"(
#version 460 core

in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D videoTexture;
uniform bool isYUV;
uniform bool flipY;

// Vignette parameters
layout(std140, binding = 0) uniform VignetteData {
    float intensity;
    float radius;
    float feather;
    float centerX;
    float centerY;
    float aspectRatio;
};

void main()
{
    // Handle Y-coordinate flipping if needed
    vec2 texCoord = TexCoord;
    if (flipY) {
        texCoord.y = 1.0 - texCoord.y;
    }
    
    // Sample the input texture
    vec4 color = texture(videoTexture, texCoord);
    
    // Calculate distance from center with aspect ratio correction
    vec2 center = vec2(0.5 + centerX * 0.5, 0.5 + centerY * 0.5);
    vec2 uv = texCoord - center;
    uv.x *= aspectRatio; // Correct for aspect ratio
    float dist = length(uv);
    
    // Calculate vignette mask
    float vignette = 1.0 - smoothstep(radius, radius + feather, dist);
    vignette = mix(1.0 - intensity, 1.0, vignette);
    
    // Apply vignette
    FragColor = vec4(color.rgb * vignette, color.a);
}
)";
}

size_t OpenGLVignettePass::GetUniformBufferSize() const {
    return sizeof(UniformBufferData);
}

void OpenGLVignettePass::PackUniformBuffer(uint8_t* buffer, const OpenGLRenderPassContext& context) {
    UniformBufferData* data = reinterpret_cast<UniformBufferData*>(buffer);
    data->intensity = m_intensity;
    data->radius = m_radius;
    data->feather = m_feather;
    data->centerX = m_centerX;
    data->centerY = m_centerY;
    data->aspectRatio = static_cast<float>(context.inputWidth) / static_cast<float>(context.inputHeight);
    data->padding[0] = 0.0f;
    data->padding[1] = 0.0f;
}
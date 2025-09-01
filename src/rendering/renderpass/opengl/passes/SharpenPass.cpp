#include "SharpenPass.h"
#include "core/Logger.h"
#include <cstring>

OpenGLSharpenPass::OpenGLSharpenPass() : OpenGLSimpleRenderPass("Sharpen") {
    m_strength = 1.0f;
    m_threshold = 0.1f;
}

void OpenGLSharpenPass::UpdateParameters(const std::map<std::string, RenderPassParameter>& parameters) {
    for (const auto& [name, value] : parameters) {
        if (name == "strength" && std::holds_alternative<float>(value)) {
            m_strength = std::get<float>(value);
        } else if (name == "threshold" && std::holds_alternative<float>(value)) {
            m_threshold = std::get<float>(value);
        }
    }
    
    // Call base class to handle parameter update notification
    OpenGLSimpleRenderPass::UpdateParameters(parameters);
}

std::string OpenGLSharpenPass::GetFragmentShaderSource() const {
    return R"(
#version 460 core

in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D videoTexture;
uniform bool isYUV;
uniform bool flipY;

// Sharpen parameters
layout(std140, binding = 0) uniform SharpenData {
    float strength;
    float threshold;
    float texelSizeX;
    float texelSizeY;
};

void main()
{
    // Handle Y-coordinate flipping if needed
    vec2 texCoord = TexCoord;
    if (flipY) {
        texCoord.y = 1.0 - texCoord.y;
    }
    
    // Sample center pixel
    vec4 center = texture(videoTexture, texCoord);
    
    // Sample neighboring pixels for edge detection
    vec4 top    = texture(videoTexture, texCoord + vec2(0.0, -texelSizeY));
    vec4 bottom = texture(videoTexture, texCoord + vec2(0.0, texelSizeY));
    vec4 left   = texture(videoTexture, texCoord + vec2(-texelSizeX, 0.0));
    vec4 right  = texture(videoTexture, texCoord + vec2(texelSizeX, 0.0));
    
    // Calculate edge enhancement using unsharp masking
    vec4 edges = center * 5.0 - (top + bottom + left + right);
    
    // Calculate luminance for threshold check
    float centerLuma = dot(center.rgb, vec3(0.299, 0.587, 0.114));
    float edgeLuma = dot(edges.rgb, vec3(0.299, 0.587, 0.114));
    
    // Apply threshold to avoid noise amplification
    float sharpenAmount = (abs(edgeLuma) > threshold) ? strength : 0.0;
    
    // Apply sharpening
    vec4 sharpened = center + edges * sharpenAmount;
    
    // Preserve alpha channel
    FragColor = vec4(sharpened.rgb, center.a);
}
)";
}

size_t OpenGLSharpenPass::GetUniformBufferSize() const {
    return sizeof(UniformBufferData);
}

void OpenGLSharpenPass::PackUniformBuffer(uint8_t* buffer, const OpenGLRenderPassContext& context) {
    UniformBufferData* data = reinterpret_cast<UniformBufferData*>(buffer);
    data->strength = m_strength;
    data->threshold = m_threshold;
    data->texelSizeX = 1.0f / static_cast<float>(context.inputWidth);
    data->texelSizeY = 1.0f / static_cast<float>(context.inputHeight);
    data->padding[0] = 0.0f;
    data->padding[1] = 0.0f;
    data->padding[2] = 0.0f;
    data->padding[3] = 0.0f;
}
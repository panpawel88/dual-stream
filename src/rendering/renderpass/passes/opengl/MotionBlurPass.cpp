#include "MotionBlurPass.h"
#include "core/Logger.h"
#include <cstring>

OpenGLMotionBlurPass::OpenGLMotionBlurPass() : OpenGLSimpleRenderPass("MotionBlur") {
    m_strength = 0.02f;  // Default matching D3D11
    m_samples = 8;
}

void OpenGLMotionBlurPass::UpdateParameters(const std::map<std::string, RenderPassParameter>& parameters) {
    for (const auto& [name, value] : parameters) {
        if (name == "blur_strength" && std::holds_alternative<float>(value)) {
            m_strength = std::get<float>(value);
        } else if (name == "sample_count" && std::holds_alternative<int>(value)) {
            m_samples = std::get<int>(value);
        }
    }
    
    // Call base class to handle parameter update notification
    OpenGLSimpleRenderPass::UpdateParameters(parameters);
}

std::string OpenGLMotionBlurPass::GetFragmentShaderSource() const {
    return R"(
#version 460 core

in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D videoTexture;
uniform bool isYUV;

// Motion blur parameters (matching D3D11 implementation)
layout(std140, binding = 0) uniform MotionBlurData {
    float blurStrength;
    int sampleCount;
    vec2 padding;
};

void main()
{
    vec4 result = vec4(0.0);
    
    // Use horizontal blur direction (matching D3D11)
    vec2 blurDirection = vec2(blurStrength * 0.01, 0.0);
    
    // Sample along the blur direction with proper offset distribution
    for (int i = 0; i < sampleCount; i++) {
        float offset = (float(i) / float(sampleCount - 1) - 0.5) * 2.0; // -1.0 to 1.0
        vec2 sampleUV = clamp(TexCoord + blurDirection * offset, vec2(0.0), vec2(1.0));
        result += texture(videoTexture, sampleUV);
    }
    
    FragColor = result / float(sampleCount);
}
)";
}

size_t OpenGLMotionBlurPass::GetUniformBufferSize() const {
    return sizeof(UniformBufferData);
}

void OpenGLMotionBlurPass::PackUniformBuffer(uint8_t* buffer, const OpenGLRenderPassContext& context) {
    UniformBufferData* data = reinterpret_cast<UniformBufferData*>(buffer);
    
    // Pack according to uniform buffer layout (matching D3D11)
    data->blurStrength = m_strength;
    data->sampleCount = m_samples;
    data->padding1 = 0.0f;
    data->padding2 = 0.0f;
}
#include "BloomPass.h"
#include "core/Logger.h"
#include <cstring>

OpenGLBloomPass::OpenGLBloomPass() : OpenGLSimpleRenderPass("Bloom") {
    m_threshold = 0.8f;
    m_intensity = 1.0f;
    m_radius = 1.5f;
    m_blendFactor = 0.3f;
}

void OpenGLBloomPass::UpdateParameters(const std::map<std::string, RenderPassParameter>& parameters) {
    for (const auto& [name, value] : parameters) {
        if (name == "threshold" && std::holds_alternative<float>(value)) {
            m_threshold = std::get<float>(value);
        } else if (name == "intensity" && std::holds_alternative<float>(value)) {
            m_intensity = std::get<float>(value);
        } else if (name == "radius" && std::holds_alternative<float>(value)) {
            m_radius = std::get<float>(value);
        } else if (name == "blend_factor" && std::holds_alternative<float>(value)) {
            m_blendFactor = std::get<float>(value);
        }
    }
    
    // Call base class to handle parameter update notification
    OpenGLSimpleRenderPass::UpdateParameters(parameters);
}

std::string OpenGLBloomPass::GetFragmentShaderSource() const {
    return R"(
#version 460 core

in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D videoTexture;
uniform bool isYUV;

// Bloom parameters
layout(std140, binding = 0) uniform BloomData {
    float threshold;
    float intensity;
    float radius;
    float blendFactor;
    float texelSizeX;
    float texelSizeY;
};

void main()
{
    vec2 texCoord = TexCoord;
    
    // Sample original pixel
    vec4 original = texture(videoTexture, texCoord);
    
    // Extract bright areas (simple threshold)
    vec4 bright = original;
    float brightness = dot(bright.rgb, vec3(0.299, 0.587, 0.114)); // Luminance
    if (brightness < threshold) {
        bright = vec4(0, 0, 0, 0);
    } else {
        bright.rgb = (bright.rgb - threshold) / (1.0 - threshold);
    }
    
    // Simple box blur for bloom effect
    vec4 bloom = vec4(0, 0, 0, 0);
    vec2 offset = vec2(texelSizeX, texelSizeY) * radius;
    int samples = 0;
    
    // 5x5 sampling kernel for bloom
    for (int x = -2; x <= 2; x++) {
        for (int y = -2; y <= 2; y++) {
            vec2 sampleCoord = texCoord + vec2(float(x), float(y)) * offset;
            vec4 sampleColor = texture(videoTexture, sampleCoord);
            
            // Extract bright areas from sample
            float sampleBrightness = dot(sampleColor.rgb, vec3(0.299, 0.587, 0.114));
            if (sampleBrightness >= threshold) {
                sampleColor.rgb = (sampleColor.rgb - threshold) / (1.0 - threshold);
                bloom += sampleColor * intensity;
                samples++;
            }
        }
    }
    
    if (samples > 0) {
        bloom /= float(samples);
    }
    
    // Blend original with bloom effect
    vec4 result = mix(original, original + bloom, blendFactor);
    result.a = original.a; // Preserve alpha
    
    FragColor = clamp(result, 0.0, 1.0);
}
)";
}

size_t OpenGLBloomPass::GetUniformBufferSize() const {
    return sizeof(UniformBufferData);
}

void OpenGLBloomPass::PackUniformBuffer(uint8_t* buffer, const OpenGLRenderPassContext& context) {
    UniformBufferData* data = reinterpret_cast<UniformBufferData*>(buffer);
    data->threshold = m_threshold;
    data->intensity = m_intensity;
    data->radius = m_radius;
    data->blendFactor = m_blendFactor;
    data->texelSizeX = 1.0f / static_cast<float>(context.inputWidth);
    data->texelSizeY = 1.0f / static_cast<float>(context.inputHeight);
    data->padding[0] = 0.0f;
    data->padding[1] = 0.0f;
}
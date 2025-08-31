#include "BloomPass.h"
#include "core/Logger.h"
#include <cstring>

OpenGLBloomPass::OpenGLBloomPass() : OpenGLSimpleRenderPass("Bloom") {
    m_threshold = 1.0f;
    m_intensity = 1.0f;
    m_blurSize = 2.0f;
}

void OpenGLBloomPass::UpdateParameters(const std::map<std::string, RenderPassParameter>& parameters) {
    for (const auto& [name, value] : parameters) {
        if (name == "threshold" && std::holds_alternative<float>(value)) {
            m_threshold = std::get<float>(value);
        } else if (name == "intensity" && std::holds_alternative<float>(value)) {
            m_intensity = std::get<float>(value);
        } else if (name == "blur_size" && std::holds_alternative<float>(value)) {
            m_blurSize = std::get<float>(value);
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
    float blurSize;
    float texelSizeX;
    float texelSizeY;
};

void main()
{
    // Sample the original pixel
    vec4 originalColor = texture(videoTexture, TexCoord);
    
    // Extract bright areas based on threshold
    float luminance = dot(originalColor.rgb, vec3(0.299, 0.587, 0.114));
    vec3 brightColor = originalColor.rgb * max(0.0, luminance - threshold);
    
    // Apply simple box blur to the bright areas
    vec3 blurredBright = vec3(0.0);
    int samples = int(blurSize * 2.0) + 1;
    float sampleCount = 0.0;
    
    for (int x = -int(blurSize); x <= int(blurSize); x++) {
        for (int y = -int(blurSize); y <= int(blurSize); y++) {
            vec2 offset = vec2(float(x) * texelSizeX, float(y) * texelSizeY);
            vec2 sampleCoord = TexCoord + offset;
            
            // Check bounds
            if (sampleCoord.x >= 0.0 && sampleCoord.x <= 1.0 && 
                sampleCoord.y >= 0.0 && sampleCoord.y <= 1.0) {
                vec4 sampleColor = texture(videoTexture, sampleCoord);
                float sampleLuminance = dot(sampleColor.rgb, vec3(0.299, 0.587, 0.114));
                vec3 sampleBright = sampleColor.rgb * max(0.0, sampleLuminance - threshold);
                blurredBright += sampleBright;
                sampleCount += 1.0;
            }
        }
    }
    
    // Average the blur samples
    if (sampleCount > 0.0) {
        blurredBright /= sampleCount;
    }
    
    // Combine original with bloom effect
    vec3 finalColor = originalColor.rgb + blurredBright * intensity;
    
    FragColor = vec4(finalColor, originalColor.a);
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
    data->blurSize = m_blurSize;
    data->texelSizeX = 1.0f / static_cast<float>(context.inputWidth);
    data->texelSizeY = 1.0f / static_cast<float>(context.inputHeight);
    data->padding[0] = 0.0f;
    data->padding[1] = 0.0f;
    data->padding[2] = 0.0f;
}
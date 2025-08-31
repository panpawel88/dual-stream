#include "MotionBlurPass.h"
#include "core/Logger.h"
#include <cstring>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

OpenGLMotionBlurPass::OpenGLMotionBlurPass() : OpenGLSimpleRenderPass("MotionBlur") {
    m_strength = 2.0f;
    m_angle = 0.0f;
    m_samples = 8;
}

void OpenGLMotionBlurPass::UpdateParameters(const std::map<std::string, RenderPassParameter>& parameters) {
    for (const auto& [name, value] : parameters) {
        if (name == "strength" && std::holds_alternative<float>(value)) {
            m_strength = std::get<float>(value);
        } else if (name == "angle" && std::holds_alternative<float>(value)) {
            m_angle = std::get<float>(value);
        } else if (name == "samples" && std::holds_alternative<int>(value)) {
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

// Motion blur parameters
layout(std140, binding = 0) uniform MotionBlurData {
    float strength;
    float angle;
    float directionX;
    float directionY;
    float texelSizeX;
    float texelSizeY;
    int samples;
};

void main()
{
    vec4 color = vec4(0.0);
    
    // Calculate blur direction vector
    vec2 blurDirection = vec2(directionX, directionY) * strength;
    blurDirection.x *= texelSizeX;
    blurDirection.y *= texelSizeY;
    
    // Sample along the motion vector
    for (int i = 0; i < samples; i++) {
        float t = (float(i) / float(samples - 1)) - 0.5; // -0.5 to 0.5
        vec2 sampleCoord = TexCoord + blurDirection * t;
        
        // Ensure we don't sample outside texture bounds
        if (sampleCoord.x >= 0.0 && sampleCoord.x <= 1.0 && 
            sampleCoord.y >= 0.0 && sampleCoord.y <= 1.0) {
            color += texture(videoTexture, sampleCoord);
        } else {
            // Use edge pixel for out-of-bounds samples
            vec2 clampedCoord = clamp(sampleCoord, vec2(0.0), vec2(1.0));
            color += texture(videoTexture, clampedCoord);
        }
    }
    
    // Average the samples
    FragColor = color / float(samples);
}
)";
}

size_t OpenGLMotionBlurPass::GetUniformBufferSize() const {
    return sizeof(UniformBufferData);
}

void OpenGLMotionBlurPass::PackUniformBuffer(uint8_t* buffer, const OpenGLRenderPassContext& context) {
    UniformBufferData* data = reinterpret_cast<UniformBufferData*>(buffer);
    
    // Convert angle to radians and calculate direction vector
    float angleRad = m_angle * static_cast<float>(M_PI) / 180.0f;
    
    data->strength = m_strength;
    data->angle = m_angle;
    data->directionX = cosf(angleRad);
    data->directionY = sinf(angleRad);
    data->texelSizeX = 1.0f / static_cast<float>(context.inputWidth);
    data->texelSizeY = 1.0f / static_cast<float>(context.inputHeight);
    data->samples = m_samples;
    data->padding = 0.0f;
}
#pragma once

#include "../OpenGLSimpleRenderPass.h"

/**
 * OpenGL motion blur render pass that applies horizontal blur effect.
 * Creates a sense of motion by blurring pixels horizontally (matching D3D11 implementation).
 * 
 * Supported parameters:
 * - blur_strength: Blur intensity (0.0 - 1.0, default 0.02)
 * - sample_count: Number of blur samples (1 - 32, default 8)
 */
class OpenGLMotionBlurPass : public OpenGLSimpleRenderPass {
public:
    OpenGLMotionBlurPass();
    virtual ~OpenGLMotionBlurPass() = default;

    void UpdateParameters(const std::map<std::string, RenderPassParameter>& parameters) override;

protected:
    // Override virtual methods from OpenGLSimpleRenderPass
    std::string GetFragmentShaderSource() const override;
    size_t GetUniformBufferSize() const override;
    void PackUniformBuffer(uint8_t* buffer, const OpenGLRenderPassContext& context) override;

private:
    struct UniformBufferData {
        float blurStrength;  // offset 0
        int sampleCount;     // offset 4
        float padding1;      // offset 8
        float padding2;      // offset 12 (total size 16 bytes)
    };
    
    float m_strength;
    int m_samples;
};
#pragma once

#include "../../opengl/OpenGLSimpleRenderPass.h"

/**
 * OpenGL bloom render pass that creates a glow effect around bright areas.
 * Enhances bright pixels to create a luminous, glowing appearance.
 * 
 * Supported parameters:
 * - threshold: Minimum brightness to trigger bloom (0.0 - 2.0, default 0.8)
 * - intensity: Bloom effect strength (0.0 - 5.0, default 1.0)
 * - radius: Size of the bloom blur effect (0.5 - 5.0, default 1.5)
 * - blend_factor: Bloom blend strength (0.0 - 1.0, default 0.3)
 */
class OpenGLBloomPass : public OpenGLSimpleRenderPass {
public:
    OpenGLBloomPass();
    virtual ~OpenGLBloomPass() = default;

    void UpdateParameters(const std::map<std::string, RenderPassParameter>& parameters) override;

protected:
    // Override virtual methods from OpenGLSimpleRenderPass
    std::string GetFragmentShaderSource() const override;
    size_t GetUniformBufferSize() const override;
    void PackUniformBuffer(uint8_t* buffer, const OpenGLRenderPassContext& context) override;

private:
    struct UniformBufferData {
        float threshold;
        float intensity;
        float radius;
        float blendFactor;
        float texelSizeX;
        float texelSizeY;
        float padding[2]; // Pad to 32-byte alignment for std140
    };
    
    float m_threshold;
    float m_intensity;
    float m_radius;
    float m_blendFactor;
};
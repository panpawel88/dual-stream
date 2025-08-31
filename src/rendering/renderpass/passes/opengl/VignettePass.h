#pragma once

#include "../../opengl/OpenGLSimpleRenderPass.h"

/**
 * OpenGL vignette render pass that darkens the edges/corners of the frame.
 * Creates a focus effect by gradually darkening towards the edges.
 * 
 * Supported parameters:
 * - intensity: Strength of the vignette darkening (0.0 - 1.0, default 0.5)
 * - radius: Inner radius where vignetting starts (0.0 - 1.0, default 0.6)
 * - feather: Softness of the vignette transition (0.1 - 0.8, default 0.4)
 * - center_x: Horizontal center offset (-1.0 - 1.0, default 0.0)
 * - center_y: Vertical center offset (-1.0 - 1.0, default 0.0)
 */
class OpenGLVignettePass : public OpenGLSimpleRenderPass {
public:
    OpenGLVignettePass();
    virtual ~OpenGLVignettePass() = default;

    void UpdateParameters(const std::map<std::string, RenderPassParameter>& parameters) override;

protected:
    // Override virtual methods from OpenGLSimpleRenderPass
    std::string GetFragmentShaderSource() const override;
    size_t GetUniformBufferSize() const override;
    void PackUniformBuffer(uint8_t* buffer, const OpenGLRenderPassContext& context) override;

private:
    struct UniformBufferData {
        float intensity;
        float radius;
        float feather;
        float centerX;
        float centerY;
        float aspectRatio;
        float padding[2]; // Pad to 16-byte alignment
    };
    
    float m_intensity;
    float m_radius;
    float m_feather;
    float m_centerX;
    float m_centerY;
};
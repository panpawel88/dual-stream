#pragma once

#include "../../opengl/OpenGLSimpleRenderPass.h"

/**
 * OpenGL motion blur render pass that applies directional blur effect.
 * Creates a sense of motion by blurring pixels in a specific direction.
 * 
 * Supported parameters:
 * - strength: Blur intensity (0.0 - 10.0, default 2.0)
 * - angle: Blur direction in degrees (0.0 - 360.0, default 0.0)
 * - samples: Number of blur samples (4 - 16, default 8)
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
        float strength;
        float angle;
        float directionX;
        float directionY;
        float texelSizeX;
        float texelSizeY;
        int samples;
        float padding;
    };
    
    float m_strength;
    float m_angle;
    int m_samples;
};
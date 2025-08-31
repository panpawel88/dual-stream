#pragma once

#include "../../opengl/OpenGLSimpleRenderPass.h"

/**
 * OpenGL sharpen render pass that enhances edges and details in the image.
 * Uses unsharp masking technique for edge enhancement.
 * 
 * Supported parameters:
 * - strength: Sharpening intensity (0.0 - 2.0, default 1.0)
 * - threshold: Minimum difference threshold to apply sharpening (0.0 - 1.0, default 0.1)
 */
class OpenGLSharpenPass : public OpenGLSimpleRenderPass {
public:
    OpenGLSharpenPass();
    virtual ~OpenGLSharpenPass() = default;

    void UpdateParameters(const std::map<std::string, RenderPassParameter>& parameters) override;

protected:
    // Override virtual methods from OpenGLSimpleRenderPass
    std::string GetFragmentShaderSource() const override;
    size_t GetUniformBufferSize() const override;
    void PackUniformBuffer(uint8_t* buffer, const OpenGLRenderPassContext& context) override;

private:
    struct UniformBufferData {
        float strength;
        float threshold;
        float texelSizeX;
        float texelSizeY;
        float padding[4]; // Pad to 16-byte alignment
    };
    
    float m_strength;
    float m_threshold;
};
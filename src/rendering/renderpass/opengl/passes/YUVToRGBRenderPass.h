#pragma once

#include "../OpenGLSimpleRenderPass.h"

/**
 * OpenGL YUV to RGB conversion render pass
 * Converts YUV color space textures to RGB for display
 * Supports NV12 and YUV420P formats
 */
class OpenGLYUVToRGBRenderPass : public OpenGLSimpleRenderPass {
public:
    OpenGLYUVToRGBRenderPass();
    virtual ~OpenGLYUVToRGBRenderPass() = default;

protected:
    // Override virtual methods from OpenGLSimpleRenderPass
    std::string GetFragmentShaderSource() const override;
};
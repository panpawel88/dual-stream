#pragma once

#include "../../opengl/OpenGLSimpleRenderPass.h"

/**
 * OpenGL passthrough render pass - simply copies input to output
 * Used for testing the render pass pipeline without any effects
 */
class OpenGLPassthroughPass : public OpenGLSimpleRenderPass {
public:
    OpenGLPassthroughPass();
    virtual ~OpenGLPassthroughPass() = default;

protected:
    // Override virtual methods from OpenGLSimpleRenderPass
    std::string GetFragmentShaderSource() const override;
};
#pragma once

#include "../IRenderPass.h"
#include "OpenGLRenderPassContext.h"
#include <glad/gl.h>

// Forward declarations
class RenderPassConfig;

/**
 * OpenGL specific render pass base class
 * Extends the API-agnostic interface with OpenGL functionality
 */
class OpenGLRenderPass : public IRenderPass {
public:
    OpenGLRenderPass(const std::string& name) : IRenderPass(name) {}
    virtual ~OpenGLRenderPass() = default;

    /**
     * Initialize the render pass with OpenGL context
     * @param config Configuration for the pass
     * @return true on success
     */
    virtual bool Initialize(const RenderPassConfig& config) = 0;
    
    /**
     * Execute the render pass
     * @param context OpenGL rendering context with timing info
     * @param inputTexture Input texture to process (GL texture ID)
     * @param outputFramebuffer Output framebuffer (0 for default framebuffer)
     * @param outputTexture Output texture (GL texture ID, used if outputFramebuffer != 0)
     * @return true on success
     */
    virtual bool Execute(const OpenGLRenderPassContext& context,
                        GLuint inputTexture,
                        GLuint outputFramebuffer,
                        GLuint outputTexture = 0) = 0;
};
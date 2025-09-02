#pragma once

#include "../../OpenGLHeaders.h"

/**
 * Shared OpenGL resources for render passes
 * Manages common resources like fullscreen quad geometry, samplers, and render states
 * Implements singleton pattern to ensure resources are shared across all passes
 */
class OpenGLRenderPassResources {
public:
    static OpenGLRenderPassResources* GetInstance();
    static void DestroyInstance();
    
    /**
     * Initialize shared resources
     * @return true on success
     */
    bool Initialize();
    
    /**
     * Cleanup all shared resources
     */
    void Cleanup();
    
    /**
     * Render a fullscreen quad using the shared VAO
     */
    void RenderFullscreenQuad();
    
    /**
     * Get shared sampler for linear filtering
     */
    GLuint GetLinearSampler() const { return m_linearSampler; }
    
    /**
     * Get shared sampler for point filtering
     */
    GLuint GetPointSampler() const { return m_pointSampler; }

private:
    OpenGLRenderPassResources();
    ~OpenGLRenderPassResources();
    
    bool CreateFullscreenQuad();
    bool CreateSamplers();
    void SetupRenderStates();

private:
    static OpenGLRenderPassResources* s_instance;
    
    // Fullscreen quad resources
    GLuint m_vao;
    GLuint m_vbo;
    GLuint m_ebo;
    
    // Shared samplers
    GLuint m_linearSampler;
    GLuint m_pointSampler;
    
    bool m_initialized;
};
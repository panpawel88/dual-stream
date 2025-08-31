#pragma once

#include "../IRenderPassPipeline.h"
#include "OpenGLRenderPass.h"
#include <vector>
#include <memory>
#include <string>

// Forward declaration
class RenderPassConfig;

/**
 * OpenGL specific render pass pipeline implementation
 * Manages a pipeline of OpenGL render passes, handling framebuffer allocation and pass chaining
 */
class OpenGLRenderPassPipeline : public IRenderPassPipeline {
public:
    OpenGLRenderPassPipeline();
    ~OpenGLRenderPassPipeline();

    /**
     * Initialize the pipeline
     * @return true on success
     */
    bool Initialize();
    
    // IRenderPassPipeline interface
    void Cleanup() override;
    void AddPass(std::unique_ptr<IRenderPass> pass) override;
    void SetEnabled(bool enabled) override { m_enabled = enabled; }
    bool IsEnabled() const override { return m_enabled; }
    bool SetPassEnabled(const std::string& passName, bool enabled) override;
    IRenderPass* GetPass(const std::string& passName) const override;
    size_t GetPassCount() const override { return m_passes.size(); }
    bool UpdatePassParameters(const std::string& passName, 
                             const std::map<std::string, RenderPassParameter>& parameters) override;
    
    // OpenGL-specific methods
    void AddOpenGLPass(std::unique_ptr<OpenGLRenderPass> pass);
    
    /**
     * Execute the entire pipeline
     * @param context OpenGL rendering context
     * @param inputTexture Input texture to process
     * @param outputFramebuffer Final output framebuffer (0 for default framebuffer)
     * @param outputTexture Final output texture (if outputFramebuffer != 0)
     * @return true on success
     */
    bool Execute(const OpenGLRenderPassContext& context,
                GLuint inputTexture,
                GLuint outputFramebuffer,
                GLuint outputTexture = 0);

private:
    /**
     * Ensure intermediate framebuffers are allocated and sized correctly
     */
    bool EnsureIntermediateFramebuffers(int width, int height);
    
    /**
     * Create intermediate framebuffer with texture
     */
    bool CreateIntermediateFramebuffer(int width, int height, GLenum format,
                                      GLuint& framebuffer, GLuint& texture);
    
    /**
     * Perform direct copy from input to output (when pipeline is disabled)
     */
    bool DirectCopy(GLuint inputTexture, GLuint outputFramebuffer, GLuint outputTexture,
                   int width, int height);
    
    /**
     * Create resources for direct copy
     */
    bool CreateCopyResources();

private:
    std::vector<std::unique_ptr<OpenGLRenderPass>> m_passes;
    bool m_enabled;
    
    // Cached YUV conversion pass for dynamic insertion
    std::unique_ptr<OpenGLRenderPass> m_yuvToRgbPass;
    
    // Intermediate framebuffers for pass chaining (ping-pong buffers)
    GLuint m_intermediateFramebuffer[2];
    GLuint m_intermediateTexture[2];
    
    // Current texture dimensions
    int m_textureWidth;
    int m_textureHeight;
    GLenum m_textureFormat;
    
    // Resources for direct copy
    GLuint m_copyProgram;
    GLuint m_copyVertexShader;
    GLuint m_copyFragmentShader;
    
    bool m_initialized;
};
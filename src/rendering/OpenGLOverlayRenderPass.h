#pragma once
#include "OverlayRenderPass.h"
#include "renderpass/opengl/OpenGLRenderPass.h"
#include "renderpass/opengl/OpenGLRenderPassContext.h"
#include "OpenGLHeaders.h"

class OpenGLOverlayRenderPass : public OverlayRenderPass, public OpenGLRenderPass {
public:
    OpenGLOverlayRenderPass();
    ~OpenGLOverlayRenderPass() override;
    
    // OpenGLRenderPass interface - OpenGL specific
    bool Initialize(const RenderPassConfig& config) override;
    bool Execute(const OpenGLRenderPassContext& context,
                GLuint inputTexture,
                GLuint outputFramebuffer,
                GLuint outputTexture = 0) override;
    void Cleanup() override;
    
    // IRenderPass interface
    PassType GetType() const override { return PassType::Simple; }
    void UpdateParameters(const std::map<std::string, RenderPassParameter>& parameters) override;
    
protected:
    // OverlayRenderPass abstract methods - OpenGL implementations
    bool InitializeImGuiBackend() override;
    void CleanupImGuiBackend() override;
    void BeginImGuiFrame() override;
    void EndImGuiFrame() override;
    
private:
    // Passthrough rendering resources
    GLuint m_shaderProgram = 0;
    GLuint m_vao = 0;
    GLuint m_vbo = 0;
    
    bool InitializePassthroughShader();
    void CleanupPassthroughShader();
};
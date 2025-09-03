#pragma once
#include "../../OverlayRenderPass.h"
#include "../OpenGLSimpleRenderPass.h"
#include "../OpenGLRenderPassContext.h"
#include "../../../OpenGLHeaders.h"

class OpenGLOverlayRenderPass : public OverlayRenderPass, public OpenGLSimpleRenderPass {
public:
    OpenGLOverlayRenderPass();
    ~OpenGLOverlayRenderPass() override;
    
    // OpenGLSimpleRenderPass interface
    bool Initialize(const RenderPassConfig& config) override;
    bool InitializeWithHWND(const RenderPassConfig& config, void* hwnd);
    bool Execute(const OpenGLRenderPassContext& context,
                GLuint inputTexture,
                GLuint outputFramebuffer,
                GLuint outputTexture = 0) override;
    void UpdateParameters(const std::map<std::string, RenderPassParameter>& parameters) override;
    
    // Shader source for passthrough functionality
    std::string GetFragmentShaderSource() const override;
    
    // IRenderPass interface
    PassType GetType() const override { return PassType::Simple; }
    
protected:
    // OverlayRenderPass abstract methods - OpenGL implementations
    bool InitializeImGuiBackend() override;
    void CleanupImGuiBackend() override;
    void BeginImGuiFrame() override;
    void EndImGuiFrame() override;
    
private:
    // No additional private members needed - OpenGLSimpleRenderPass handles shaders
};
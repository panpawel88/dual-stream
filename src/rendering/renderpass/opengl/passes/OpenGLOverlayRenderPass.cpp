#include "OpenGLOverlayRenderPass.h"
#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "../../../../ui/ImGuiManager.h"
#include "../../../../ui/UIRegistry.h"
#include "../../../../ui/NotificationManager.h"
#include "../../../../core/Logger.h"

OpenGLOverlayRenderPass::OpenGLOverlayRenderPass() 
    : OverlayRenderPass(), OpenGLRenderPass("Overlay") {
}

OpenGLOverlayRenderPass::~OpenGLOverlayRenderPass() = default;

bool OpenGLOverlayRenderPass::Initialize(const RenderPassConfig& config) {
    // Get window size from config (we'll need to handle this properly)
    int width = 1920; // TODO: Get from config
    int height = 1080; // TODO: Get from config
    
    // Call base class common initialization
    if (!InitializeCommon(width, height, nullptr)) {
        return false;
    }
    
    // Initialize passthrough shader
    if (!InitializePassthroughShader()) {
        Logger::GetInstance().Error("Failed to initialize passthrough shader for overlay");
        return false;
    }
    
    return true;
}

bool OpenGLOverlayRenderPass::InitializeImGuiBackend() {
    const char* glsl_version = "#version 460";
    return ImGui_ImplOpenGL3_Init(glsl_version);
}

void OpenGLOverlayRenderPass::BeginImGuiFrame() {
    ImGui_ImplOpenGL3_NewFrame();
}

void OpenGLOverlayRenderPass::EndImGuiFrame() {
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

bool OpenGLOverlayRenderPass::Execute(const OpenGLRenderPassContext& context,
                                     GLuint inputTexture,
                                     GLuint outputFramebuffer,
                                     GLuint outputTexture) {
    if (!m_initialized) {
        return false;
    }
    
    // Bind output framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, outputFramebuffer);
    glViewport(0, 0, context.outputWidth, context.outputHeight);
    
    // First, copy input texture to output (passthrough)
    // TODO: Implement passthrough rendering using m_shaderProgram
    
    // Then render ImGui overlay (visibility handled in RenderImGuiContent)
    RenderImGuiContent();
    
    return true;
}

void OpenGLOverlayRenderPass::CleanupImGuiBackend() {
    ImGui_ImplOpenGL3_Shutdown();
}

bool OpenGLOverlayRenderPass::InitializePassthroughShader() {
    // TODO: Implement passthrough shader creation
    // Similar to existing OpenGL render passes
    return true;
}

void OpenGLOverlayRenderPass::CleanupPassthroughShader() {
    if (m_shaderProgram) {
        glDeleteProgram(m_shaderProgram);
        m_shaderProgram = 0;
    }
    if (m_vao) {
        glDeleteVertexArrays(1, &m_vao);
        m_vao = 0;
    }
    if (m_vbo) {
        glDeleteBuffers(1, &m_vbo);
        m_vbo = 0;
    }
}

void OpenGLOverlayRenderPass::Cleanup() {
    OverlayRenderPass::Cleanup();
    CleanupPassthroughShader();
}

void OpenGLOverlayRenderPass::UpdateParameters(const std::map<std::string, RenderPassParameter>& parameters) {
    // Handle overlay-specific parameters
    for (const auto& param : parameters) {
        const std::string& name = param.first;
        const RenderPassParameter& value = param.second;
        
        if (name == "show_ui_registry") {
            if (std::holds_alternative<bool>(value)) {
                SetUIRegistryVisible(std::get<bool>(value));
                LOG_INFO("Overlay: UI Registry visibility set to ", std::get<bool>(value));
            }
        } else if (name == "show_notifications") {
            if (std::holds_alternative<bool>(value)) {
                SetNotificationsVisible(std::get<bool>(value));
                LOG_INFO("Overlay: Notifications visibility set to ", std::get<bool>(value));
            }
        }
    }
}
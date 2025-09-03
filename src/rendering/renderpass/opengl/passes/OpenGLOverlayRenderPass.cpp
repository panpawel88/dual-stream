#include "OpenGLOverlayRenderPass.h"
#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "../../../../ui/ImGuiManager.h"
#include "../../../../ui/UIRegistry.h"
#include "../../../../ui/NotificationManager.h"
#include "../../../../core/Logger.h"

OpenGLOverlayRenderPass::OpenGLOverlayRenderPass() 
    : OverlayRenderPass(), OpenGLSimpleRenderPass("Overlay") {
}

OpenGLOverlayRenderPass::~OpenGLOverlayRenderPass() = default;

bool OpenGLOverlayRenderPass::Initialize(const RenderPassConfig& config) {
    return InitializeWithHWND(config, nullptr);
}

bool OpenGLOverlayRenderPass::InitializeWithHWND(const RenderPassConfig& config, void* hwnd) {
    // First call the parent class initialization to set up shaders
    if (!OpenGLSimpleRenderPass::Initialize(config)) {
        Logger::GetInstance().Error("Failed to initialize OpenGLSimpleRenderPass for overlay");
        return false;
    }
    
    // Get window size from config
    int width = 1920; // TODO: Get from config
    int height = 1080; // TODO: Get from config
    
    // Then call the overlay common initialization with hwnd
    if (!InitializeCommon(width, height, hwnd)) {
        Logger::GetInstance().Error("Failed to initialize OverlayRenderPass common for overlay");
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

std::string OpenGLOverlayRenderPass::GetFragmentShaderSource() const {
    return R"(
#version 460 core

in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D videoTexture;
uniform bool isYUV;
uniform bool flipY;

void main()
{
    // Handle Y-coordinate flipping if needed
    vec2 texCoord = TexCoord;
    if (flipY) {
        texCoord.y = 1.0 - texCoord.y;
    }
    // Simple passthrough - copy input to output
    FragColor = texture(videoTexture, texCoord);
}
)";
}

bool OpenGLOverlayRenderPass::Execute(const OpenGLRenderPassContext& context,
                                     GLuint inputTexture,
                                     GLuint outputFramebuffer,
                                     GLuint outputTexture) {
    if (!m_initialized) {
        return false;
    }
    
    // First, call parent class Execute to render the passthrough (video content)
    if (!OpenGLSimpleRenderPass::Execute(context, inputTexture, outputFramebuffer, outputTexture)) {
        Logger::GetInstance().Error("OpenGLOverlayRenderPass: Failed to execute passthrough rendering");
        return false;
    }
    
    // Then render ImGui overlay on top with proper blending
    // The framebuffer is already bound by parent class
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    RenderImGuiContent();
    
    return true;
}

void OpenGLOverlayRenderPass::CleanupImGuiBackend() {
    ImGui_ImplOpenGL3_Shutdown();
}

void OpenGLOverlayRenderPass::UpdateParameters(const std::map<std::string, RenderPassParameter>& parameters) {
    // First call parent class to handle standard render pass parameters
    OpenGLSimpleRenderPass::UpdateParameters(parameters);
    
    // Then handle overlay-specific parameters
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
#include "OverlayRenderPass.h"
#include "../../ui/ImGuiManager.h"
#include "../../ui/UIRegistry.h"
#include "../../ui/NotificationManager.h"
#include "../../core/Logger.h"

OverlayRenderPass::OverlayRenderPass()
    : IRenderPass("Overlay")
{
}

OverlayRenderPass::~OverlayRenderPass() = default;

bool OverlayRenderPass::InitializeCommon(int width, int height) {
    m_width = width;
    m_height = height;
    
    // Initialize ImGui if not already done
    ImGuiManager& imgui = ImGuiManager::GetInstance();
    if (!imgui.Initialize()) {
        Logger::GetInstance().Error("Failed to initialize ImGui for overlay");
        return false;
    }
    
    // Initialize backend-specific ImGui (implemented by derived classes)
    if (!InitializeImGuiBackend()) {
        Logger::GetInstance().Error("Failed to initialize ImGui backend for overlay");
        return false;
    }
    
    m_initialized = true;
    Logger::GetInstance().Info("Overlay render pass initialized");
    return true;
}

void OverlayRenderPass::RenderImGuiContent() {
    if (!m_initialized || !m_visible) {
        return;
    }
    
    // Begin ImGui frame (backend-specific, implemented by derived classes)
    BeginImGuiFrame();
    
    // Start the ImGui frame
    ImGuiManager::GetInstance().NewFrame();
    
    // Render UI registry content
    UIRegistry::GetInstance().RenderUI();
    
    // Render notifications
    NotificationManager::GetInstance().Update();
    
    // End the ImGui frame
    ImGuiManager::GetInstance().Render();
    
    // End ImGui frame (backend-specific, implemented by derived classes)
    EndImGuiFrame();
}

void OverlayRenderPass::Cleanup() {
    if (m_initialized) {
        CleanupImGuiBackend();
        m_initialized = false;
    }
}
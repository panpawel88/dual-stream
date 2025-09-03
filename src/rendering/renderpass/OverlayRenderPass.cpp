#include "OverlayRenderPass.h"
#include "../../ui/ImGuiManager.h"
#include "../../ui/UIRegistry.h"
#include "../../ui/NotificationManager.h"
#include "../../core/Logger.h"
#include "imgui.h"

OverlayRenderPass::OverlayRenderPass()
    : IRenderPass("Overlay")
{
}

OverlayRenderPass::~OverlayRenderPass() = default;

bool OverlayRenderPass::InitializeCommon(int width, int height, void* hwnd) {
    m_width = width;
    m_height = height;
    
    // Initialize ImGui if not already done
    ImGuiManager& imgui = ImGuiManager::GetInstance();
    if (!imgui.Initialize(hwnd)) {
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
    if (!m_initialized) {
        return;
    }
    
    // Check if any components are visible
    bool hasVisibleComponents = m_uiRegistryVisible || m_notificationsVisible;
    if (!hasVisibleComponents) {
        return;
    }
    
    // Begin ImGui frame (backend-specific, implemented by derived classes)
    BeginImGuiFrame();
    
    // Start the ImGui frame
    ImGuiManager::GetInstance().NewFrame();
    
    // Render UI registry content if enabled
    if (m_uiRegistryVisible) {
        UIRegistry::GetInstance().RenderUI();
    }
    
    // Render notifications if enabled
    if (m_notificationsVisible) {
        NotificationManager::GetInstance().Update();
    }
    
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

void OverlayRenderPass::DrawUI() {
    if (ImGui::CollapsingHeader("Visibility Controls", ImGuiTreeNodeFlags_DefaultOpen)) {
        bool uiRegistryVisible = m_uiRegistryVisible.load();
        if (ImGui::Checkbox("Show UI Registry", &uiRegistryVisible)) {
            SetUIRegistryVisible(uiRegistryVisible);
        }
        
        bool notificationsVisible = m_notificationsVisible.load();
        if (ImGui::Checkbox("Show Notifications", &notificationsVisible)) {
            SetNotificationsVisible(notificationsVisible);
        }
    }
}
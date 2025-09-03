#include "OverlayManager.h"
#include "../rendering/renderpass/OverlayRenderPass.h"
#include "UIRegistry.h"

OverlayManager& OverlayManager::GetInstance() {
    static OverlayManager instance;
    return instance;
}

void OverlayManager::SetOverlayRenderPass(OverlayRenderPass* overlayPass) {
    if (m_overlayPass) {
        UIRegistry::GetInstance().UnregisterDrawable(m_overlayPass);
    }
    
    m_overlayPass = overlayPass;
    
    if (m_overlayPass) {
        UIRegistry::GetInstance().RegisterDrawable(m_overlayPass);
    }
}

void OverlayManager::ToggleUIRegistry() {
    if (m_overlayPass) {
        m_overlayPass->ToggleUIRegistryVisibility();
    }
}

void OverlayManager::SetUIRegistryVisible(bool visible) {
    if (m_overlayPass) {
        m_overlayPass->SetUIRegistryVisible(visible);
    }
}

bool OverlayManager::IsUIRegistryVisible() const {
    return m_overlayPass ? m_overlayPass->IsUIRegistryVisible() : false;
}

void OverlayManager::ToggleNotifications() {
    if (m_overlayPass) {
        m_overlayPass->ToggleNotificationsVisibility();
    }
}

void OverlayManager::SetNotificationsVisible(bool visible) {
    if (m_overlayPass) {
        m_overlayPass->SetNotificationsVisible(visible);
    }
}

bool OverlayManager::IsNotificationsVisible() const {
    return m_overlayPass ? m_overlayPass->IsNotificationsVisible() : false;
}
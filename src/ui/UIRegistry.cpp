#include "UIRegistry.h"
#include "imgui.h"
#include "../core/Logger.h"
#include <algorithm>

UIRegistry& UIRegistry::GetInstance() {
    static UIRegistry instance;
    return instance;
}

void UIRegistry::RegisterDrawable(std::shared_ptr<IUIDrawable> drawable) {
    if (!drawable) {
        Logger::GetInstance().Warning("Attempted to register null drawable");
        return;
    }
    
    std::lock_guard<std::mutex> lock(m_mutex);
    m_drawables.push_back(drawable);
    m_needsRegrouping = true;
    
    Logger::GetInstance().Debug("UI drawable registered: {}", drawable->GetUIName());
}

void UIRegistry::UnregisterDrawable(std::shared_ptr<IUIDrawable> drawable) {
    if (!drawable) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(m_mutex);
    
    m_drawables.erase(
        std::remove_if(m_drawables.begin(), m_drawables.end(),
            [&drawable](const std::weak_ptr<IUIDrawable>& weak) {
                return weak.expired() || weak.lock() == drawable;
            }),
        m_drawables.end());
    
    m_needsRegrouping = true;
}

void UIRegistry::Clear() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_drawables.clear();
    m_categorizedDrawables.clear();
    m_needsRegrouping = false;
}

void UIRegistry::RenderUI() {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    // Clean up expired weak pointers
    CleanupExpiredDrawables();
    
    // Regroup if needed
    if (m_needsRegrouping) {
        RegroupDrawables();
    }
    
    // Render main overlay window
    if (ImGui::Begin("Runtime Parameters", nullptr, 
        ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoCollapse)) {
        
        if (m_categorizedDrawables.empty()) {
            ImGui::Text("No configurable modules available");
        } else {
            // Render each category
            for (const auto& [categoryName, drawables] : m_categorizedDrawables) {
                if (!drawables.empty()) {
                    RenderCategory(categoryName, drawables);
                }
            }
        }
    }
    ImGui::End();
}

void UIRegistry::RegroupDrawables() {
    m_categorizedDrawables.clear();
    
    for (const auto& weak : m_drawables) {
        if (auto drawable = weak.lock()) {
            std::string category = drawable->GetUICategory();
            m_categorizedDrawables[category].push_back(weak);
        }
    }
    
    m_needsRegrouping = false;
}

void UIRegistry::RenderCategory(const std::string& categoryName, const std::vector<std::weak_ptr<IUIDrawable>>& drawables) {
    if (ImGui::CollapsingHeader(categoryName.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Indent();
        
        for (const auto& weak : drawables) {
            if (auto drawable = weak.lock()) {
                try {
                    drawable->DrawUI();
                } catch (const std::exception& e) {
                    Logger::GetInstance().Error("Exception in DrawUI for {}: {}", drawable->GetUIName(), e.what());
                    ImGui::Text("Error rendering UI for %s", drawable->GetUIName().c_str());
                }
            }
        }
        
        ImGui::Unindent();
    }
}

void UIRegistry::CleanupExpiredDrawables() {
    m_drawables.erase(
        std::remove_if(m_drawables.begin(), m_drawables.end(),
            [](const std::weak_ptr<IUIDrawable>& weak) {
                return weak.expired();
            }),
        m_drawables.end());
    
    // Clean up categorized drawables too
    for (auto& [category, drawables] : m_categorizedDrawables) {
        drawables.erase(
            std::remove_if(drawables.begin(), drawables.end(),
                [](const std::weak_ptr<IUIDrawable>& weak) {
                    return weak.expired();
                }),
            drawables.end());
    }
}
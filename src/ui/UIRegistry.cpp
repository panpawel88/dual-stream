#include "UIRegistry.h"
#include "imgui.h"
#include "../core/Logger.h"
#include <algorithm>

UIRegistry& UIRegistry::GetInstance() {
    static UIRegistry instance;
    return instance;
}

void UIRegistry::RegisterDrawable(IUIDrawable* drawable) {
    if (!drawable) {
        Logger::GetInstance().Warning("Attempted to register null drawable");
        return;
    }
    
    std::lock_guard<std::mutex> lock(m_mutex);
    m_drawables.push_back(drawable);
    m_needsRegrouping = true;
    
    Logger::GetInstance().Debug("UI drawable registered: {}", drawable->GetUIName());
}

void UIRegistry::UnregisterDrawable(IUIDrawable* drawable) {
    if (!drawable) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(m_mutex);
    
    m_drawables.erase(
        std::remove(m_drawables.begin(), m_drawables.end(), drawable),
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
    
    for (auto drawable : m_drawables) {
        if (drawable) {
            std::string category = drawable->GetUICategory();
            m_categorizedDrawables[category].push_back(drawable);
        }
    }
    
    m_needsRegrouping = false;
}

void UIRegistry::RenderCategory(const std::string& categoryName, const std::vector<IUIDrawable*>& drawables) {
    if (ImGui::CollapsingHeader(categoryName.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Indent();
        
        for (auto drawable : drawables) {
            if (drawable) {
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


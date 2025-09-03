#pragma once
#include "IUIDrawable.h"
#include <vector>
#include <memory>
#include <unordered_map>
#include <string>
#include <mutex>

class UIRegistry {
public:
    static UIRegistry& GetInstance();
    
    void RegisterDrawable(IUIDrawable* drawable);
    void UnregisterDrawable(IUIDrawable* drawable);
    void Clear();
    
    void RenderUI();
    
private:
    UIRegistry() = default;
    ~UIRegistry() = default;
    
    mutable std::mutex m_mutex;
    std::vector<IUIDrawable*> m_drawables;
    
    // Cached groupings for better UI organization
    std::unordered_map<std::string, std::vector<IUIDrawable*>> m_categorizedDrawables;
    bool m_needsRegrouping = true;
    
    void RegroupDrawables();
    void RenderCategory(const std::string& categoryName, const std::vector<IUIDrawable*>& drawables);
};
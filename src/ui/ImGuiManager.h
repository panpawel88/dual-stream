#pragma once
#include <memory>

struct ImGuiContext;

class ImGuiManager {
public:
    static ImGuiManager& GetInstance();
    
    bool Initialize(void* hwnd = nullptr);
    void Shutdown();
    
    void NewFrame();
    void Render();
    
    bool IsInitialized() const { return m_initialized; }
    
private:
    ImGuiManager() = default;
    ~ImGuiManager() = default;
    
    bool m_initialized = false;
    ImGuiContext* m_context = nullptr;
};
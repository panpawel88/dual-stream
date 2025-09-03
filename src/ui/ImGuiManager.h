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
    
    bool ProcessWindowMessage(void* hwnd, unsigned int msg, unsigned long long wParam, long long lParam);
    
    bool IsInitialized() const { return m_initialized; }
    
private:
    ImGuiManager() = default;
    ~ImGuiManager() = default;
    
    bool m_initialized = false;
    ImGuiContext* m_context = nullptr;
};
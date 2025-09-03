#include "ImGuiManager.h"
#include "imgui.h"
#include "imgui_impl_win32.h"
#include "../core/Logger.h"
#include <windows.h>

// Forward declaration for ImGui Win32 message handler
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

ImGuiManager& ImGuiManager::GetInstance() {
    static ImGuiManager instance;
    return instance;
}

bool ImGuiManager::Initialize(void* hwnd) {
    if (m_initialized) {
        return true;
    }
    
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    m_context = ImGui::CreateContext();
    if (!m_context) {
        LOG_ERROR("Failed to create ImGui context");
        return false;
    }
    
    ImGui::SetCurrentContext(m_context);
    
    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    
    // Setup platform/renderer backends (will be completed by specific renderers)
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    // Note: Docking requires ImGui docking branch, not available in standard release
    // io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    
    // Initialize Win32 platform backend if hwnd is provided
    if (hwnd) {
        if (!ImGui_ImplWin32_Init(hwnd)) {
            LOG_ERROR("Failed to initialize ImGui Win32 backend");
            ImGui::DestroyContext(m_context);
            m_context = nullptr;
            return false;
        }
    }
    
    m_initialized = true;
    LOG_INFO("ImGui initialized successfully");
    return true;
}

void ImGuiManager::NewFrame() {
    if (!m_initialized) return;
    
    // Platform-specific new frame (Win32)
    ImGui_ImplWin32_NewFrame();
    
    // Start the Dear ImGui frame
    ImGui::NewFrame();
}

void ImGuiManager::Render() {
    if (!m_initialized) return;
    
    ImGui::Render();
}

void ImGuiManager::Shutdown() {
    if (!m_initialized) return;
    
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext(m_context);
    m_context = nullptr;
    m_initialized = false;
}

bool ImGuiManager::ProcessWindowMessage(void* hwnd, unsigned int msg, unsigned long long wParam, long long lParam) {
    if (!m_initialized) {
        return false;
    }
    
    // Forward the message to ImGui's Win32 handler
    // Returns true if ImGui handled/consumed the message
    LRESULT result = ImGui_ImplWin32_WndProcHandler(static_cast<HWND>(hwnd), msg, 
                                                   static_cast<WPARAM>(wParam), 
                                                   static_cast<LPARAM>(lParam));
    return result != 0;
}
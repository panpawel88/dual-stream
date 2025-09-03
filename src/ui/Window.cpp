#include "Window.h"
#include "ImGuiManager.h"
#include "core/Logger.h"
#include <iostream>
#include <string>
#include <windows.h>

bool Window::s_classRegistered = false;

Window::Window() : m_hwnd(nullptr), m_width(0), m_height(0), m_shouldClose(false), m_isFullscreen(false) {
    for (int i = 0; i < 256; i++) {
        m_keyPressed[i] = false;
    }
    
    m_windowedRect = {0, 0, 0, 0};
    m_windowedStyle = 0;
}

Window::~Window() {
    if (m_hwnd) {
        DestroyWindow(m_hwnd);
    }
}

bool Window::Create(const std::string& title, int width, int height) {
    m_width = width;
    m_height = height;
    
    // For console applications, try different approaches to get valid HINSTANCE
    HINSTANCE hInstance = GetModuleHandle(NULL);

    // Register window class if not already registered
    if (!s_classRegistered) {
        WNDCLASSEX wc = {};
        wc.cbSize = sizeof(WNDCLASSEX);
        wc.style = CS_HREDRAW | CS_VREDRAW;
        wc.lpfnWndProc = WindowProc;
        wc.hInstance = hInstance;
        wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
        wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
        wc.lpszClassName = L"FFmpegPlayerWindow";
        wc.hIcon = LoadIcon(nullptr, IDI_APPLICATION);
        wc.hIconSm = LoadIcon(nullptr, IDI_APPLICATION);
        
        if (!RegisterClassEx(&wc)) {
            DWORD error = GetLastError();
            LOG_ERROR("Failed to register window class, error code: ", error);
            return false;
        }
        s_classRegistered = true;
    }
    
    std::wstring wideTitle(title.begin(), title.end());
    
    // Calculate window size including borders
    RECT rect = {0, 0, width, height};
    DWORD style = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX;
    m_windowedStyle = style; // Store the windowed style for fullscreen toggle
    AdjustWindowRect(&rect, style, FALSE);
    
    int windowWidth = rect.right - rect.left;
    int windowHeight = rect.bottom - rect.top;
    
    int screenWidth = GetSystemMetrics(SM_CXSCREEN);
    int screenHeight = GetSystemMetrics(SM_CYSCREEN);
    int x = (screenWidth - windowWidth) / 2;
    int y = (screenHeight - windowHeight) / 2;
    
    m_hwnd = CreateWindowEx(
        0,
        L"FFmpegPlayerWindow",
        wideTitle.c_str(),
        style,
        x, y,
        windowWidth, windowHeight,
        nullptr,
        nullptr,
        hInstance,
        this
    );
    
    if (!m_hwnd) {
        DWORD error = GetLastError();
        LOG_ERROR("Failed to create window, error code: ", error);
        return false;
    }
    
    return true;
}

void Window::Show() {
    if (m_hwnd) {
        ShowWindow(m_hwnd, SW_SHOW);
        UpdateWindow(m_hwnd);
    }
}

void Window::Hide() {
    if (m_hwnd) {
        ShowWindow(m_hwnd, SW_HIDE);
    }
}

bool Window::ProcessMessages() {
    MSG msg = {};
    while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
        if (msg.message == WM_QUIT) {
            m_shouldClose = true;
            return false;
        }
        
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    
    return !m_shouldClose;
}

void Window::SetTitle(const std::string& title) {
    if (m_hwnd) {
        std::wstring wideTitle(title.begin(), title.end());
        SetWindowText(m_hwnd, wideTitle.c_str());
    }
}

bool Window::IsKeyPressed(int key) const {
    if (key < 0 || key >= 256) {
        return false;
    }
    return m_keyPressed[key];
}

void Window::ClearKeyPress(int key) {
    if (key >= 0 && key < 256) {
        m_keyPressed[key] = false;
    }
}

LRESULT CALLBACK Window::WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    Window* window = nullptr;
    
    if (uMsg == WM_NCCREATE) {
        CREATESTRUCT* createStruct = reinterpret_cast<CREATESTRUCT*>(lParam);
        window = reinterpret_cast<Window*>(createStruct->lpCreateParams);
        SetWindowLongPtr(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(window));
    } else {
        window = reinterpret_cast<Window*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));
    }
    
    if (window) {
        return window->HandleMessage(hwnd, uMsg, wParam, lParam);
    }
    
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

LRESULT Window::HandleMessage(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    // First, forward the message to ImGui if it's initialized
    ImGuiManager& imgui = ImGuiManager::GetInstance();
    if (imgui.IsInitialized()) {
        bool imguiHandled = imgui.ProcessWindowMessage(hwnd, uMsg, wParam, lParam);
        if (imguiHandled) {
            // ImGui consumed the message, don't process it further
            return 0;
        }
    }
    
    // Process the message normally if ImGui didn't handle it
    switch (uMsg) {
        case WM_CLOSE:
            m_shouldClose = true;
            return 0;
            
        case WM_KEYDOWN:
            if (wParam < 256) {
                m_keyPressed[wParam] = true;
            }
            
            if (wParam == VK_ESCAPE) {
                m_shouldClose = true;
            }
            return 0;
            
        case WM_PAINT: {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hwnd, &ps);
            
            RECT rect;
            GetClientRect(hwnd, &rect);
            FillRect(hdc, &rect, (HBRUSH)GetStockObject(BLACK_BRUSH));
            
            EndPaint(hwnd, &ps);
            return 0;
        }
        
        default:
            return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }
}

bool Window::ToggleFullscreen() {
    return SetFullscreen(!m_isFullscreen);
}

bool Window::SetFullscreen(bool fullscreen) {
    if (!m_hwnd || m_isFullscreen == fullscreen) {
        return true; // Already in desired state
    }
    
    if (fullscreen) {
        // Entering fullscreen mode
        
        // Store current window state
        GetWindowRect(m_hwnd, &m_windowedRect);
        
        // Get primary monitor dimensions
        HMONITOR hMonitor = MonitorFromWindow(m_hwnd, MONITOR_DEFAULTTOPRIMARY);
        MONITORINFO monitorInfo = {};
        monitorInfo.cbSize = sizeof(MONITORINFO);
        if (!GetMonitorInfo(hMonitor, &monitorInfo)) {
            LOG_ERROR("Failed to get monitor info for fullscreen");
            return false;
        }
        
        // Change window style to borderless
        SetWindowLong(m_hwnd, GWL_STYLE, WS_POPUP);
        
        // Set window position and size to cover entire screen
        SetWindowPos(m_hwnd, HWND_TOP,
            monitorInfo.rcMonitor.left, monitorInfo.rcMonitor.top,
            monitorInfo.rcMonitor.right - monitorInfo.rcMonitor.left,
            monitorInfo.rcMonitor.bottom - monitorInfo.rcMonitor.top,
            SWP_FRAMECHANGED | SWP_SHOWWINDOW);
            
        // Update internal dimensions
        m_width = monitorInfo.rcMonitor.right - monitorInfo.rcMonitor.left;
        m_height = monitorInfo.rcMonitor.bottom - monitorInfo.rcMonitor.top;
        
        m_isFullscreen = true;
        LOG_INFO("Entered fullscreen mode (", m_width, "x", m_height, ")");
        
    } else {
        // Exiting fullscreen mode
        
        // Restore window style
        SetWindowLong(m_hwnd, GWL_STYLE, m_windowedStyle);
        
        // Restore window position and size
        SetWindowPos(m_hwnd, HWND_NOTOPMOST,
            m_windowedRect.left, m_windowedRect.top,
            m_windowedRect.right - m_windowedRect.left,
            m_windowedRect.bottom - m_windowedRect.top,
            SWP_FRAMECHANGED | SWP_SHOWWINDOW);
            
        // Update internal dimensions to client area size
        RECT clientRect;
        GetClientRect(m_hwnd, &clientRect);
        m_width = clientRect.right - clientRect.left;
        m_height = clientRect.bottom - clientRect.top;
        
        m_isFullscreen = false;
        LOG_INFO("Exited fullscreen mode (", m_width, "x", m_height, ")");
    }
    
    return true;
}
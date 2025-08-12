#include "Window.h"
#include "Logger.h"
#include <iostream>
#include <string>
#include <windows.h>

bool Window::s_classRegistered = false;

Window::Window() : m_hwnd(nullptr), m_width(0), m_height(0), m_shouldClose(false) {
    // Initialize key press array
    for (int i = 0; i < 256; i++) {
        m_keyPressed[i] = false;
    }
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
    if (!hInstance) {
        // Try getting the executable's module handle
        wchar_t exePath[MAX_PATH];
        GetModuleFileName(NULL, exePath, MAX_PATH);
        hInstance = GetModuleHandle(exePath);
        
        if (!hInstance) {
            // Last resort - use a dummy value that Windows will accept
            hInstance = reinterpret_cast<HINSTANCE>(0x400000);  // Default base address for executables
        }
    }
    
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
    
    // Convert title to wide string
    std::wstring wideTitle(title.begin(), title.end());
    
    // Calculate window size including borders
    RECT rect = {0, 0, width, height};
    DWORD style = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX;
    AdjustWindowRect(&rect, style, FALSE);
    
    int windowWidth = rect.right - rect.left;
    int windowHeight = rect.bottom - rect.top;
    
    // Center window on screen
    int screenWidth = GetSystemMetrics(SM_CXSCREEN);
    int screenHeight = GetSystemMetrics(SM_CYSCREEN);
    int x = (screenWidth - windowWidth) / 2;
    int y = (screenHeight - windowHeight) / 2;
    
    // Create window
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
    switch (uMsg) {
        case WM_CLOSE:
            m_shouldClose = true;
            return 0;
            
        case WM_KEYDOWN:
            // Track key presses
            if (wParam < 256) {
                m_keyPressed[wParam] = true;
            }
            
            // Handle ESC immediately
            if (wParam == VK_ESCAPE) {
                m_shouldClose = true;
            }
            return 0;
            
        case WM_PAINT: {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hwnd, &ps);
            
            // Fill with black background for now
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
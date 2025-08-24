#pragma once

#include <windows.h>
#include <string>

class Window {
public:
    Window();
    ~Window();
    
    bool Create(const std::string& title, int width, int height);
    void Show();
    void Hide();
    bool ProcessMessages();
    void SetTitle(const std::string& title);
    
    HWND GetHandle() const { return m_hwnd; }
    int GetWidth() const { return m_width; }
    int GetHeight() const { return m_height; }
    bool ShouldClose() const { return m_shouldClose; }
    
    // Key input
    bool IsKeyPressed(int key) const;
    void ClearKeyPress(int key);
    
private:
    static LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
    LRESULT HandleMessage(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
    
    HWND m_hwnd;
    int m_width;
    int m_height;
    bool m_shouldClose;
    bool m_keyPressed[256]; // Track key presses
    static bool s_classRegistered;
};
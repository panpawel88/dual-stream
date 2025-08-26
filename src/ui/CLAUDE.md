# User Interface System

This directory implements the window management and user input handling system, providing a flexible Win32-based interface with modern features like fullscreen support and dynamic resizing.

## Architecture Overview

The UI system centers around a single `Window` class that encapsulates all Win32 window management, input handling, and display control functionality.

## Core Component

### Window Class
**File:** `Window.h/cpp`
**Purpose:** Complete Win32 window management with modern features

**Key Features:**
- **Dynamic Window Management:** Resizable windows with size change detection
- **Fullscreen Support:** Toggle fullscreen mode (F11 key) with state restoration
- **Input Handling:** Keyboard input with edge detection and state management
- **Message Processing:** Win32 message loop integration
- **HWND Access:** Direct window handle access for renderer integration

## Window Management Features

### Dynamic Sizing
**Major Change from Original:** Windows are now resizable, not fixed-size

```cpp
class Window {
public:
    // Dynamic window creation
    bool Create(const std::string& title, int width, int height);
    
    // Size access and monitoring
    int GetWidth() const { return m_width; }
    int GetHeight() const { return m_height; }
    
    // Applications can detect size changes for renderer adjustment
    if (currentWidth != lastWindowWidth || currentHeight != lastWindowHeight) {
        renderer->Resize(currentWidth, currentHeight);
    }
};
```

**Window Sizing Logic:**
```cpp
// Application chooses window size based on video resolution and display limits
int maxVideoWidth = std::max(video1Info.width, video2Info.width);
int maxVideoHeight = std::max(video1Info.height, video2Info.height);

// Limit to display resolution
int screenWidth = GetSystemMetrics(SM_CXSCREEN);  
int screenHeight = GetSystemMetrics(SM_CYSCREEN);

int windowWidth = std::min(maxVideoWidth, screenWidth);
int windowHeight = std::min(maxVideoHeight, screenHeight);
```

### Fullscreen Support
**New Feature:** Complete fullscreen mode with state management

```cpp
class Window {
public:
    bool ToggleFullscreen();           // F11 key toggle
    bool SetFullscreen(bool fullscreen); // Programmatic control
    bool IsFullscreen() const { return m_isFullscreen; }

private:
    bool m_isFullscreen;
    RECT m_windowedRect;    // Restore windowed position/size
    DWORD m_windowedStyle;  // Restore windowed style flags
};
```

**Fullscreen Implementation:**
- **State Preservation:** Saves windowed position, size, and style
- **Seamless Transition:** Smooth transition between windowed/fullscreen
- **Renderer Integration:** Automatic renderer resize on fullscreen changes
- **Restore Capability:** Perfect restoration of windowed state

## Input System

### Keyboard Input Handling
**Enhanced from Original:** Advanced key state management with edge detection

```cpp
class Window {
public:
    bool IsKeyPressed(int key) const;    // Check current key state
    void ClearKeyPress(int key);         // Clear key press state (for triggers)

private:
    bool m_keyPressed[256];              // Track all key states
    
    // Win32 message handling
    LRESULT HandleMessage(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
        switch (uMsg) {
            case WM_KEYDOWN:
                if (wParam < 256) {
                    m_keyPressed[wParam] = true;  // Set key pressed
                }
                break;
                
            case WM_KEYUP:
                if (wParam < 256) {
                    m_keyPressed[wParam] = false; // Clear key pressed
                }
                break;
        }
    }
};
```

**Key Mapping:**
- **'1' Key:** Switch to video 1 (handled by KeyboardSwitchingTrigger)
- **'2' Key:** Switch to video 2 (handled by KeyboardSwitchingTrigger)  
- **F11 Key:** Toggle fullscreen mode (handled directly by Window)
- **ESC Key:** Exit application (handled by Win32 message loop)

### Input State Management
**Design Pattern:** Clean separation between input detection and action handling

```cpp
// Trigger system queries input state
bool KeyboardSwitchingTrigger::Update() {
    // Window provides raw input state
    if (m_window->IsKeyPressed('1') && !m_key1Triggered) {
        m_key1Triggered = true;  // Edge detection in trigger
    }
}

// Triggers clear input after action
void KeyboardSwitchingTrigger::Reset() {
    if (m_key1Triggered) {
        m_window->ClearKeyPress('1');  // Clear window key state
        m_key1Triggered = false;       // Clear trigger state
    }
}
```

## Win32 Integration

### Window Class Registration  
**Efficient Resource Management:** Static class registration with reference counting

```cpp
class Window {
private:
    static bool s_classRegistered;     // Shared class registration state
    
    bool Create(const std::string& title, int width, int height) {
        if (!s_classRegistered) {
            // Register window class once per application
            WNDCLASS wc = {};
            wc.lpfnWndProc = WindowProc;
            wc.hInstance = GetModuleHandle(nullptr);
            wc.lpszClassName = "DualStreamVideoPlayerWindow";
            RegisterClass(&wc);
            s_classRegistered = true;
        }
        
        // Create window instance
        m_hwnd = CreateWindow(...);
    }
};
```

### Message Processing
**Integration Pattern:** Cooperative message loop with application logic

```cpp
// Application main loop
while (window.ProcessMessages()) {
    // Window handles Win32 messages
    // Application handles video/rendering logic
    
    videoManager.UpdateSwitchingTrigger();
    videoManager.ProcessSwitchingTriggers();
    
    if (videoManager.ShouldUpdateFrame()) {
        videoManager.UpdateFrame();
    }
    
    renderer->Present(renderTexture);
    Sleep(1); // Prevent busy waiting
}
```

**Message Loop Implementation:**
```cpp
bool Window::ProcessMessages() {
    MSG msg;
    while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
        if (msg.message == WM_QUIT) {
            return false;  // Signal application exit
        }
        
        TranslateMessage(&msg);
        DispatchMessage(&msg);  // Calls WindowProc
    }
    return !m_shouldClose;
}
```

## Window Procedure Handling

### Event Processing
**Comprehensive Message Handling:** All essential Win32 messages handled

```cpp
LRESULT Window::HandleMessage(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
        case WM_CLOSE:
            m_shouldClose = true;
            return 0;
            
        case WM_SIZE:
            // Handle window resize
            m_width = LOWORD(lParam);
            m_height = HIWORD(lParam);
            return 0;
            
        case WM_KEYDOWN:
            // Handle key press events
            if (wParam == VK_F11) {
                ToggleFullscreen();  // Built-in fullscreen toggle
            } else if (wParam == VK_ESCAPE) {
                m_shouldClose = true;  // ESC to exit
            } else if (wParam < 256) {
                m_keyPressed[wParam] = true;  // General key tracking
            }
            return 0;
            
        case WM_KEYUP:
            // Handle key release events  
            if (wParam < 256) {
                m_keyPressed[wParam] = false;
            }
            return 0;
            
        default:
            return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }
}
```

## Renderer Integration

### HWND Access
**Clean Integration:** Window provides handle for renderer initialization

```cpp
// Renderer initialization
Window window;
window.Create("DualStream Video Player", width, height);
window.Show();

auto renderer = RendererFactory::CreateRenderer();
renderer->Initialize(window.GetHandle(), width, height);  // HWND passed to renderer
```

### Dynamic Resize Support  
**Real-Time Adaptation:** Automatic renderer adjustment on window size changes

```cpp
// Main loop resize handling
int lastWindowWidth = window.GetWidth();
int lastWindowHeight = window.GetHeight();

while (window.ProcessMessages()) {
    int currentWidth = window.GetWidth();
    int currentHeight = window.GetHeight();
    
    if (currentWidth != lastWindowWidth || currentHeight != lastWindowHeight) {
        LOG_INFO("Window size changed to ", currentWidth, "x", currentHeight);
        
        if (!renderer->Resize(currentWidth, currentHeight)) {
            LOG_ERROR("Failed to resize renderer");
            break;
        }
        
        lastWindowWidth = currentWidth;
        lastWindowHeight = currentHeight;
    }
    
    // Continue with video processing...
}
```

## State Management

### Window State Tracking
**Comprehensive State:** All relevant window properties tracked

```cpp
class Window {
private:
    HWND m_hwnd;                // Win32 window handle
    int m_width, m_height;      // Current window dimensions
    bool m_shouldClose;         // Exit flag from user/system
    bool m_keyPressed[256];     // Complete keyboard state
    
    // Fullscreen state management
    bool m_isFullscreen;        // Current fullscreen state
    RECT m_windowedRect;        // Saved windowed position/size
    DWORD m_windowedStyle;      // Saved windowed style flags
};
```

### Lifecycle Management
**RAII Pattern:** Automatic resource cleanup

```cpp
Window::~Window() {
    if (m_hwnd) {
        DestroyWindow(m_hwnd);  // Automatic Win32 cleanup
        m_hwnd = nullptr;
    }
}
```

## Performance Considerations

### Message Loop Efficiency
- **PeekMessage Usage:** Non-blocking message processing allows video processing
- **Minimal Sleep:** 1ms sleep prevents excessive CPU usage while maintaining responsiveness
- **Event-Driven:** Only processes messages when available

### Input Latency
- **Direct Key State:** Immediate key state access for responsive switching
- **Edge Detection:** Trigger system prevents key repeat issues
- **State Clearing:** Explicit state management prevents input lag

## Integration Points

### With Trigger System
```cpp
// Window provides raw input access for triggers
class KeyboardSwitchingTrigger {
    Window* m_window;  // Window reference for input access
    
    void Update() override {
        if (m_window->IsKeyPressed('1')) { /* handle */ }
        if (m_window->IsKeyPressed('2')) { /* handle */ }
    }
};
```

### With Rendering System
```cpp
// Window provides HWND for renderer initialization and resize events
auto renderer = RendererFactory::CreateRenderer();
renderer->Initialize(window.GetHandle(), window.GetWidth(), window.GetHeight());

// Dynamic resize support
if (window.GetWidth() != lastWidth || window.GetHeight() != lastHeight) {
    renderer->Resize(window.GetWidth(), window.GetHeight());
}
```

### With Application Logic
```cpp
// Clean integration with main application loop
while (window.ProcessMessages()) {
    // Window handles UI events
    // Application handles video/rendering
    // Cooperative multitasking pattern
}
```

This UI system provides a robust, modern window management foundation with comprehensive input handling and seamless integration with the video processing and rendering systems.
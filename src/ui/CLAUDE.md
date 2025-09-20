# User Interface System

This directory implements a comprehensive user interface system combining Win32 window management with ImGui-based overlay functionality for modern UI elements, notifications, and debug information.

## Architecture Overview

The UI system has evolved from a single window class into a multi-component architecture:

```
src/ui/
├── Window.h/cpp                   # Core Win32 window management  
├── ImGuiManager.h/cpp              # ImGui context and rendering management
├── OverlayManager.h/cpp            # Overlay visibility and coordination
├── GlobalInputHandler.h/cpp        # Centralized input processing
├── UIRegistry.h/cpp                # UI component registration system
├── NotificationManager.h/cpp       # Toast notification system
├── IUIDrawable.h                   # Interface for UI components
└── CLAUDE.md                       # This documentation
```

## Core Components

### IUIDrawable Interface
**File:** `IUIDrawable.h`
**Purpose:** Standard interface for all UI components in the system

**Key Interface:**
```cpp
class IUIDrawable {
public:
    virtual ~IUIDrawable() = default;

    virtual void DrawUI() = 0;  // Render UI elements
    virtual std::string GetUIName() const = 0;  // Component identifier
    virtual std::string GetUICategory() const = 0;  // Category for organization
};
```

### Window Class (Foundation)
**File:** `Window.h/cpp`
**Purpose:** Complete Win32 window management with modern features

**Key Features:**
- **Dynamic Window Management:** Resizable windows with size change detection
- **Fullscreen Support:** Toggle fullscreen mode (F11 key) with state restoration
- **Input Handling:** Keyboard input with edge detection and state management
- **Message Processing:** Win32 message loop integration
- **HWND Access:** Direct window handle access for renderer integration

### ImGui Integration
**File:** `ImGuiManager.h/cpp`
**Purpose:** ImGui context management and rendering coordination

**Key Features:**
- **Singleton Pattern:** Global ImGui context management
- **Platform Integration:** Win32 and renderer backend setup
- **Frame Management:** ImGui frame lifecycle (NewFrame/Render)
- **Input Processing:** Win32 message integration for ImGui
- **Context Safety:** Proper ImGui initialization and shutdown

### Overlay Management
**File:** `OverlayManager.h/cpp` 
**Purpose:** Coordinates visibility and rendering of overlay UI elements

**Key Features:**
- **UI Registry Integration:** Manages debug/config UI visibility
- **Notification System:** Controls toast notification display
- **Render Pass Coordination:** Integrates with overlay render passes
- **Global State:** Centralized overlay visibility control

### Global Input Handling
**File:** `GlobalInputHandler.h/cpp`
**Purpose:** Centralized input processing for UI and application hotkeys

**Key Features:**
- **Hotkey Processing:** Global application shortcuts (F1, F2, etc.)
- **UI Toggle Control:** Keyboard shortcuts for overlay management
- **Input Coordination:** Manages input between Win32 and ImGui systems

### Component Registration System
**File:** `UIRegistry.h/cpp`, `IUIDrawable.h`
**Purpose:** Registration and management system for UI components

**Key Features:**
- **Component Registration:** Dynamic UI component registration
- **Draw Interface:** Standardized rendering interface for UI elements
- **Lifecycle Management:** Automatic UI component cleanup
- **Debug UI:** Runtime UI component inspection and control
- **Category Organization:** UI components grouped by category (Camera, Rendering, etc.)

**Camera Control Integration:**
```cpp
// Camera UI registration example
auto cameraControlUI = std::make_shared<CameraControlUI>();
if (cameraControlUI->Initialize(&cameraManager, renderer.get())) {
    UIRegistry::GetInstance().RegisterDrawable(cameraControlUI.get());
    cameraManager.RegisterFrameListener(cameraControlUI); // For live preview
}
```

### Notification System
**File:** `NotificationManager.h/cpp`
**Purpose:** Toast notification system for user feedback

**Key Features:**
- **Toast Notifications:** Non-intrusive user notifications
- **Priority System:** Different notification levels and styling
- **Auto-dismiss:** Configurable timeout and dismiss behavior
- **ImGui Integration:** Seamless rendering within overlay system

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
    
    // Input processing integration
    bool ProcessMessage(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
    
    // Applications can detect size changes for renderer adjustment
    if (currentWidth != lastWindowWidth || currentHeight != lastWindowHeight) {
        renderer->Resize(currentWidth, currentHeight);
        ImGuiManager::GetInstance().HandleResize(currentWidth, currentHeight);
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

## Integration with Overlay Rendering

### Render Pass Integration
**Connection:** UI system integrates with overlay render passes for seamless rendering

```cpp
// OverlayManager coordinates with render passes
OverlayManager::SetOverlayRenderPass(overlayRenderPass);

// Render pass calls UI system for overlay content
class OverlayRenderPass {
    void RenderOverlays() {
        ImGuiManager::GetInstance().NewFrame();
        
        if (OverlayManager::GetInstance().IsUIRegistryVisible()) {
            UIRegistry::GetInstance().DrawDebugUI();
        }
        
        if (OverlayManager::GetInstance().IsNotificationsVisible()) {
            NotificationManager::GetInstance().DrawNotifications();
        }
        
        ImGuiManager::GetInstance().Render();
    }
};
```

### Input Flow Architecture
```
Win32 Messages → Window::HandleMessage() → GlobalInputHandler
     │
     └─────────────────────────────────┤
                                    │
                 ┌─────────────────────────────────┘
                 │
                 ├─ ImGuiManager::ProcessWindowMessage()  # ImGui input
                 ├─ OverlayManager toggle hotkeys (F1, F2) # UI control
                 └─ Application input (1, 2, ESC, F11)      # Video control
```

## Modern UI Features

### Debug UI System
**Runtime Configuration:** Live editing of application parameters
- **Component Registry:** All registered UI components accessible via F1
- **Parameter Editing:** Real-time modification of render pass parameters
- **Performance Monitoring:** Frame timing and resource usage display
- **System Status:** Video decoder status, renderer information
- **Camera Control Panel:** Live camera property adjustment and frame preview

**Camera Control UI Features:**
- **Live Preview:** Real-time camera feed display with configurable resolution
- **Property Sliders:** Runtime adjustment of brightness, contrast, saturation, gain
- **Smart Ranges:** Properties automatically normalized to 0-100% range
- **Device Information:** Camera name, resolution, FPS, and frame statistics
- **Multi-Backend Support:** Works with both DirectX 11 and OpenGL renderers

### Toast Notification System
**User Feedback:** Non-intrusive status and error notifications
- **Success Notifications:** Configuration changes, mode switches
- **Error Notifications:** File loading errors, hardware failures
- **Info Notifications:** Feature toggles, status updates
- **Auto-dismiss:** Configurable timeout with manual dismiss option

### Overlay Management
**Seamless Integration:** Overlay UI elements that don't interfere with video content
- **Render Pass Integration:** Overlays rendered as final pass in pipeline
- **Transparency Support:** Alpha blending with video content
- **Input Passthrough:** UI input doesn't interfere with video controls
- **Performance Optimized:** Zero overhead when overlays are hidden

## Configuration and Hotkeys

### Input Mappings
```
Application Controls:
│── 1, 2 Keys     ─ Video switching
│── F11 Key      ─ Fullscreen toggle  
│── ESC Key      ─ Exit application
│
UI Controls:
│── F1 Key       ─ Toggle debug UI registry (includes Camera Control)
│── F2 Key       ─ Toggle notifications
└── Mouse/Keys   ─ ImGui interaction when overlay active

### Camera UI Controls (when enabled):
│── Camera Control Panel ─ Live property adjustment
│── Preview Window      ─ Real-time camera feed
└── Property Sliders    ─ Brightness, contrast, saturation, gain
```

### ImGui Theme Integration
**Visual Design:** Modern, minimal UI that complements video content
- **Dark Theme:** Non-distracting dark color scheme
- **Transparency:** Semi-transparent backgrounds for video visibility
- **Minimal Chrome:** Clean, minimal window decorations
- **Video-First:** UI never obscures important video content

## Camera Control UI Integration

### CameraControlUI Component
**Implementation:** Camera control UI follows the established IUIDrawable pattern

**Key Features:**
- **IUIDrawable Implementation:** Integrates seamlessly with existing UI system
- **ICameraFrameListener:** Receives camera frames for live preview display
- **Multi-Backend Texture Conversion:** Automatic texture creation for ImGui display
- **Thread-Safe Operation:** Safe interaction with background camera capture

**UI Panel Organization:**
```
Camera Control Window
├── Camera Information
│   ├── Device Name
│   ├── Resolution & FPS
│   └── Frame Statistics
├── Property Controls
│   ├── Brightness (0-100%)
│   ├── Contrast (0-100%)
│   ├── Saturation (0-100%)
│   └── Gain (0-100%)
└── Live Preview
    ├── Real-time Camera Feed
    ├── Preview Resolution Controls
    └── Frame Rate Limiting
```

### Frame-to-Texture Pipeline
**CameraFrameTexture Component:** Converts camera frames to renderer textures

**Conversion Process:**
```
Camera Frame (cv::Mat) → Format Conversion → RGBA Buffer → GPU Texture → ImGui::Image()

Renderer-Specific Paths:
├── DirectX 11: ID3D11Texture2D + ID3D11ShaderResourceView
└── OpenGL: glTexture2D with GL_RGBA format
```

**Performance Optimization:**
- **Throttled Updates:** Configurable preview refresh rate (default: 10 FPS)
- **Automatic Scaling:** Large frames scaled for UI display (max 640x480)
- **Conditional Processing:** Only processes frames when preview is visible
- **Texture Caching:** Efficient texture reuse and memory management

### Configuration Integration
**INI Settings:** Camera UI fully configurable via existing configuration system

```ini
[camera_ui]
enable_camera_ui = true          # Enable camera UI independently
preview_enabled = true           # Show live camera preview
preview_max_width = 640          # Maximum preview resolution
preview_max_height = 480
preview_fps = 10.0              # Preview refresh rate
```

### Thread Safety Architecture
**Background Integration:** Camera UI safely integrates with camera capture threads

```cpp
// Frame processing on background thread
FrameProcessingResult ProcessFrame(std::shared_ptr<const CameraFrame> frame) {
    std::lock_guard<std::mutex> lock(m_frameMutex);
    m_currentFrame = frame;  // Safe shared_ptr assignment
    m_hasNewFrame = true;
    return FrameProcessingResult::SUCCESS;
}

// UI updates on main thread during DrawUI()
void DrawUI() {
    if (m_hasNewFrame && ShouldUpdatePreview()) {
        UpdateFrameTexture();  // Convert frame to GPU texture
    }
    ImGui::Image(textureID, displaySize);  // Render in ImGui
}
```

This evolved UI system provides a comprehensive modern interface combining robust Win32 window management with advanced ImGui-based overlay functionality for debugging, configuration, camera control, and user feedback while maintaining seamless integration with the video processing and rendering systems.
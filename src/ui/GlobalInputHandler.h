#pragma once
#include <windows.h>
#include <unordered_map>
#include <functional>

class Window; // Forward declaration

/**
 * Global input handler for application-wide keyboard shortcuts.
 * Independent of video switching triggers and always active.
 */
class GlobalInputHandler {
public:
    static GlobalInputHandler& GetInstance();
    
    // Key binding system
    using KeyCallback = std::function<void()>;
    void RegisterKeyBinding(int virtualKeyCode, KeyCallback callback);
    void UnregisterKeyBinding(int virtualKeyCode);
    
    // Update method to be called in main loop
    void Update();
    
    // Predefined system shortcuts
    void RegisterOverlayToggle(int keyCode = VK_INSERT);  // Toggles UI Registry visibility
    void RegisterFullscreenToggle(int keyCode, Window* window);
    
private:
    GlobalInputHandler() = default;
    ~GlobalInputHandler() = default;
    
    struct KeyState {
        KeyCallback callback;
        bool previouslyPressed = false;
    };
    
    std::unordered_map<int, KeyState> m_keyBindings;
};
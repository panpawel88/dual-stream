#include "GlobalInputHandler.h"
#include "OverlayManager.h"
#include "NotificationManager.h"
#include "../core/Logger.h"

GlobalInputHandler& GlobalInputHandler::GetInstance() {
    static GlobalInputHandler instance;
    return instance;
}

void GlobalInputHandler::RegisterKeyBinding(int virtualKeyCode, KeyCallback callback) {
    m_keyBindings[virtualKeyCode] = {callback, false};
    Logger::GetInstance().Debug("Registered global key binding for virtual key code: {}", virtualKeyCode);
}

void GlobalInputHandler::UnregisterKeyBinding(int virtualKeyCode) {
    m_keyBindings.erase(virtualKeyCode);
    Logger::GetInstance().Debug("Unregistered global key binding for virtual key code: {}", virtualKeyCode);
}

void GlobalInputHandler::Update() {
    for (auto& [keyCode, keyState] : m_keyBindings) {
        bool currentlyPressed = (GetAsyncKeyState(keyCode) & 0x8000) != 0;
        
        // Trigger on key press (not while held)
        if (currentlyPressed && !keyState.previouslyPressed) {
            try {
                keyState.callback();
            } catch (const std::exception& e) {
                Logger::GetInstance().Error("Exception in key callback for key {}: {}", keyCode, e.what());
            }
        }
        
        keyState.previouslyPressed = currentlyPressed;
    }
}

void GlobalInputHandler::RegisterOverlayToggle(int keyCode) {
    RegisterKeyBinding(keyCode, []() {
        OverlayManager::GetInstance().ToggleOverlay();
        
        // Show notification
        bool isVisible = OverlayManager::GetInstance().IsOverlayVisible();
        NotificationManager::GetInstance().ShowInfo("Overlay", 
            isVisible ? "Overlay enabled" : "Overlay disabled");

        Logger::GetInstance().Info("Overlay toggled: ", isVisible ? "enabled" : "disabled");
    });
}

void GlobalInputHandler::RegisterFullscreenToggle(int keyCode) {
    RegisterKeyBinding(keyCode, []() {
        // TODO: Add fullscreen toggle functionality
        // This would integrate with the Window class
        Logger::GetInstance().Info("Fullscreen toggle requested (not yet implemented)");
        NotificationManager::GetInstance().ShowInfo("Fullscreen", "Toggle requested");
    });
}
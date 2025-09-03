#include "GlobalInputHandler.h"
#include "Window.h"
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
        OverlayManager::GetInstance().ToggleUIRegistry();
        
        // Show notification
        bool isVisible = OverlayManager::GetInstance().IsUIRegistryVisible();
        NotificationManager::GetInstance().ShowInfo("UI Registry", 
            isVisible ? "UI Registry enabled" : "UI Registry disabled");

        Logger::GetInstance().Info("UI Registry toggled: ", isVisible ? "enabled" : "disabled");
    });
}

void GlobalInputHandler::RegisterFullscreenToggle(int keyCode, Window* window) {
    RegisterKeyBinding(keyCode, [window]() {
        if (window) {
            bool wasFullscreen = window->IsFullscreen();
            window->ToggleFullscreen();
            bool isFullscreen = window->IsFullscreen();
            
            NotificationManager::GetInstance().ShowInfo("Fullscreen", 
                isFullscreen ? "Fullscreen enabled" : "Fullscreen disabled");
            
            Logger::GetInstance().Info("Fullscreen toggled: ", 
                isFullscreen ? "enabled" : "disabled");
        } else {
            Logger::GetInstance().Error("Cannot toggle fullscreen: Window reference is null");
        }
    });
}
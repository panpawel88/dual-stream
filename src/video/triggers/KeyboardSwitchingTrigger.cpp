#include "KeyboardSwitchingTrigger.h"
#include "ui/Window.h"

KeyboardSwitchingTrigger::KeyboardSwitchingTrigger(Window* window)
    : m_window(window), m_key1Triggered(false), m_key2Triggered(false) {
}

bool KeyboardSwitchingTrigger::ShouldSwitchToVideo1() {
    return m_key1Triggered;
}

bool KeyboardSwitchingTrigger::ShouldSwitchToVideo2() {
    return m_key2Triggered;
}

void KeyboardSwitchingTrigger::Update() {
    if (!m_window) {
        return;
    }
    
    // Check for key presses and set trigger flags
    if (m_window->IsKeyPressed('1') && !m_key1Triggered) {
        m_key1Triggered = true;
    }
    
    if (m_window->IsKeyPressed('2') && !m_key2Triggered) {
        m_key2Triggered = true;
    }
}

void KeyboardSwitchingTrigger::Reset() {
    // Clear the trigger flags and key press states
    if (m_window) {
        if (m_key1Triggered) {
            m_window->ClearKeyPress('1');
            m_key1Triggered = false;
        }
        
        if (m_key2Triggered) {
            m_window->ClearKeyPress('2');
            m_key2Triggered = false;
        }
    }
}

std::string KeyboardSwitchingTrigger::GetName() const {
    return "Keyboard";
}
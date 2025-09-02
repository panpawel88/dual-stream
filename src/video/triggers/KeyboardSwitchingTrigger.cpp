#include "KeyboardSwitchingTrigger.h"
#include "ui/Window.h"

KeyboardSwitchingTrigger::KeyboardSwitchingTrigger(Window* window, size_t videoCount)
    : m_window(window), m_videoCount(videoCount), m_triggeredVideoIndex(0), m_keyTriggered(false) {
}

std::optional<size_t> KeyboardSwitchingTrigger::GetTargetVideoIndex() {
    if (m_keyTriggered) {
        return m_triggeredVideoIndex;
    }
    return std::nullopt;
}

void KeyboardSwitchingTrigger::Update() {
    if (!m_window || m_keyTriggered) {
        return;
    }
    
    // Check for number key presses (1-9 for videos 0-8)
    for (char key = '1'; key <= '9'; key++) {
        if (m_window->IsKeyPressed(key)) {
            size_t videoIndex = static_cast<size_t>(key - '1'); // '1' maps to index 0, '2' to 1, etc.
            if (videoIndex < m_videoCount) {
                m_triggeredVideoIndex = videoIndex;
                m_keyTriggered = true;
                
            }
            break;
        }
    }
    
    // Check for '0' key which maps to video index 9
    if (m_window->IsKeyPressed('0')) {
        if (9 < m_videoCount) {
            m_triggeredVideoIndex = 9;
            m_keyTriggered = true;
            
        }
    }
}

void KeyboardSwitchingTrigger::Reset() {
    // Clear the trigger flags and key press states
    if (m_window && m_keyTriggered) {
        // Clear the specific key that was pressed
        char keyPressed = static_cast<char>('1' + m_triggeredVideoIndex);
        if (m_triggeredVideoIndex == 9) {
            keyPressed = '0';
        }
        m_window->ClearKeyPress(keyPressed);
        m_keyTriggered = false;
    }
}

std::string KeyboardSwitchingTrigger::GetName() const {
    return "Keyboard";
}
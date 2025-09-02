#pragma once

#include "ISwitchingTrigger.h"

// Forward declaration to avoid including Windows.h in header
class Window;

/**
 * Keyboard-based switching trigger that responds to key presses.
 * This is the default trigger that implements the original behavior (keys 1 and 2).
 */
class KeyboardSwitchingTrigger : public ISwitchingTrigger {
public:
    /**
     * Construct keyboard trigger with window for input handling.
     * @param window Pointer to window instance for reading key states
     * @param videoCount Number of videos available for switching (1-10)
     */
    KeyboardSwitchingTrigger(Window* window, size_t videoCount);
    
    ~KeyboardSwitchingTrigger() override = default;
    
    // ISwitchingTrigger implementation
    std::optional<size_t> GetTargetVideoIndex() override;
    void Update() override;
    void Reset() override;
    std::string GetName() const override;

private:
    Window* m_window;
    size_t m_videoCount;
    size_t m_triggeredVideoIndex;
    bool m_keyTriggered;
};
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
     */
    explicit KeyboardSwitchingTrigger(Window* window);
    
    ~KeyboardSwitchingTrigger() override = default;
    
    // ISwitchingTrigger implementation
    bool ShouldSwitchToVideo1() override;
    bool ShouldSwitchToVideo2() override;
    void Update() override;
    void Reset() override;
    std::string GetName() const override;

private:
    Window* m_window;
    bool m_key1Triggered;
    bool m_key2Triggered;
};
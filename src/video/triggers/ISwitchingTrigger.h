#pragma once

#include <string>

/**
 * Interface for video switching trigger strategies.
 * Defines when video switching should occur, independent of how the switch is performed.
 */
class ISwitchingTrigger {
public:
    virtual ~ISwitchingTrigger() = default;
    
    /**
     * Check if a switch to video 1 should be triggered.
     * @return true if video should switch to video 1
     */
    virtual bool ShouldSwitchToVideo1() = 0;
    
    /**
     * Check if a switch to video 2 should be triggered.
     * @return true if video should switch to video 2
     */
    virtual bool ShouldSwitchToVideo2() = 0;
    
    /**
     * Update trigger state. Called each frame to maintain trigger logic.
     */
    virtual void Update() = 0;
    
    /**
     * Reset trigger state after a switch has occurred.
     * Used to prevent repeated triggering from the same input.
     */
    virtual void Reset() = 0;
    
    /**
     * Get the name of this trigger strategy for debugging/logging.
     * @return string name of the trigger strategy
     */
    virtual std::string GetName() const = 0;
};
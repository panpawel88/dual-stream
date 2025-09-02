#pragma once

#include <string>
#include <optional>

/**
 * Interface for video switching trigger strategies.
 * Defines when video switching should occur, independent of how the switch is performed.
 */
class ISwitchingTrigger {
public:
    virtual ~ISwitchingTrigger() = default;
    
    /**
     * Check if a video switch should be triggered and get the target video index.
     * @return Optional containing the target video index (0-based) if a switch should occur, std::nullopt otherwise
     */
    virtual std::optional<size_t> GetTargetVideoIndex() = 0;
    
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
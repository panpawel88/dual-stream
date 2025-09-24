#pragma once

#include "ISwitchingTrigger.h"

/**
 * A no-operation switching trigger for single video mode.
 * Never triggers any video switches since there's nothing to switch to.
 */
class NoOpSwitchingTrigger : public ISwitchingTrigger {
public:
    NoOpSwitchingTrigger();

    std::optional<size_t> GetTargetVideoIndex() override;
    void Update() override;
    void Reset() override;
    std::string GetName() const override;
};
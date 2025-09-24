#include "NoOpSwitchingTrigger.h"

NoOpSwitchingTrigger::NoOpSwitchingTrigger() {
    // Nothing to initialize
}

std::optional<size_t> NoOpSwitchingTrigger::GetTargetVideoIndex() {
    // Never trigger any switches in single video mode
    return std::nullopt;
}

void NoOpSwitchingTrigger::Update() {
    // Nothing to update - no input processing needed
}

void NoOpSwitchingTrigger::Reset() {
    // Nothing to reset
}

std::string NoOpSwitchingTrigger::GetName() const {
    return "No-Op (Single Video)";
}
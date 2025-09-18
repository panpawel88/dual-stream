#include "ICameraFrameListener.h"
#include "CircularBuffer.h"

// Default implementation for GetPreferredOverflowPolicy
OverflowPolicy ICameraFrameListener::GetPreferredOverflowPolicy() const {
    return OverflowPolicy::DROP_OLDEST;
}
#pragma once

#include "../D3D11SimpleRenderPass.h"

/**
 * Motion blur render pass that applies a simple directional blur effect.
 * 
 * Supported parameters:
 * - blur_strength: Strength of the blur effect (0.0 - 1.0)
 * - sample_count: Number of samples for the blur (1 - 32)
 */
class MotionBlurPass : public D3D11SimpleRenderPass {
public:
    MotionBlurPass() : D3D11SimpleRenderPass("MotionBlur") {}
    
    // Uses the built-in "MotionBlur" shader from D3D11SimpleRenderPass
    // Parameters are automatically handled by the base class constant buffer system
};
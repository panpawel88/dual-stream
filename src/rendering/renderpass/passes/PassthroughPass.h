#pragma once

#include "../D3D11SimpleRenderPass.h"

/**
 * Simple passthrough render pass that copies input to output without modification.
 * Useful for testing the render pass pipeline.
 */
class PassthroughPass : public D3D11SimpleRenderPass {
public:
    PassthroughPass() : D3D11SimpleRenderPass("Passthrough") {}
    
    // Uses the built-in "Passthrough" shader from D3D11SimpleRenderPass
    // No additional configuration needed
};
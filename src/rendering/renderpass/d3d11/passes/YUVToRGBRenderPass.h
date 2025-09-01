#pragma once

#include "../D3D11SimpleRenderPass.h"

/**
 * YUV to RGB color space conversion render pass.
 * Converts YUV input textures to RGB format using BT.709 color space.
 * Supports both single texture (Y only) and dual texture (Y+UV for NV12) inputs.
 * This pass is automatically inserted by the pipeline when YUV input is detected.
 */
class YUVToRGBRenderPass : public D3D11SimpleRenderPass {
public:
    YUVToRGBRenderPass() : D3D11SimpleRenderPass("YUVToRGB") {}
    
    // Uses the built-in "YUVToRGB" shader from D3D11SimpleRenderPass
    // Automatically handles both single and dual texture YUV formats
};
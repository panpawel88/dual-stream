#pragma once

#include <d3d11.h>

/**
 * Common context data shared between all graphics APIs
 */
struct RenderPassContextBase {
    float deltaTime;        // Time since last frame
    float totalTime;        // Total elapsed time
    int frameNumber;        // Frame counter
    int inputWidth;         // Input texture width
    int inputHeight;        // Input texture height
    bool isYUV;             // True if input texture is in YUV format
};

/**
 * DirectX 11 specific render pass context
 */
struct D3D11RenderPassContext : public RenderPassContextBase {
    ID3D11DeviceContext* deviceContext;
    ID3D11ShaderResourceView* uvSRV;  // Second texture plane for NV12 (UV), nullptr for single-plane
    DXGI_FORMAT textureFormat;  // Exact texture format (NV12, BGRA8, etc.)
};

// Type aliases for backward compatibility
using RenderPassContext = D3D11RenderPassContext;
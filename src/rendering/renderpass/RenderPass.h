#pragma once

#include "IRenderPass.h"
#include "RenderPassContext.h"
#include <d3d11.h>
#include <wrl/client.h>

using Microsoft::WRL::ComPtr;

// Forward declarations
class RenderPassConfig;

/**
 * DirectX 11 specific render pass base class
 * Extends the API-agnostic interface with D3D11 functionality
 */
class D3D11RenderPass : public IRenderPass {
public:
    D3D11RenderPass(const std::string& name) : IRenderPass(name) {}
    virtual ~D3D11RenderPass() = default;

    /**
     * Initialize the render pass with D3D11 device
     * @param device D3D11 device for resource creation
     * @param config Configuration for the pass
     * @return true on success
     */
    virtual bool Initialize(ID3D11Device* device, const RenderPassConfig& config) = 0;
    
    /**
     * Execute the render pass
     * @param context D3D11 rendering context with device and timing info
     * @param inputSRV Input texture to process
     * @param outputRTV Output render target
     * @return true on success
     */
    virtual bool Execute(const D3D11RenderPassContext& context,
                        ID3D11ShaderResourceView* inputSRV,
                        ID3D11RenderTargetView* outputRTV) = 0;
};

// Type alias for backward compatibility
using RenderPass = D3D11RenderPass;
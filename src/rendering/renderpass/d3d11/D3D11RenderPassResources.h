#pragma once

#include <d3d11.h>
#include <wrl/client.h>
#include <memory>

using Microsoft::WRL::ComPtr;

/**
 * Shared resource manager for D3D11 render passes
 * Provides common resources to avoid duplication and improve performance
 */
class D3D11RenderPassResources {
public:
    // Singleton access
    static D3D11RenderPassResources& GetInstance();
    
    // Initialize with D3D11 device
    bool Initialize(ID3D11Device* device);
    void Cleanup();
    
    // Resource access
    ID3D11Buffer* GetFullscreenQuadVertexBuffer() const { return m_fullscreenVertexBuffer.Get(); }
    ID3D11Buffer* GetFullscreenQuadIndexBuffer() const { return m_fullscreenIndexBuffer.Get(); }
    
    // Sampler states
    ID3D11SamplerState* GetLinearClampSampler() const { return m_linearClampSampler.Get(); }
    ID3D11SamplerState* GetPointClampSampler() const { return m_pointClampSampler.Get(); }
    
    // Blend states
    ID3D11BlendState* GetNoBlendState() const { return m_noBlendState.Get(); }
    ID3D11BlendState* GetAlphaBlendState() const { return m_alphaBlendState.Get(); }
    ID3D11BlendState* GetAdditiveBlendState() const { return m_additiveBlendState.Get(); }
    
    // Rasterizer states
    ID3D11RasterizerState* GetNoCullRasterizer() const { return m_noCullRasterizer.Get(); }
    ID3D11RasterizerState* GetBackCullRasterizer() const { return m_backCullRasterizer.Get(); }
    
    // Vertex structure information
    static constexpr UINT GetFullscreenQuadVertexStride() { return sizeof(float) * 5; } // pos(3) + tex(2)
    static constexpr UINT GetFullscreenQuadIndexCount() { return 6; }
    
private:
    D3D11RenderPassResources() = default;
    ~D3D11RenderPassResources() = default;
    D3D11RenderPassResources(const D3D11RenderPassResources&) = delete;
    D3D11RenderPassResources& operator=(const D3D11RenderPassResources&) = delete;
    
    bool CreateFullscreenQuad(ID3D11Device* device);
    bool CreateSamplerStates(ID3D11Device* device);
    bool CreateBlendStates(ID3D11Device* device);
    bool CreateRasterizerStates(ID3D11Device* device);
    
    // Geometry resources
    ComPtr<ID3D11Buffer> m_fullscreenVertexBuffer;
    ComPtr<ID3D11Buffer> m_fullscreenIndexBuffer;
    
    // Sampler states
    ComPtr<ID3D11SamplerState> m_linearClampSampler;
    ComPtr<ID3D11SamplerState> m_pointClampSampler;
    
    // Blend states
    ComPtr<ID3D11BlendState> m_noBlendState;
    ComPtr<ID3D11BlendState> m_alphaBlendState;
    ComPtr<ID3D11BlendState> m_additiveBlendState;
    
    // Rasterizer states
    ComPtr<ID3D11RasterizerState> m_noCullRasterizer;
    ComPtr<ID3D11RasterizerState> m_backCullRasterizer;
    
    bool m_initialized = false;
};
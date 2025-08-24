#pragma once

#include <d3d11.h>
#include <dxgi.h>
#include <wrl/client.h>
#include "IRenderer.h"

using Microsoft::WRL::ComPtr;

class D3D11Renderer : public IRenderer {
public:
    D3D11Renderer();
    ~D3D11Renderer();
    
    // IRenderer interface implementation
    bool Initialize(HWND hwnd, int width, int height) override;
    void Cleanup() override;
    bool Present(const RenderTexture& texture) override;
    bool Resize(int width, int height) override;
    bool IsInitialized() const override { return m_initialized; }
    RendererType GetRendererType() const override { return RendererType::DirectX11; }
    bool SupportsCudaInterop() const override { return false; } // D3D11 doesn't support CUDA interop
    
    // D3D11-specific methods (for downcasting)
    ID3D11Device* GetDevice() const { return m_device.Get(); }
    ID3D11DeviceContext* GetContext() const { return m_context.Get(); }
    
private:
    bool m_initialized;
    HWND m_hwnd;
    int m_width;
    int m_height;
    
    // DirectX 11 components
    ComPtr<ID3D11Device> m_device;
    ComPtr<ID3D11DeviceContext> m_context;
    ComPtr<IDXGISwapChain> m_swapChain;
    ComPtr<ID3D11RenderTargetView> m_renderTargetView;
    ComPtr<ID3D11Texture2D> m_backBuffer;
    
    // Rendering pipeline
    ComPtr<ID3D11VertexShader> m_vertexShader;
    ComPtr<ID3D11PixelShader> m_pixelShaderRGB;
    ComPtr<ID3D11PixelShader> m_pixelShaderYUV;
    ComPtr<ID3D11InputLayout> m_inputLayout;
    ComPtr<ID3D11Buffer> m_vertexBuffer;
    ComPtr<ID3D11Buffer> m_indexBuffer;
    ComPtr<ID3D11SamplerState> m_samplerState;
    ComPtr<ID3D11BlendState> m_blendState;
    ComPtr<ID3D11RasterizerState> m_rasterizerState;
    
    // Current frame resources
    ComPtr<ID3D11ShaderResourceView> m_currentFrameSRV;
    ComPtr<ID3D11ShaderResourceView> m_currentFrameUVSRV; // For UV plane in NV12
    
    // Initialization helpers
    bool CreateDeviceAndSwapChain();
    bool CreateRenderTarget();
    bool CreateShaders();
    bool CreateGeometry();
    bool CreateStates();
    
    // Rendering helpers
    bool PresentD3D11Texture(const RenderTexture& texture);
    bool PresentSoftwareTexture(const RenderTexture& texture);
    bool UpdateFrameTexture(ID3D11Texture2D* videoTexture, bool isYUV, DXGI_FORMAT format);
    void SetupRenderState(bool isYUV);
    void DrawQuad();
    
    void Reset();
};

// Vertex structure for full-screen quad
struct QuadVertex {
    float position[3];  // x, y, z
    float texCoord[2];  // u, v
};
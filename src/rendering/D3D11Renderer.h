#pragma once

#include <d3d11.h>
#include <dxgi.h>
#include <wrl/client.h>
#include "IRenderer.h"
#include "renderpass/RenderPassPipeline.h"
#include <memory>


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
    bool CaptureFramebuffer(uint8_t* outputBuffer, size_t bufferSize, int& width, int& height) override;
    
    // D3D11-specific methods
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
    
    // Render pass pipeline
    std::unique_ptr<RenderPassPipeline> m_renderPassPipeline;
    int m_frameNumber;
    float m_totalTime;
    
    // Rendering configuration
    int m_vsyncMode;         // 0 = off, 1 = on, 2 = adaptive
    int m_bufferCount;       // Number of back buffers (1-3)
    DXGI_SWAP_EFFECT m_swapEffect; // Presentation mode
    
    bool CreateDeviceAndSwapChain();
    bool CreateRenderTarget();
    bool CreateShaders();
    bool CreateGeometry();
    bool CreateStates();
    
    bool PresentD3D11Texture(const RenderTexture& texture);
    bool PresentSoftwareTexture(const RenderTexture& texture);
    bool PresentD3D11TextureDirect(ID3D11ShaderResourceView* inputSRV, bool isYUV, int contentWidth, int contentHeight);
    bool UpdateFrameTexture(ID3D11Texture2D* texture, bool isYUV, DXGI_FORMAT format);
    void SetupRenderState(bool isYUV);
    void DrawQuad();
    void DrawAdjustedQuad(int contentWidth, int contentHeight, int textureWidth, int textureHeight);
    
    void Reset();
};

// Vertex structure for full-screen quad
struct QuadVertex {
    float position[3];  // x, y, z
    float texCoord[2];  // u, v
};
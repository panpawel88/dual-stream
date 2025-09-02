#pragma once
#include "OverlayRenderPass.h"
#include "renderpass/RenderPass.h"
#include "renderpass/RenderPassContext.h"
#include <d3d11.h>
#include <wrl/client.h>

class D3D11OverlayRenderPass : public OverlayRenderPass, public D3D11RenderPass {
public:
    D3D11OverlayRenderPass();
    ~D3D11OverlayRenderPass() override;
    
    // D3D11RenderPass interface - DirectX 11 specific
    bool Initialize(ID3D11Device* device, const RenderPassConfig& config) override;
    bool Execute(const D3D11RenderPassContext& context,
                ID3D11ShaderResourceView* inputSRV,
                ID3D11RenderTargetView* outputRTV) override;
    void Cleanup() override;
    
    // IRenderPass interface
    PassType GetType() const override { return PassType::Simple; }
    void UpdateParameters(const std::map<std::string, RenderPassParameter>& parameters) override;
    
protected:
    // OverlayRenderPass abstract methods - D3D11 implementations
    bool InitializeImGuiBackend() override;
    void CleanupImGuiBackend() override;
    void BeginImGuiFrame() override;
    void EndImGuiFrame() override;
    
private:
    Microsoft::WRL::ComPtr<ID3D11Device> m_device;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> m_context;
    
    // Passthrough rendering (copy input to output, then overlay ImGui)
    Microsoft::WRL::ComPtr<ID3D11VertexShader> m_vertexShader;
    Microsoft::WRL::ComPtr<ID3D11PixelShader> m_pixelShader;
    Microsoft::WRL::ComPtr<ID3D11Buffer> m_vertexBuffer;
    Microsoft::WRL::ComPtr<ID3D11InputLayout> m_inputLayout;
    Microsoft::WRL::ComPtr<ID3D11SamplerState> m_samplerState;
    Microsoft::WRL::ComPtr<ID3D11BlendState> m_blendState;
    Microsoft::WRL::ComPtr<ID3D11RasterizerState> m_rasterizerState;
    
    bool InitializePassthroughShaders();
    void CleanupPassthroughShaders();
};
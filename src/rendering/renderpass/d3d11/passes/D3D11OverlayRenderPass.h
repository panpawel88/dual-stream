#pragma once
#include "../../OverlayRenderPass.h"
#include "../D3D11SimpleRenderPass.h"
#include "../../RenderPassContext.h"
#include <d3d11.h>
#include <wrl/client.h>
#include <string>

class D3D11OverlayRenderPass : public OverlayRenderPass, public D3D11SimpleRenderPass {
public:
    D3D11OverlayRenderPass();
    ~D3D11OverlayRenderPass() override;
    
    // D3D11SimpleRenderPass interface - DirectX 11 specific
    bool Initialize(ID3D11Device* device, const RenderPassConfig& config) override;
    bool Initialize(ID3D11Device* device, const RenderPassConfig& config, void* hwnd);
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
    
    // D3D11SimpleRenderPass virtual method overrides
    std::string GetPixelShaderSource() const override;
    
private:
    // Overlay-specific blend state for ImGui rendering
    Microsoft::WRL::ComPtr<ID3D11BlendState> m_overlayBlendState;
    
    // Helper methods
    bool CreateOverlayBlendState(ID3D11Device* device);
};
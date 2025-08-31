#pragma once

#include "IRenderPassPipeline.h"
#include "RenderPass.h"
#include <d3d11.h>
#include <wrl/client.h>
#include <vector>
#include <memory>
#include <string>

using Microsoft::WRL::ComPtr;

// Forward declaration
class RenderPassConfig;

/**
 * DirectX 11 specific render pass pipeline implementation
 * Manages a pipeline of D3D11 render passes, handling texture allocation and pass chaining
 */
class D3D11RenderPassPipeline : public IRenderPassPipeline {
public:
    D3D11RenderPassPipeline();
    ~D3D11RenderPassPipeline();

    /**
     * Initialize the pipeline with D3D11 device
     * @param device D3D11 device for resource creation
     * @return true on success
     */
    bool Initialize(ID3D11Device* device);
    
    // IRenderPassPipeline interface
    void Cleanup() override;
    void AddPass(std::unique_ptr<IRenderPass> pass) override;
    void SetEnabled(bool enabled) override { m_enabled = enabled; }
    bool IsEnabled() const override { return m_enabled; }
    bool SetPassEnabled(const std::string& passName, bool enabled) override;
    IRenderPass* GetPass(const std::string& passName) const override;
    size_t GetPassCount() const override { return m_passes.size(); }
    bool UpdatePassParameters(const std::string& passName, 
                             const std::map<std::string, RenderPassParameter>& parameters) override;
    
    // D3D11-specific methods
    void AddD3D11Pass(std::unique_ptr<D3D11RenderPass> pass);
    
    /**
     * Execute the entire pipeline
     * @param context Rendering context
     * @param inputSRV Input texture to process
     * @param outputRTV Final output render target
     * @return true on success
     */
    bool Execute(const D3D11RenderPassContext& context,
                ID3D11ShaderResourceView* inputSRV,
                ID3D11RenderTargetView* outputRTV);

private:
    /**
     * Ensure intermediate textures are allocated and sized correctly
     */
    bool EnsureIntermediateTextures(int width, int height);
    
    /**
     * Create intermediate render targets
     */
    bool CreateIntermediateTexture(int width, int height, DXGI_FORMAT format,
                                  ComPtr<ID3D11Texture2D>& texture,
                                  ComPtr<ID3D11ShaderResourceView>& srv,
                                  ComPtr<ID3D11RenderTargetView>& rtv);
    
    /**
     * Perform direct copy from input to output (when pipeline is disabled)
     */
    bool DirectCopy(ID3D11DeviceContext* context,
                   ID3D11ShaderResourceView* inputSRV,
                   ID3D11RenderTargetView* outputRTV,
                   int width, int height);

private:
    ComPtr<ID3D11Device> m_device;
    std::vector<std::unique_ptr<D3D11RenderPass>> m_passes;
    bool m_enabled;
    
    // Cached YUV conversion pass for dynamic insertion
    std::unique_ptr<D3D11RenderPass> m_yuvToRgbPass;
    
    // Intermediate textures for pass chaining (ping-pong buffers)
    ComPtr<ID3D11Texture2D> m_intermediateTexture[2];
    ComPtr<ID3D11ShaderResourceView> m_intermediateSRV[2];
    ComPtr<ID3D11RenderTargetView> m_intermediateRTV[2];
    
    // Current texture dimensions
    int m_textureWidth;
    int m_textureHeight;
    DXGI_FORMAT m_textureFormat;
    
    // Resources for direct copy
    ComPtr<ID3D11VertexShader> m_copyVertexShader;
    ComPtr<ID3D11PixelShader> m_copyPixelShader;
    ComPtr<ID3D11InputLayout> m_copyInputLayout;
    ComPtr<ID3D11Buffer> m_copyVertexBuffer;
    ComPtr<ID3D11Buffer> m_copyIndexBuffer;
    ComPtr<ID3D11SamplerState> m_copySamplerState;
    ComPtr<ID3D11BlendState> m_copyBlendState;
    ComPtr<ID3D11RasterizerState> m_copyRasterizerState;
    
    bool CreateCopyResources();
};

// Type alias for backward compatibility
using RenderPassPipeline = D3D11RenderPassPipeline;
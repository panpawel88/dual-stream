#pragma once

#include "RenderPass.h"
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>

using Microsoft::WRL::ComPtr;

/**
 * Simple render pass implementation using vertex + pixel shaders
 */
class D3D11SimpleRenderPass : public RenderPass {
public:
    D3D11SimpleRenderPass(const std::string& name) : RenderPass(name) {}
    virtual ~D3D11SimpleRenderPass();

    // RenderPass interface
    PassType GetType() const override { return PassType::Simple; }
    bool Initialize(ID3D11Device* device, const RenderPassConfig& config) override;
    void Cleanup() override;
    bool Execute(const RenderPassContext& context,
                ID3D11ShaderResourceView* inputSRV,
                ID3D11RenderTargetView* outputRTV) override;
    void UpdateParameters(const std::map<std::string, RenderPassParameter>& parameters) override;

protected:
    // Shader loading
    bool LoadVertexShader(ID3D11Device* device, const std::string& shaderPath);
    bool LoadPixelShader(ID3D11Device* device, const std::string& shaderPath);
    bool LoadShadersFromResource(ID3D11Device* device, const std::string& shaderName);
    
    // Constant buffer management
    bool CreateConstantBuffer(ID3D11Device* device, size_t size);
    bool UpdateConstantBuffer(ID3D11DeviceContext* context);
    void PackParameters();
    
    // Rendering
    bool CreateFullscreenQuad(ID3D11Device* device);
    void RenderFullscreenQuad(ID3D11DeviceContext* context);

protected:
    // D3D11 resources
    ComPtr<ID3D11Device> m_device;
    ComPtr<ID3D11VertexShader> m_vertexShader;
    ComPtr<ID3D11PixelShader> m_pixelShader;
    ComPtr<ID3D11InputLayout> m_inputLayout;
    ComPtr<ID3D11Buffer> m_vertexBuffer;
    ComPtr<ID3D11Buffer> m_indexBuffer;
    ComPtr<ID3D11Buffer> m_constantBuffer;
    ComPtr<ID3D11SamplerState> m_samplerState;
    ComPtr<ID3D11BlendState> m_blendState;
    ComPtr<ID3D11RasterizerState> m_rasterizerState;
    
    // Parameter management
    std::map<std::string, RenderPassParameter> m_parameters;
    std::vector<uint8_t> m_constantBufferData;
    size_t m_constantBufferSize;
    bool m_constantBufferDirty;
    
    // Shader paths
    std::string m_vertexShaderPath;
    std::string m_pixelShaderPath;
    std::string m_shaderName; // For built-in shaders

private:
    bool CreateRenderStates(ID3D11Device* device);
    HRESULT CompileShaderFromFile(const std::string& filename, const std::string& entryPoint, 
                                 const std::string& profile, ID3DBlob** blob);
    HRESULT CompileShaderFromString(const std::string& shaderCode, const std::string& entryPoint,
                                   const std::string& profile, ID3DBlob** blob);
};

// Vertex structure for fullscreen quad
struct RenderPassVertex {
    float position[3];  // x, y, z
    float texCoord[2];  // u, v
};
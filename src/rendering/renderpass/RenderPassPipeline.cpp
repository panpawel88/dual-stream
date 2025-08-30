#include "RenderPassPipeline.h"
#include "RenderPassConfig.h"
#include "passes/YUVToRGBRenderPass.h"
#include "core/Logger.h"
#include <d3dcompiler.h>

// Simple copy shaders for direct passthrough
const char* g_copyVertexShaderSource = R"(
struct VS_INPUT {
    float3 pos : POSITION;
    float2 tex : TEXCOORD0;
};

struct VS_OUTPUT {
    float4 pos : SV_POSITION;
    float2 tex : TEXCOORD0;
};

VS_OUTPUT main(VS_INPUT input) {
    VS_OUTPUT output;
    output.pos = float4(input.pos, 1.0f);
    output.tex = input.tex;
    return output;
}
)";

const char* g_copyPixelShaderSource = R"(
Texture2D inputTexture : register(t0);
SamplerState inputSampler : register(s0);

struct PS_INPUT {
    float4 pos : SV_POSITION;
    float2 tex : TEXCOORD0;
};

float4 main(PS_INPUT input) : SV_TARGET {
    return inputTexture.Sample(inputSampler, input.tex);
}
)";

RenderPassPipeline::RenderPassPipeline()
    : m_enabled(false)
    , m_textureWidth(0)
    , m_textureHeight(0)
    , m_textureFormat(DXGI_FORMAT_R8G8B8A8_UNORM) {
}

RenderPassPipeline::~RenderPassPipeline() {
    Cleanup();
}

bool RenderPassPipeline::Initialize(ID3D11Device* device) {
    m_device = device;
    
    // Create resources for direct copy (when pipeline is disabled)
    if (!CreateCopyResources()) {
        LOG_ERROR("RenderPassPipeline: Failed to create copy resources");
        return false;
    }
    
    LOG_INFO("RenderPassPipeline initialized successfully");
    return true;
}

void RenderPassPipeline::Cleanup() {
    // Clean up all passes
    for (auto& pass : m_passes) {
        if (pass) {
            pass->Cleanup();
        }
    }
    m_passes.clear();
    
    // Clean up dynamically created YUV pass
    if (m_yuvToRgbPass) {
        m_yuvToRgbPass->Cleanup();
        m_yuvToRgbPass.reset();
    }
    
    // Clean up copy resources
    m_copyRasterizerState.Reset();
    m_copyBlendState.Reset();
    m_copySamplerState.Reset();
    m_copyIndexBuffer.Reset();
    m_copyVertexBuffer.Reset();
    m_copyInputLayout.Reset();
    m_copyPixelShader.Reset();
    m_copyVertexShader.Reset();
    
    // Clean up intermediate textures
    for (int i = 0; i < 2; i++) {
        m_intermediateRTV[i].Reset();
        m_intermediateSRV[i].Reset();
        m_intermediateTexture[i].Reset();
    }
    
    m_device.Reset();
}

void RenderPassPipeline::AddPass(std::unique_ptr<RenderPass> pass) {
    if (pass) {
        m_passes.push_back(std::move(pass));
        LOG_INFO("RenderPassPipeline: Added pass '", m_passes.back()->GetName(), "'");
    }
}

bool RenderPassPipeline::Execute(const RenderPassContext& context,
                                ID3D11ShaderResourceView* inputSRV,
                                ID3D11RenderTargetView* outputRTV) {
    if (!m_enabled || m_passes.empty()) {
        // Pipeline disabled or no passes - direct copy input to output
        return DirectCopy(context.deviceContext, inputSRV, outputRTV, 
                         context.inputWidth, context.inputHeight);
    }
    
    // Build list of enabled passes, with dynamic YUV conversion if needed
    std::vector<RenderPass*> enabledPasses;
    
    // Check if we need YUV to RGB conversion at the beginning
    bool needsYuvConversion = context.isYUV;
    if (needsYuvConversion) {
        // Ensure YUV conversion pass exists and is initialized
        if (!m_yuvToRgbPass) {
            m_yuvToRgbPass = std::make_unique<YUVToRGBRenderPass>();
            
            // Initialize with config specifying YUVToRGB shader
            RenderPassConfig yuvConfig;
            yuvConfig.SetString("shader", "YUVToRGB");
            if (!m_yuvToRgbPass->Initialize(m_device.Get(), yuvConfig)) {
                LOG_ERROR("RenderPassPipeline: Failed to initialize YUV to RGB conversion pass");
                needsYuvConversion = false; // Fall back to direct copy
            } else {
                LOG_DEBUG("RenderPassPipeline: Dynamically created YUV to RGB conversion pass");
            }
        }
        
        if (m_yuvToRgbPass && needsYuvConversion) {
            enabledPasses.push_back(m_yuvToRgbPass.get());
        }
    }
    
    // Add user-configured passes
    for (auto& pass : m_passes) {
        if (pass && pass->IsEnabled()) {
            enabledPasses.push_back(pass.get());
        }
    }
    
    if (enabledPasses.empty()) {
        // No enabled passes - direct copy
        return DirectCopy(context.deviceContext, inputSRV, outputRTV,
                         context.inputWidth, context.inputHeight);
    }
    
    // Ensure intermediate textures are allocated
    if (!EnsureIntermediateTextures(context.inputWidth, context.inputHeight)) {
        LOG_ERROR("RenderPassPipeline: Failed to ensure intermediate textures");
        return false;
    }
    
    // Execute passes
    bool success = true;
    ID3D11ShaderResourceView* currentInput = inputSRV;
    ID3D11RenderTargetView* currentOutput = nullptr;
    
    for (size_t i = 0; i < enabledPasses.size(); i++) {
        RenderPass* pass = enabledPasses[i];
        
        // Determine output target
        if (i == enabledPasses.size() - 1) {
            // Last pass - render to final output
            currentOutput = outputRTV;
        } else {
            // Intermediate pass - render to intermediate texture
            int bufferIndex = i % 2;
            currentOutput = m_intermediateRTV[bufferIndex].Get();
        }
        
        // Execute pass
        LOG_DEBUG("RenderPassPipeline: Executing pass '", pass->GetName(), "'");
        if (!pass->Execute(context, currentInput, currentOutput)) {
            LOG_ERROR("RenderPassPipeline: Pass '", pass->GetName(), "' failed");
            LOG_ERROR("Context: isYUV=", context.isYUV, ", uvSRV=", (context.uvSRV ? "available" : "null"));
            success = false;
            break;
        }
        
        // Set up input for next pass (if not the last pass)
        if (i < enabledPasses.size() - 1) {
            int bufferIndex = i % 2;
            currentInput = m_intermediateSRV[bufferIndex].Get();
        }
    }
    
    return success;
}

bool RenderPassPipeline::SetPassEnabled(const std::string& passName, bool enabled) {
    for (auto& pass : m_passes) {
        if (pass && pass->GetName() == passName) {
            pass->SetEnabled(enabled);
            LOG_INFO("RenderPassPipeline: Pass '", passName, "' ", (enabled ? "enabled" : "disabled"));
            return true;
        }
    }
    
    LOG_WARNING("RenderPassPipeline: Pass '", passName, "' not found");
    return false;
}

RenderPass* RenderPassPipeline::GetPass(const std::string& passName) const {
    for (const auto& pass : m_passes) {
        if (pass && pass->GetName() == passName) {
            return pass.get();
        }
    }
    return nullptr;
}

bool RenderPassPipeline::UpdatePassParameters(const std::string& passName,
                                             const std::map<std::string, RenderPassParameter>& parameters) {
    RenderPass* pass = GetPass(passName);
    if (!pass) {
        LOG_WARNING("RenderPassPipeline: Pass '", passName, "' not found for parameter update");
        return false;
    }
    
    pass->UpdateParameters(parameters);
    LOG_DEBUG("RenderPassPipeline: Updated parameters for pass '", passName, "'");
    return true;
}

bool RenderPassPipeline::EnsureIntermediateTextures(int width, int height) {
    // Check if we need to recreate textures
    if (m_textureWidth == width && m_textureHeight == height && 
        m_intermediateTexture[0] && m_intermediateTexture[1]) {
        return true; // Already correct size
    }
    
    // Clean up old textures
    for (int i = 0; i < 2; i++) {
        m_intermediateRTV[i].Reset();
        m_intermediateSRV[i].Reset();
        m_intermediateTexture[i].Reset();
    }
    
    // Create new textures
    for (int i = 0; i < 2; i++) {
        if (!CreateIntermediateTexture(width, height, m_textureFormat,
                                      m_intermediateTexture[i],
                                      m_intermediateSRV[i],
                                      m_intermediateRTV[i])) {
            LOG_ERROR("RenderPassPipeline: Failed to create intermediate texture ", i);
            return false;
        }
    }
    
    m_textureWidth = width;
    m_textureHeight = height;
    
    LOG_DEBUG("RenderPassPipeline: Created intermediate textures (", width, "x", height, ")");
    return true;
}

bool RenderPassPipeline::CreateIntermediateTexture(int width, int height, DXGI_FORMAT format,
                                                  ComPtr<ID3D11Texture2D>& texture,
                                                  ComPtr<ID3D11ShaderResourceView>& srv,
                                                  ComPtr<ID3D11RenderTargetView>& rtv) {
    // Create texture
    D3D11_TEXTURE2D_DESC textureDesc = {};
    textureDesc.Width = width;
    textureDesc.Height = height;
    textureDesc.MipLevels = 1;
    textureDesc.ArraySize = 1;
    textureDesc.Format = format;
    textureDesc.SampleDesc.Count = 1;
    textureDesc.Usage = D3D11_USAGE_DEFAULT;
    textureDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
    
    if (FAILED(m_device->CreateTexture2D(&textureDesc, nullptr, &texture))) {
        return false;
    }
    
    // Create SRV
    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format = format;
    srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MipLevels = 1;
    
    if (FAILED(m_device->CreateShaderResourceView(texture.Get(), &srvDesc, &srv))) {
        return false;
    }
    
    // Create RTV
    D3D11_RENDER_TARGET_VIEW_DESC rtvDesc = {};
    rtvDesc.Format = format;
    rtvDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
    
    return SUCCEEDED(m_device->CreateRenderTargetView(texture.Get(), &rtvDesc, &rtv));
}

bool RenderPassPipeline::DirectCopy(ID3D11DeviceContext* context,
                                   ID3D11ShaderResourceView* inputSRV,
                                   ID3D11RenderTargetView* outputRTV,
                                   int width, int height) {
    // Set render target
    context->OMSetRenderTargets(1, &outputRTV, nullptr);
    
    // Set viewport
    D3D11_VIEWPORT viewport = {};
    viewport.Width = static_cast<float>(width);
    viewport.Height = static_cast<float>(height);
    viewport.MinDepth = 0.0f;
    viewport.MaxDepth = 1.0f;
    context->RSSetViewports(1, &viewport);
    
    // Set shaders
    context->VSSetShader(m_copyVertexShader.Get(), nullptr, 0);
    context->PSSetShader(m_copyPixelShader.Get(), nullptr, 0);
    
    // Set input texture
    if (inputSRV) {
        context->PSSetShaderResources(0, 1, &inputSRV);
    }
    
    // Set sampler state
    context->PSSetSamplers(0, 1, m_copySamplerState.GetAddressOf());
    
    // Set render states
    float blendFactor[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    context->OMSetBlendState(m_copyBlendState.Get(), blendFactor, 0xFFFFFFFF);
    context->RSSetState(m_copyRasterizerState.Get());
    
    // Set vertex buffer
    UINT stride = sizeof(float) * 5; // 3 position + 2 texcoord
    UINT offset = 0;
    context->IASetVertexBuffers(0, 1, m_copyVertexBuffer.GetAddressOf(), &stride, &offset);
    
    // Set index buffer
    context->IASetIndexBuffer(m_copyIndexBuffer.Get(), DXGI_FORMAT_R32_UINT, 0);
    
    // Set input layout
    context->IASetInputLayout(m_copyInputLayout.Get());
    
    // Set primitive topology
    context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    
    // Draw
    context->DrawIndexed(6, 0, 0);
    
    // Unbind resources
    ID3D11ShaderResourceView* nullSRV = nullptr;
    context->PSSetShaderResources(0, 1, &nullSRV);
    
    return true;
}

bool RenderPassPipeline::CreateCopyResources() {
    // Compile copy shaders
    UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;
#ifdef _DEBUG
    flags |= D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#endif
    
    ComPtr<ID3DBlob> vsBlob;
    ComPtr<ID3DBlob> errorBlob;
    
    // Compile vertex shader
    HRESULT hr = D3DCompile(g_copyVertexShaderSource, strlen(g_copyVertexShaderSource),
                           nullptr, nullptr, nullptr, "main", "vs_5_0", flags, 0, &vsBlob, &errorBlob);
    
    if (FAILED(hr)) {
        if (errorBlob) {
            LOG_ERROR("Copy vertex shader compilation error: ", static_cast<char*>(errorBlob->GetBufferPointer()));
        }
        return false;
    }
    
    if (FAILED(m_device->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), nullptr, &m_copyVertexShader))) {
        return false;
    }
    
    // Create input layout
    D3D11_INPUT_ELEMENT_DESC layout[] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 }
    };
    
    if (FAILED(m_device->CreateInputLayout(layout, 2, vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), &m_copyInputLayout))) {
        return false;
    }
    
    // Compile pixel shader
    ComPtr<ID3DBlob> psBlob;
    hr = D3DCompile(g_copyPixelShaderSource, strlen(g_copyPixelShaderSource),
                    nullptr, nullptr, nullptr, "main", "ps_5_0", flags, 0, &psBlob, &errorBlob);
    
    if (FAILED(hr)) {
        if (errorBlob) {
            LOG_ERROR("Copy pixel shader compilation error: ", static_cast<char*>(errorBlob->GetBufferPointer()));
        }
        return false;
    }
    
    if (FAILED(m_device->CreatePixelShader(psBlob->GetBufferPointer(), psBlob->GetBufferSize(), nullptr, &m_copyPixelShader))) {
        return false;
    }
    
    // Create fullscreen quad vertex buffer
    struct CopyVertex {
        float position[3];
        float texCoord[2];
    };
    
    CopyVertex vertices[] = {
        { { -1.0f,  1.0f, 0.0f }, { 0.0f, 0.0f } },  // Top-left
        { {  1.0f,  1.0f, 0.0f }, { 1.0f, 0.0f } },  // Top-right
        { {  1.0f, -1.0f, 0.0f }, { 1.0f, 1.0f } },  // Bottom-right
        { { -1.0f, -1.0f, 0.0f }, { 0.0f, 1.0f } }   // Bottom-left
    };
    
    D3D11_BUFFER_DESC bufferDesc = {};
    bufferDesc.Usage = D3D11_USAGE_DEFAULT;
    bufferDesc.ByteWidth = sizeof(vertices);
    bufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    
    D3D11_SUBRESOURCE_DATA initData = {};
    initData.pSysMem = vertices;
    
    if (FAILED(m_device->CreateBuffer(&bufferDesc, &initData, &m_copyVertexBuffer))) {
        return false;
    }
    
    // Create index buffer
    UINT indices[] = { 0, 1, 2, 0, 2, 3 };
    
    bufferDesc.ByteWidth = sizeof(indices);
    bufferDesc.BindFlags = D3D11_BIND_INDEX_BUFFER;
    initData.pSysMem = indices;
    
    if (FAILED(m_device->CreateBuffer(&bufferDesc, &initData, &m_copyIndexBuffer))) {
        return false;
    }
    
    // Create sampler state
    D3D11_SAMPLER_DESC samplerDesc = {};
    samplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
    samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    
    if (FAILED(m_device->CreateSamplerState(&samplerDesc, &m_copySamplerState))) {
        return false;
    }
    
    // Create blend state
    D3D11_BLEND_DESC blendDesc = {};
    blendDesc.RenderTarget[0].BlendEnable = FALSE;
    blendDesc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    
    if (FAILED(m_device->CreateBlendState(&blendDesc, &m_copyBlendState))) {
        return false;
    }
    
    // Create rasterizer state
    D3D11_RASTERIZER_DESC rasterizerDesc = {};
    rasterizerDesc.FillMode = D3D11_FILL_SOLID;
    rasterizerDesc.CullMode = D3D11_CULL_BACK;
    rasterizerDesc.FrontCounterClockwise = FALSE;
    rasterizerDesc.DepthClipEnable = TRUE;
    
    return SUCCEEDED(m_device->CreateRasterizerState(&rasterizerDesc, &m_copyRasterizerState));
}
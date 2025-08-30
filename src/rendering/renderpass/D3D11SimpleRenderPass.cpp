#include "D3D11SimpleRenderPass.h"
#include "RenderPassConfig.h"
#include "core/Logger.h"
#include <fstream>

D3D11SimpleRenderPass::~D3D11SimpleRenderPass() {
    Cleanup();
}

bool D3D11SimpleRenderPass::Initialize(ID3D11Device* device, const RenderPassConfig& config) {
    m_device = device;
    
    // Load shader configuration
    std::string shaderName = config.GetString("shader");
    if (shaderName.empty()) {
        LOG_ERROR("RenderPass '", m_name, "': No shader specified");
        return false;
    }
    
    // Check if it's a built-in shader or file path
    if (shaderName.find('.') == std::string::npos) {
        // Built-in shader
        m_shaderName = shaderName;
        if (!LoadShadersFromResource(device, shaderName)) {
            LOG_ERROR("RenderPass '", m_name, "': Failed to load built-in shader '", shaderName, "'");
            return false;
        }
    } else {
        // External shader file
        m_vertexShaderPath = "src/rendering/shaders/FullscreenQuad.hlsl"; // Standard vertex shader
        m_pixelShaderPath = "src/rendering/shaders/" + shaderName;
        
        if (!LoadVertexShader(device, m_vertexShaderPath) || !LoadPixelShader(device, m_pixelShaderPath)) {
            LOG_ERROR("RenderPass '", m_name, "': Failed to load shader files");
            return false;
        }
    }
    
    // Create fullscreen quad geometry
    if (!CreateFullscreenQuad(device)) {
        LOG_ERROR("RenderPass '", m_name, "': Failed to create fullscreen quad");
        return false;
    }
    
    // Create render states
    if (!CreateRenderStates(device)) {
        LOG_ERROR("RenderPass '", m_name, "': Failed to create render states");
        return false;
    }
    
    // Load parameters from config
    auto parameters = config.GetAllParameters();
    if (!parameters.empty()) {
        UpdateParameters(parameters);
        
        // Create constant buffer if we have parameters
        m_constantBufferSize = std::max(size_t(256), parameters.size() * 16); // Minimum 256 bytes
        if (!CreateConstantBuffer(device, m_constantBufferSize)) {
            LOG_ERROR("RenderPass '", m_name, "': Failed to create constant buffer");
            return false;
        }
    }
    
    LOG_INFO("RenderPass '", m_name, "' initialized successfully");
    return true;
}

void D3D11SimpleRenderPass::Cleanup() {
    m_rasterizerState.Reset();
    m_blendState.Reset();
    m_samplerState.Reset();
    m_constantBuffer.Reset();
    m_indexBuffer.Reset();
    m_vertexBuffer.Reset();
    m_inputLayout.Reset();
    m_pixelShader.Reset();
    m_vertexShader.Reset();
    m_device.Reset();
    
    m_constantBufferData.clear();
    m_parameters.clear();
}

bool D3D11SimpleRenderPass::Execute(const RenderPassContext& context,
                                   ID3D11ShaderResourceView* inputSRV,
                                   ID3D11RenderTargetView* outputRTV) {
    if (!m_enabled || !m_vertexShader || !m_pixelShader) {
        return false;
    }
    
    ID3D11DeviceContext* deviceContext = context.deviceContext;
    
    // Set render target
    deviceContext->OMSetRenderTargets(1, &outputRTV, nullptr);
    
    // Set viewport
    D3D11_VIEWPORT viewport = {};
    viewport.Width = static_cast<float>(context.inputWidth);
    viewport.Height = static_cast<float>(context.inputHeight);
    viewport.MinDepth = 0.0f;
    viewport.MaxDepth = 1.0f;
    deviceContext->RSSetViewports(1, &viewport);
    
    // Set shaders
    deviceContext->VSSetShader(m_vertexShader.Get(), nullptr, 0);
    deviceContext->PSSetShader(m_pixelShader.Get(), nullptr, 0);
    
    // Set input texture
    if (inputSRV) {
        deviceContext->PSSetShaderResources(0, 1, &inputSRV);
    }
    
    // Set sampler state
    if (m_samplerState) {
        deviceContext->PSSetSamplers(0, 1, m_samplerState.GetAddressOf());
    }
    
    // Update and set constant buffer if needed
    if (m_constantBuffer && m_constantBufferDirty) {
        if (!UpdateConstantBuffer(deviceContext)) {
            LOG_ERROR("RenderPass '", m_name, "': Failed to update constant buffer");
            return false;
        }
    }
    
    if (m_constantBuffer) {
        deviceContext->PSSetConstantBuffers(0, 1, m_constantBuffer.GetAddressOf());
    }
    
    // Set render states
    if (m_blendState) {
        float blendFactor[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        deviceContext->OMSetBlendState(m_blendState.Get(), blendFactor, 0xFFFFFFFF);
    }
    
    if (m_rasterizerState) {
        deviceContext->RSSetState(m_rasterizerState.Get());
    }
    
    // Render fullscreen quad
    RenderFullscreenQuad(deviceContext);
    
    // Unbind resources
    ID3D11ShaderResourceView* nullSRV = nullptr;
    deviceContext->PSSetShaderResources(0, 1, &nullSRV);
    
    return true;
}

void D3D11SimpleRenderPass::UpdateParameters(const std::map<std::string, RenderPassParameter>& parameters) {
    m_parameters = parameters;
    PackParameters();
    m_constantBufferDirty = true;
}

bool D3D11SimpleRenderPass::LoadShadersFromResource(ID3D11Device* device, const std::string& shaderName) {
    // For now, we'll compile shaders from embedded strings
    // In a full implementation, these would be resources or files
    
    const char* vertexShaderSource = R"(
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
    
    std::string pixelShaderSource;
    
    if (shaderName == "Passthrough") {
        pixelShaderSource = R"(
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
    } else if (shaderName == "MotionBlur") {
        pixelShaderSource = R"(
Texture2D inputTexture : register(t0);
SamplerState inputSampler : register(s0);

cbuffer MotionBlurParams : register(b0) {
    float blurStrength;
    int sampleCount;
    float2 padding;
};

struct PS_INPUT {
    float4 pos : SV_POSITION;
    float2 tex : TEXCOORD0;
};

float4 main(PS_INPUT input) : SV_TARGET {
    float4 result = float4(0, 0, 0, 0);
    int samples = max(1, sampleCount);
    
    // Simple motion blur by sampling along a direction
    float2 blurDirection = float2(blurStrength * 0.01, 0);
    
    for (int i = 0; i < samples; i++) {
        float offset = (float(i) / float(samples - 1) - 0.5) * 2.0;
        float2 sampleUV = input.tex + blurDirection * offset;
        result += inputTexture.Sample(inputSampler, sampleUV);
    }
    
    return result / float(samples);
}
)";
    } else {
        LOG_ERROR("Unknown built-in shader: ", shaderName);
        return false;
    }
    
    // Compile vertex shader
    ComPtr<ID3DBlob> vsBlob;
    if (FAILED(CompileShaderFromString(vertexShaderSource, "main", "vs_5_0", &vsBlob))) {
        LOG_ERROR("Failed to compile vertex shader for ", shaderName);
        return false;
    }
    
    if (FAILED(device->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), nullptr, &m_vertexShader))) {
        LOG_ERROR("Failed to create vertex shader for ", shaderName);
        return false;
    }
    
    // Create input layout
    D3D11_INPUT_ELEMENT_DESC layout[] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 }
    };
    
    if (FAILED(device->CreateInputLayout(layout, 2, vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), &m_inputLayout))) {
        LOG_ERROR("Failed to create input layout for ", shaderName);
        return false;
    }
    
    // Compile pixel shader
    ComPtr<ID3DBlob> psBlob;
    if (FAILED(CompileShaderFromString(pixelShaderSource, "main", "ps_5_0", &psBlob))) {
        LOG_ERROR("Failed to compile pixel shader for ", shaderName);
        return false;
    }
    
    if (FAILED(device->CreatePixelShader(psBlob->GetBufferPointer(), psBlob->GetBufferSize(), nullptr, &m_pixelShader))) {
        LOG_ERROR("Failed to create pixel shader for ", shaderName);
        return false;
    }
    
    return true;
}

bool D3D11SimpleRenderPass::LoadVertexShader(ID3D11Device* device, const std::string& shaderPath) {
    // TODO: Implement file loading
    LOG_ERROR("External shader file loading not yet implemented");
    return false;
}

bool D3D11SimpleRenderPass::LoadPixelShader(ID3D11Device* device, const std::string& shaderPath) {
    // TODO: Implement file loading
    LOG_ERROR("External shader file loading not yet implemented");
    return false;
}

bool D3D11SimpleRenderPass::CreateConstantBuffer(ID3D11Device* device, size_t size) {
    m_constantBufferSize = ((size + 15) / 16) * 16; // Align to 16 bytes
    m_constantBufferData.resize(m_constantBufferSize, 0);
    
    D3D11_BUFFER_DESC desc = {};
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.ByteWidth = static_cast<UINT>(m_constantBufferSize);
    desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    
    return SUCCEEDED(device->CreateBuffer(&desc, nullptr, &m_constantBuffer));
}

bool D3D11SimpleRenderPass::UpdateConstantBuffer(ID3D11DeviceContext* context) {
    if (!m_constantBuffer || m_constantBufferData.empty()) {
        return false;
    }
    
    context->UpdateSubresource(m_constantBuffer.Get(), 0, nullptr, m_constantBufferData.data(), 0, 0);
    m_constantBufferDirty = false;
    return true;
}

void D3D11SimpleRenderPass::PackParameters() {
    if (m_constantBufferData.empty()) {
        return;
    }
    
    // Simple parameter packing - just store floats and ints sequentially
    size_t offset = 0;
    for (const auto& pair : m_parameters) {
        const std::string& name = pair.first;
        const RenderPassParameter& param = pair.second;
        
        std::visit([&](auto&& value) {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, float>) {
                if (offset + sizeof(float) <= m_constantBufferSize) {
                    memcpy(m_constantBufferData.data() + offset, &value, sizeof(float));
                    offset += sizeof(float);
                }
            } else if constexpr (std::is_same_v<T, int>) {
                if (offset + sizeof(int) <= m_constantBufferSize) {
                    memcpy(m_constantBufferData.data() + offset, &value, sizeof(int));
                    offset += sizeof(int);
                }
            } else if constexpr (std::is_same_v<T, std::array<float, 2>>) {
                if (offset + sizeof(value) <= m_constantBufferSize) {
                    memcpy(m_constantBufferData.data() + offset, value.data(), sizeof(value));
                    offset += sizeof(value);
                }
            } else if constexpr (std::is_same_v<T, std::array<float, 3>>) {
                if (offset + sizeof(value) <= m_constantBufferSize) {
                    memcpy(m_constantBufferData.data() + offset, value.data(), sizeof(value));
                    offset += sizeof(value);
                }
            } else if constexpr (std::is_same_v<T, std::array<float, 4>>) {
                if (offset + sizeof(value) <= m_constantBufferSize) {
                    memcpy(m_constantBufferData.data() + offset, value.data(), sizeof(value));
                    offset += sizeof(value);
                }
            }
        }, param);
        
        // Align to 16 bytes for next parameter
        offset = ((offset + 15) / 16) * 16;
    }
}

bool D3D11SimpleRenderPass::CreateFullscreenQuad(ID3D11Device* device) {
    // Create vertex buffer for fullscreen quad
    RenderPassVertex vertices[] = {
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
    
    if (FAILED(device->CreateBuffer(&bufferDesc, &initData, &m_vertexBuffer))) {
        return false;
    }
    
    // Create index buffer
    UINT indices[] = { 0, 1, 2, 0, 2, 3 };
    
    bufferDesc.ByteWidth = sizeof(indices);
    bufferDesc.BindFlags = D3D11_BIND_INDEX_BUFFER;
    initData.pSysMem = indices;
    
    return SUCCEEDED(device->CreateBuffer(&bufferDesc, &initData, &m_indexBuffer));
}

void D3D11SimpleRenderPass::RenderFullscreenQuad(ID3D11DeviceContext* context) {
    // Set vertex buffer
    UINT stride = sizeof(RenderPassVertex);
    UINT offset = 0;
    context->IASetVertexBuffers(0, 1, m_vertexBuffer.GetAddressOf(), &stride, &offset);
    
    // Set index buffer
    context->IASetIndexBuffer(m_indexBuffer.Get(), DXGI_FORMAT_R32_UINT, 0);
    
    // Set input layout
    context->IASetInputLayout(m_inputLayout.Get());
    
    // Set primitive topology
    context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    
    // Draw
    context->DrawIndexed(6, 0, 0);
}

bool D3D11SimpleRenderPass::CreateRenderStates(ID3D11Device* device) {
    // Create sampler state
    D3D11_SAMPLER_DESC samplerDesc = {};
    samplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
    samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    
    if (FAILED(device->CreateSamplerState(&samplerDesc, &m_samplerState))) {
        return false;
    }
    
    // Create blend state (no blending)
    D3D11_BLEND_DESC blendDesc = {};
    blendDesc.RenderTarget[0].BlendEnable = FALSE;
    blendDesc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    
    if (FAILED(device->CreateBlendState(&blendDesc, &m_blendState))) {
        return false;
    }
    
    // Create rasterizer state
    D3D11_RASTERIZER_DESC rasterizerDesc = {};
    rasterizerDesc.FillMode = D3D11_FILL_SOLID;
    rasterizerDesc.CullMode = D3D11_CULL_BACK;
    rasterizerDesc.FrontCounterClockwise = FALSE;
    rasterizerDesc.DepthClipEnable = TRUE;
    
    return SUCCEEDED(device->CreateRasterizerState(&rasterizerDesc, &m_rasterizerState));
}

HRESULT D3D11SimpleRenderPass::CompileShaderFromString(const std::string& shaderCode, const std::string& entryPoint,
                                                      const std::string& profile, ID3DBlob** blob) {
    UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;
#ifdef _DEBUG
    flags |= D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#endif
    
    ComPtr<ID3DBlob> errorBlob;
    HRESULT hr = D3DCompile(shaderCode.c_str(), shaderCode.length(), nullptr, nullptr, nullptr,
                           entryPoint.c_str(), profile.c_str(), flags, 0, blob, &errorBlob);
    
    if (FAILED(hr) && errorBlob) {
        LOG_ERROR("Shader compilation error: ", static_cast<char*>(errorBlob->GetBufferPointer()));
    }
    
    return hr;
}

HRESULT D3D11SimpleRenderPass::CompileShaderFromFile(const std::string& filename, const std::string& entryPoint,
                                                    const std::string& profile, ID3DBlob** blob) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        LOG_ERROR("Failed to open shader file: ", filename);
        return E_FAIL;
    }
    
    std::string shaderCode((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    return CompileShaderFromString(shaderCode, entryPoint, profile, blob);
}
#include "D3D11SimpleRenderPass.h"
#include "../RenderPassConfig.h"
#include "ShaderLibrary.h"
#include "D3D11RenderPassResources.h"
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
        // Try loading shaders from virtual methods first
        if (!LoadShadersFromSource(device)) {
            LOG_ERROR("RenderPass '", m_name, "': No shader specified and virtual shader sources not provided");
            return false;
        }
    } else {
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
    }
    
    // Initialize shared resources
    if (!InitializeSharedResources(device)) {
        LOG_ERROR("RenderPass '", m_name, "': Failed to initialize shared resources");
        return false;
    }
    
    // Load parameters from config
    auto parameters = config.GetAllParameters();
    size_t virtualConstantBufferSize = GetConstantBufferSize();
    if (!parameters.empty() || virtualConstantBufferSize > 0) {
        // Create constant buffer if we have parameters or virtual size
        if (virtualConstantBufferSize > 0) {
            m_constantBufferSize = virtualConstantBufferSize;
        } else {
            m_constantBufferSize = std::max(size_t(256), parameters.size() * 16); // Minimum 256 bytes
        }
        
        if (!CreateConstantBuffer(device, m_constantBufferSize)) {
            LOG_ERROR("RenderPass '", m_name, "': Failed to create constant buffer");
            return false;
        }

        UpdateParameters(parameters);
    }
    
    LOG_INFO("RenderPass '", m_name, "' initialized successfully");
    return true;
}

void D3D11SimpleRenderPass::Cleanup() {
    m_constantBuffer.Reset();
    m_inputLayout.Reset();
    m_pixelShader.Reset();
    m_vertexShader.Reset();
    m_device.Reset();
    
    m_constantBufferData.clear();
    m_parameters.clear();
    
    // Note: Shared resources (geometry, samplers, render states) are managed by D3D11RenderPassResources
}

bool D3D11SimpleRenderPass::Execute(const D3D11RenderPassContext& context,
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
        
        // Set UV texture if available (for YUV shaders like YUVToRGB)
        if (m_shaderName == "YUVToRGB" || context.isYUV) {
            if (context.uvSRV) {
                deviceContext->PSSetShaderResources(1, 1, &context.uvSRV);
            } else {
                // If no UV texture, bind the Y texture to slot 1 as well for fallback
                deviceContext->PSSetShaderResources(1, 1, &inputSRV);
            }
        }
    }
    
    // Set sampler state (use shared linear clamp sampler)
    auto& sharedResources = D3D11RenderPassResources::GetInstance();
    ID3D11SamplerState* sampler = sharedResources.GetLinearClampSampler();
    if (sampler) {
        deviceContext->PSSetSamplers(0, 1, &sampler);
    }
    
    // Update and set constant buffer if needed
    if (m_constantBuffer) {
        if (GetConstantBufferSize() > 0) {
            // Use virtual method for constant buffer packing
            if (!UpdateConstantBuffer(deviceContext, context)) {
                LOG_ERROR("RenderPass '", m_name, "': Failed to update constant buffer");
                return false;
            }
        } else if (m_constantBufferDirty) {
            // Use traditional parameter packing
            if (!UpdateConstantBuffer(deviceContext)) {
                LOG_ERROR("RenderPass '", m_name, "': Failed to update constant buffer");
                return false;
            }
        }
    }
    
    if (m_constantBuffer) {
        deviceContext->PSSetConstantBuffers(0, 1, m_constantBuffer.GetAddressOf());
    }
    
    // Set render states (use shared states)
    ID3D11BlendState* blendState = sharedResources.GetNoBlendState();
    if (blendState) {
        float blendFactor[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        deviceContext->OMSetBlendState(blendState, blendFactor, 0xFFFFFFFF);
    }
    
    ID3D11RasterizerState* rasterizerState = sharedResources.GetNoCullRasterizer();
    if (rasterizerState) {
        deviceContext->RSSetState(rasterizerState);
    }
    
    // Render fullscreen quad
    RenderFullscreenQuad(deviceContext);
    
    // Unbind resources
    ID3D11ShaderResourceView* nullSRVs[2] = { nullptr, nullptr };
    deviceContext->PSSetShaderResources(0, 2, nullSRVs);
    
    return true;
}

void D3D11SimpleRenderPass::UpdateParameters(const std::map<std::string, RenderPassParameter>& parameters) {
    m_parameters = parameters;
    PackParameters();
    m_constantBufferDirty = true;
}

bool D3D11SimpleRenderPass::LoadShadersFromResource(ID3D11Device* device, const std::string& shaderName) {
    // Use ShaderLibrary for consistent shader management
    std::string vertexShaderSource = ShaderLibrary::GetFullscreenQuadVertexShader();
    std::string pixelShaderSource = ShaderLibrary::GetPixelShaderByName(shaderName);
    
    if (pixelShaderSource.empty()) {
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

bool D3D11SimpleRenderPass::UpdateConstantBuffer(ID3D11DeviceContext* context, const D3D11RenderPassContext& renderContext) {
    if (!m_constantBuffer) {
        return false;
    }
    
    // Try virtual method first
    if (GetConstantBufferSize() > 0) {
        PackConstantBuffer(m_constantBufferData.data(), renderContext);
        context->UpdateSubresource(m_constantBuffer.Get(), 0, nullptr, m_constantBufferData.data(), 0, 0);
        return true;
    }
    
    // Fall back to existing parameter packing
    return UpdateConstantBuffer(context);
}

void D3D11SimpleRenderPass::PackParameters() {
    if (m_constantBufferData.empty()) {
        return;
    }
    
    // Special handling for MotionBlur shader with specific layout
    if (m_shaderName == "MotionBlur") {
        // MotionBlur cbuffer layout:
        // float blurStrength;  (offset 0)
        // int sampleCount;     (offset 4) 
        // float2 padding;      (offset 8, total size 16 bytes)
        
        float blurStrength = 0.02f;  // Default value
        int sampleCount = 8;         // Default value
        
        // Extract parameters
        auto blurParam = m_parameters.find("blur_strength");
        if (blurParam != m_parameters.end() && std::holds_alternative<float>(blurParam->second)) {
            blurStrength = std::get<float>(blurParam->second);
        }
        
        auto countParam = m_parameters.find("sample_count");
        if (countParam != m_parameters.end() && std::holds_alternative<int>(countParam->second)) {
            sampleCount = std::get<int>(countParam->second);
        }
        
        // Pack according to cbuffer layout
        memcpy(m_constantBufferData.data() + 0, &blurStrength, 4);  // float at offset 0
        memcpy(m_constantBufferData.data() + 4, &sampleCount, 4);   // int at offset 4
        // padding at offset 8-15 is zero-filled by default
    } else {
        // Generic parameter packing for other shaders
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
}

bool D3D11SimpleRenderPass::CreateFullscreenQuad(ID3D11Device* device) {
    // Fullscreen quad is now handled by shared resources
    // This method is kept for compatibility but does nothing
    return true;
}

void D3D11SimpleRenderPass::RenderFullscreenQuad(ID3D11DeviceContext* context) {
    auto& sharedResources = D3D11RenderPassResources::GetInstance();
    
    // Set vertex buffer (use shared geometry)
    ID3D11Buffer* vertexBuffer = sharedResources.GetFullscreenQuadVertexBuffer();
    if (vertexBuffer) {
        UINT stride = sharedResources.GetFullscreenQuadVertexStride();
        UINT offset = 0;
        context->IASetVertexBuffers(0, 1, &vertexBuffer, &stride, &offset);
    }
    
    // Set index buffer (use shared geometry)
    ID3D11Buffer* indexBuffer = sharedResources.GetFullscreenQuadIndexBuffer();
    if (indexBuffer) {
        context->IASetIndexBuffer(indexBuffer, DXGI_FORMAT_R16_UINT, 0);
    }
    
    // Set input layout
    context->IASetInputLayout(m_inputLayout.Get());
    
    // Set primitive topology
    context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    
    // Draw
    context->DrawIndexed(sharedResources.GetFullscreenQuadIndexCount(), 0, 0);
}

bool D3D11SimpleRenderPass::InitializeSharedResources(ID3D11Device* device) {
    // Initialize the singleton shared resources if not already done
    auto& sharedResources = D3D11RenderPassResources::GetInstance();
    return sharedResources.Initialize(device);
}

bool D3D11SimpleRenderPass::CreateRenderStates(ID3D11Device* device) {
    // Render states are now handled by shared resources
    // This method is kept for compatibility but does nothing
    return true;
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

bool D3D11SimpleRenderPass::LoadShadersFromSource(ID3D11Device* device) {
    std::string vertexShaderSource = GetVertexShaderSource();
    std::string pixelShaderSource = GetPixelShaderSource();
    
    if (vertexShaderSource.empty() || pixelShaderSource.empty()) {
        return false; // No shader sources provided
    }
    
    // Compile vertex shader
    ComPtr<ID3DBlob> vsBlob;
    if (FAILED(CompileShaderFromString(vertexShaderSource, "main", "vs_5_0", &vsBlob))) {
        LOG_ERROR("Failed to compile vertex shader for ", m_name);
        return false;
    }
    
    if (FAILED(device->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), nullptr, &m_vertexShader))) {
        LOG_ERROR("Failed to create vertex shader for ", m_name);
        return false;
    }
    
    // Create input layout
    D3D11_INPUT_ELEMENT_DESC layout[] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 }
    };
    
    if (FAILED(device->CreateInputLayout(layout, 2, vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), &m_inputLayout))) {
        LOG_ERROR("Failed to create input layout for ", m_name);
        return false;
    }
    
    // Compile pixel shader
    ComPtr<ID3DBlob> psBlob;
    if (FAILED(CompileShaderFromString(pixelShaderSource, "main", "ps_5_0", &psBlob))) {
        LOG_ERROR("Failed to compile pixel shader for ", m_name);
        return false;
    }
    
    if (FAILED(device->CreatePixelShader(psBlob->GetBufferPointer(), psBlob->GetBufferSize(), nullptr, &m_pixelShader))) {
        LOG_ERROR("Failed to create pixel shader for ", m_name);
        return false;
    }
    
    return true;
}

// Default implementations of virtual methods
std::string D3D11SimpleRenderPass::GetVertexShaderSource() const {
    return ShaderLibrary::GetFullscreenQuadVertexShader();
}

std::string D3D11SimpleRenderPass::GetPixelShaderSource() const {
    return ""; // No default pixel shader - must be overridden
}

size_t D3D11SimpleRenderPass::GetConstantBufferSize() const {
    return 0; // No constant buffer by default
}

void D3D11SimpleRenderPass::PackConstantBuffer(uint8_t* buffer, const D3D11RenderPassContext& context) {
    // Default implementation does nothing
}
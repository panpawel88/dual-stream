#include "D3D11OverlayRenderPass.h"
#include "imgui.h"
#include "imgui_impl_dx11.h"
#include "../../../../ui/ImGuiManager.h"
#include "../../../../ui/UIRegistry.h"
#include "../../../../ui/NotificationManager.h"
#include "../../../../core/Logger.h"
#include "../ShaderLibrary.h"
#include "../D3D11RenderPassResources.h"
#include "../../RenderPassConfig.h"
#include <d3dcompiler.h>

D3D11OverlayRenderPass::D3D11OverlayRenderPass() 
    : OverlayRenderPass(), D3D11RenderPass("Overlay") {
}

D3D11OverlayRenderPass::~D3D11OverlayRenderPass() = default;

bool D3D11OverlayRenderPass::Initialize(ID3D11Device* device, const RenderPassConfig& config) {
    return Initialize(device, config, nullptr);
}

bool D3D11OverlayRenderPass::Initialize(ID3D11Device* device, const RenderPassConfig& config, void* hwnd) {
    m_device = device;
    m_device->GetImmediateContext(m_context.GetAddressOf());
    
    // Get window size from config - overlay passes typically use the render target size
    // For now, use default dimensions as overlay will be sized to match the render target
    int width = 1920;  // Will be updated when rendering based on context
    int height = 1080; // Will be updated when rendering based on context
    
    // Call base class common initialization with hwnd
    if (!InitializeCommon(width, height, hwnd)) {
        return false;
    }
    
    // Initialize shared resources for fullscreen quad rendering
    auto& sharedResources = D3D11RenderPassResources::GetInstance();
    if (!sharedResources.Initialize(device)) {
        Logger::GetInstance().Error("Failed to initialize shared resources for overlay");
        return false;
    }
    
    // Initialize passthrough shaders for copying input to output
    if (!InitializePassthroughShaders()) {
        Logger::GetInstance().Error("Failed to initialize passthrough shaders for overlay");
        return false;
    }
    
    return true;
}

bool D3D11OverlayRenderPass::InitializeImGuiBackend() {
    return ImGui_ImplDX11_Init(m_device.Get(), m_context.Get());
}

void D3D11OverlayRenderPass::BeginImGuiFrame() {
    ImGui_ImplDX11_NewFrame();
}

void D3D11OverlayRenderPass::EndImGuiFrame() {
    ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
}

bool D3D11OverlayRenderPass::Execute(const D3D11RenderPassContext& context,
                                    ID3D11ShaderResourceView* inputSRV,
                                    ID3D11RenderTargetView* outputRTV) {
    if (!m_initialized) {
        return false;
    }
    
    // Set render target
    context.deviceContext->OMSetRenderTargets(1, &outputRTV, nullptr);
    
    // Set viewport to output dimensions (window size)
    D3D11_VIEWPORT viewport = {};
    viewport.Width = static_cast<float>(context.outputWidth);
    viewport.Height = static_cast<float>(context.outputHeight);
    viewport.MinDepth = 0.0f;
    viewport.MaxDepth = 1.0f;
    context.deviceContext->RSSetViewports(1, &viewport);
    
    // First, copy input to output (passthrough) if we have input
    if (inputSRV && m_vertexShader && m_pixelShader) {
        // Set shaders
        context.deviceContext->VSSetShader(m_vertexShader.Get(), nullptr, 0);
        context.deviceContext->PSSetShader(m_pixelShader.Get(), nullptr, 0);
        
        // Set input texture
        context.deviceContext->PSSetShaderResources(0, 1, &inputSRV);
        
        // Set sampler state
        ID3D11SamplerState* sampler = m_samplerState.Get();
        context.deviceContext->PSSetSamplers(0, 1, &sampler);
        
        // Set render states (no blending for passthrough)
        float blendFactor[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        context.deviceContext->OMSetBlendState(nullptr, blendFactor, 0xFFFFFFFF);
        context.deviceContext->RSSetState(m_rasterizerState.Get());
        
        // Render fullscreen quad using shared resources
        auto& sharedResources = D3D11RenderPassResources::GetInstance();
        
        // Set vertex buffer with texture coordinate adjustment if needed
        bool needsAdjustment = context.isOriginalTexture && 
                              (context.textureWidth > context.inputWidth || 
                               context.textureHeight > context.inputHeight);
                               
        ID3D11Buffer* vertexBuffer = nullptr;
        UINT stride = 0;
        
        if (needsAdjustment) {
            // Create adjusted vertex buffer for padded textures
            vertexBuffer = CreateAdjustedVertexBuffer(context);
            stride = 5 * sizeof(float); // Position (3) + TexCoord (2)
        } else {
            // Use shared geometry for standard textures
            vertexBuffer = sharedResources.GetFullscreenQuadVertexBuffer();
            stride = sharedResources.GetFullscreenQuadVertexStride();
        }
        
        if (vertexBuffer) {
            UINT offset = 0;
            context.deviceContext->IASetVertexBuffers(0, 1, &vertexBuffer, &stride, &offset);
        }
        
        // Set index buffer (use shared geometry)
        ID3D11Buffer* indexBuffer = sharedResources.GetFullscreenQuadIndexBuffer();
        if (indexBuffer) {
            context.deviceContext->IASetIndexBuffer(indexBuffer, DXGI_FORMAT_R16_UINT, 0);
        }
        
        // Set input layout
        context.deviceContext->IASetInputLayout(m_inputLayout.Get());
        
        // Set primitive topology
        context.deviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        
        // Draw
        context.deviceContext->DrawIndexed(sharedResources.GetFullscreenQuadIndexCount(), 0, 0);
        
        // Release adjusted vertex buffer if we created one
        if (needsAdjustment && vertexBuffer) {
            vertexBuffer->Release();
        }
        
        // Unbind resources
        ID3D11ShaderResourceView* nullSRV = nullptr;
        context.deviceContext->PSSetShaderResources(0, 1, &nullSRV);
    }
    
    // Then render ImGui overlay (visibility handled in RenderImGuiContent)
    // Set blend state for alpha blending with ImGui
    float blendFactor[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    context.deviceContext->OMSetBlendState(m_blendState.Get(), blendFactor, 0xFFFFFFFF);
    
    RenderImGuiContent();
    
    return true;
}

void D3D11OverlayRenderPass::CleanupImGuiBackend() {
    ImGui_ImplDX11_Shutdown();
}

void D3D11OverlayRenderPass::Cleanup() {
    OverlayRenderPass::Cleanup();
    CleanupPassthroughShaders();
    
    // Cleanup D3D11 resources
    m_device.Reset();
    m_context.Reset();
}

void D3D11OverlayRenderPass::UpdateParameters(const std::map<std::string, RenderPassParameter>& parameters) {
    // Handle overlay-specific parameters
    for (const auto& param : parameters) {
        const std::string& name = param.first;
        const RenderPassParameter& value = param.second;
        
        if (name == "show_ui_registry") {
            if (std::holds_alternative<bool>(value)) {
                SetUIRegistryVisible(std::get<bool>(value));
                LOG_INFO("Overlay: UI Registry visibility set to ", std::get<bool>(value));
            }
        } else if (name == "show_notifications") {
            if (std::holds_alternative<bool>(value)) {
                SetNotificationsVisible(std::get<bool>(value));
                LOG_INFO("Overlay: Notifications visibility set to ", std::get<bool>(value));
            }
        }
    }
}

bool D3D11OverlayRenderPass::InitializePassthroughShaders() {
    // Use ShaderLibrary for consistent shader management
    std::string vertexShaderSource = ShaderLibrary::GetFullscreenQuadVertexShader();
    std::string pixelShaderSource = ShaderLibrary::GetPassthroughPixelShader();
    
    // Compile vertex shader
    ComPtr<ID3DBlob> vsBlob;
    if (FAILED(CompileShaderFromString(vertexShaderSource, "main", "vs_5_0", &vsBlob))) {
        Logger::GetInstance().Error("Failed to compile vertex shader for overlay passthrough");
        return false;
    }
    
    if (FAILED(m_device->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), nullptr, &m_vertexShader))) {
        Logger::GetInstance().Error("Failed to create vertex shader for overlay passthrough");
        return false;
    }
    
    // Create input layout
    D3D11_INPUT_ELEMENT_DESC layout[] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 }
    };
    
    if (FAILED(m_device->CreateInputLayout(layout, 2, vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), &m_inputLayout))) {
        Logger::GetInstance().Error("Failed to create input layout for overlay passthrough");
        return false;
    }
    
    // Compile pixel shader
    ComPtr<ID3DBlob> psBlob;
    if (FAILED(CompileShaderFromString(pixelShaderSource, "main", "ps_5_0", &psBlob))) {
        Logger::GetInstance().Error("Failed to compile pixel shader for overlay passthrough");
        return false;
    }
    
    if (FAILED(m_device->CreatePixelShader(psBlob->GetBufferPointer(), psBlob->GetBufferSize(), nullptr, &m_pixelShader))) {
        Logger::GetInstance().Error("Failed to create pixel shader for overlay passthrough");
        return false;
    }
    
    // Create sampler state
    D3D11_SAMPLER_DESC samplerDesc = {};
    samplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
    samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    samplerDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
    samplerDesc.MinLOD = 0;
    samplerDesc.MaxLOD = D3D11_FLOAT32_MAX;
    
    if (FAILED(m_device->CreateSamplerState(&samplerDesc, &m_samplerState))) {
        Logger::GetInstance().Error("Failed to create sampler state for overlay passthrough");
        return false;
    }
    
    // Create blend state for alpha blending with ImGui
    D3D11_BLEND_DESC blendDesc = {};
    blendDesc.AlphaToCoverageEnable = FALSE;
    blendDesc.IndependentBlendEnable = FALSE;
    blendDesc.RenderTarget[0].BlendEnable = TRUE;
    blendDesc.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
    blendDesc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
    blendDesc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    blendDesc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_INV_DEST_ALPHA;
    blendDesc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ONE;
    blendDesc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    blendDesc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    
    if (FAILED(m_device->CreateBlendState(&blendDesc, &m_blendState))) {
        Logger::GetInstance().Error("Failed to create blend state for overlay passthrough");
        return false;
    }
    
    // Create rasterizer state
    D3D11_RASTERIZER_DESC rasterizerDesc = {};
    rasterizerDesc.FillMode = D3D11_FILL_SOLID;
    rasterizerDesc.CullMode = D3D11_CULL_NONE;
    rasterizerDesc.FrontCounterClockwise = FALSE;
    rasterizerDesc.DepthBias = 0;
    rasterizerDesc.DepthBiasClamp = 0.0f;
    rasterizerDesc.SlopeScaledDepthBias = 0.0f;
    rasterizerDesc.DepthClipEnable = FALSE;
    rasterizerDesc.ScissorEnable = FALSE;
    rasterizerDesc.MultisampleEnable = FALSE;
    rasterizerDesc.AntialiasedLineEnable = FALSE;
    
    if (FAILED(m_device->CreateRasterizerState(&rasterizerDesc, &m_rasterizerState))) {
        Logger::GetInstance().Error("Failed to create rasterizer state for overlay passthrough");
        return false;
    }
    
    return true;
}

void D3D11OverlayRenderPass::CleanupPassthroughShaders() {
    m_vertexShader.Reset();
    m_pixelShader.Reset();
    m_vertexBuffer.Reset();
    m_inputLayout.Reset();
    m_samplerState.Reset();
    m_blendState.Reset();
    m_rasterizerState.Reset();
}

HRESULT D3D11OverlayRenderPass::CompileShaderFromString(const std::string& shaderCode, const std::string& entryPoint,
                                                       const std::string& profile, ID3DBlob** blob) {
    UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;
#ifdef _DEBUG
    flags |= D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#endif
    
    ComPtr<ID3DBlob> errorBlob;
    HRESULT hr = D3DCompile(shaderCode.c_str(), shaderCode.length(), nullptr, nullptr, nullptr,
                           entryPoint.c_str(), profile.c_str(), flags, 0, blob, &errorBlob);
    
    if (FAILED(hr) && errorBlob) {
        Logger::GetInstance().Error("Shader compilation error: ", static_cast<char*>(errorBlob->GetBufferPointer()));
    }
    
    return hr;
}

ID3D11Buffer* D3D11OverlayRenderPass::CreateAdjustedVertexBuffer(const D3D11RenderPassContext& context) {
    // Calculate texture coordinate adjustment
    float texCoordU = static_cast<float>(context.inputWidth) / static_cast<float>(context.textureWidth);
    float texCoordV = static_cast<float>(context.inputHeight) / static_cast<float>(context.textureHeight);
    
    // Create adjusted vertices for this frame
    float adjustedVertices[] = {
        // Position (x, y, z)    // TexCoord (u, v) - adjusted for content area
        -1.0f, -1.0f, 0.0f,      0.0f, texCoordV,      // Bottom-left
         1.0f, -1.0f, 0.0f,      texCoordU, texCoordV,  // Bottom-right
         1.0f,  1.0f, 0.0f,      texCoordU, 0.0f,      // Top-right
        -1.0f,  1.0f, 0.0f,      0.0f, 0.0f            // Top-left
    };
    
    // Create vertex buffer
    D3D11_BUFFER_DESC desc = {};
    desc.ByteWidth = sizeof(adjustedVertices);
    desc.Usage = D3D11_USAGE_IMMUTABLE;
    desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    desc.CPUAccessFlags = 0;
    
    D3D11_SUBRESOURCE_DATA initData = {};
    initData.pSysMem = adjustedVertices;
    
    ComPtr<ID3D11Buffer> buffer;
    HRESULT hr = m_device->CreateBuffer(&desc, &initData, &buffer);
    if (FAILED(hr)) {
        Logger::GetInstance().Error("Failed to create adjusted vertex buffer for overlay pass");
        return nullptr;
    }
    
    // Return raw pointer - caller manages lifetime
    buffer->AddRef();
    return buffer.Get();
}
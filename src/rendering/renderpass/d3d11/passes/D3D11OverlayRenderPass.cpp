#include "D3D11OverlayRenderPass.h"
#include "imgui.h"
#include "imgui_impl_dx11.h"
#include "../../../../ui/ImGuiManager.h"
#include "../../../../ui/UIRegistry.h"
#include "../../../../ui/NotificationManager.h"
#include "../../../../core/Logger.h"

D3D11OverlayRenderPass::D3D11OverlayRenderPass() 
    : OverlayRenderPass(), D3D11RenderPass("Overlay") {
}

D3D11OverlayRenderPass::~D3D11OverlayRenderPass() = default;

bool D3D11OverlayRenderPass::Initialize(ID3D11Device* device, const RenderPassConfig& config) {
    m_device = device;
    m_device->GetImmediateContext(m_context.GetAddressOf());
    
    // Get window size from config or context (we'll need to handle this properly)
    int width = 1920; // TODO: Get from config or context
    int height = 1080; // TODO: Get from config or context
    
    // Call base class common initialization
    if (!InitializeCommon(width, height)) {
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
    
    // First, copy input to output (passthrough)
    // TODO: Implement passthrough rendering using shaders
    // For now, assume input is already in output or implement simple copy
    
    // Then render ImGui overlay if visible
    if (m_visible) {
        RenderImGuiContent();
    }
    
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
    // Handle overlay-specific parameters if any
    // For now, overlay doesn't have configurable parameters
}

bool D3D11OverlayRenderPass::InitializePassthroughShaders() {
    // TODO: Implement passthrough shader creation
    // Similar to existing D3D11 render passes
    // This would create shaders to copy the input texture to output before overlaying ImGui
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
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
    : OverlayRenderPass(), D3D11SimpleRenderPass("Overlay") {
}

D3D11OverlayRenderPass::~D3D11OverlayRenderPass() = default;

bool D3D11OverlayRenderPass::Initialize(ID3D11Device* device, const RenderPassConfig& config) {
    return Initialize(device, config, nullptr);
}

bool D3D11OverlayRenderPass::Initialize(ID3D11Device* device, const RenderPassConfig& config, void* hwnd) {
    // First initialize the base class (will use virtual methods for shaders)
    if (!D3D11SimpleRenderPass::Initialize(device, config)) {
        Logger::GetInstance().Error("Failed to initialize D3D11SimpleRenderPass for overlay");
        return false;
    }
    
    // Get window size from config - overlay passes typically use the render target size
    // For now, use default dimensions as overlay will be sized to match the render target
    int width = 1920;  // Will be updated when rendering based on context
    int height = 1080; // Will be updated when rendering based on context
    
    // Initialize overlay-specific components
    if (!InitializeCommon(width, height, hwnd)) {
        Logger::GetInstance().Error("Failed to initialize OverlayRenderPass common for overlay");
        return false;
    }
    
    // Create overlay-specific blend state for ImGui
    if (!CreateOverlayBlendState(device)) {
        Logger::GetInstance().Error("Failed to create overlay blend state");
        return false;
    }
    
    return true;
}

bool D3D11OverlayRenderPass::InitializeImGuiBackend() {
    // Use the device from the base class
    ID3D11Device* device = m_device.Get();
    if (!device) {
        return false;
    }
    
    ComPtr<ID3D11DeviceContext> context;
    device->GetImmediateContext(context.GetAddressOf());
    
    return ImGui_ImplDX11_Init(device, context.Get());
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
    
    // First: render input to output using base class passthrough functionality
    if (!D3D11SimpleRenderPass::Execute(context, inputSRV, outputRTV)) {
        Logger::GetInstance().Error("D3D11OverlayRenderPass: Failed to execute passthrough rendering");
        return false;
    }
    
    // Then: render ImGui overlay with alpha blending
    // The render target is already set by the base class, just need to set blend state
    float blendFactor[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    context.deviceContext->OMSetBlendState(m_overlayBlendState.Get(), blendFactor, 0xFFFFFFFF);
    
    RenderImGuiContent();
    
    return true;
}

void D3D11OverlayRenderPass::CleanupImGuiBackend() {
    ImGui_ImplDX11_Shutdown();
}

void D3D11OverlayRenderPass::Cleanup() {
    OverlayRenderPass::Cleanup();
    
    // Cleanup overlay-specific resources
    m_overlayBlendState.Reset();
    
    // Call base class cleanup (handles shaders and other resources)
    D3D11SimpleRenderPass::Cleanup();
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

std::string D3D11OverlayRenderPass::GetPixelShaderSource() const {
    return ShaderLibrary::GetPassthroughPixelShader();
}

bool D3D11OverlayRenderPass::CreateOverlayBlendState(ID3D11Device* device) {
    // Create blend state for alpha blending with ImGui overlay
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
    
    if (FAILED(device->CreateBlendState(&blendDesc, &m_overlayBlendState))) {
        Logger::GetInstance().Error("Failed to create overlay blend state");
        return false;
    }
    
    return true;
}
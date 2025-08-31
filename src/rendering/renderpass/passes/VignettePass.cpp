#include "VignettePass.h"
#include "../ShaderLibrary.h"
#include "core/Logger.h"
#include <cstring>

VignettePass::VignettePass() : D3D11SimpleRenderPass("Vignette") {
    m_intensity = 0.5f;
    m_radius = 0.6f;
    m_feather = 0.4f;
    m_centerX = 0.0f;
    m_centerY = 0.0f;
}

void VignettePass::UpdateParameters(const std::map<std::string, RenderPassParameter>& parameters) {
    for (const auto& [name, value] : parameters) {
        if (name == "intensity" && std::holds_alternative<float>(value)) {
            m_intensity = std::get<float>(value);
        } else if (name == "radius" && std::holds_alternative<float>(value)) {
            m_radius = std::get<float>(value);
        } else if (name == "feather" && std::holds_alternative<float>(value)) {
            m_feather = std::get<float>(value);
        } else if (name == "center_x" && std::holds_alternative<float>(value)) {
            m_centerX = std::get<float>(value);
        } else if (name == "center_y" && std::holds_alternative<float>(value)) {
            m_centerY = std::get<float>(value);
        }
    }
}

std::string VignettePass::GetPixelShaderSource() const {
    return ShaderLibrary::GetVignettePixelShader();
}

size_t VignettePass::GetConstantBufferSize() const {
    return sizeof(ConstantBufferData);
}

void VignettePass::PackConstantBuffer(uint8_t* buffer, const D3D11RenderPassContext& context) {
    ConstantBufferData* data = reinterpret_cast<ConstantBufferData*>(buffer);
    data->intensity = m_intensity;
    data->radius = m_radius;
    data->feather = m_feather;
    data->centerX = m_centerX;
    data->centerY = m_centerY;
    data->aspectRatio = static_cast<float>(context.inputWidth) / static_cast<float>(context.inputHeight);
}
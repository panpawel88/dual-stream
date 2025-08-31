#include "BloomPass.h"
#include "../ShaderLibrary.h"
#include "core/Logger.h"
#include <cstring>

BloomPass::BloomPass() : D3D11SimpleRenderPass("Bloom") {
    m_threshold = 0.8f;
    m_intensity = 1.0f;
    m_radius = 1.5f;
    m_blendFactor = 0.3f;
}

void BloomPass::UpdateParameters(const std::map<std::string, RenderPassParameter>& parameters) {
    for (const auto& [name, value] : parameters) {
        if (name == "threshold" && std::holds_alternative<float>(value)) {
            m_threshold = std::get<float>(value);
        } else if (name == "intensity" && std::holds_alternative<float>(value)) {
            m_intensity = std::get<float>(value);
        } else if (name == "radius" && std::holds_alternative<float>(value)) {
            m_radius = std::get<float>(value);
        } else if (name == "blend_factor" && std::holds_alternative<float>(value)) {
            m_blendFactor = std::get<float>(value);
        }
    }
}

std::string BloomPass::GetPixelShaderSource() const {
    return ShaderLibrary::GetSimpleBloomPixelShader();
}

size_t BloomPass::GetConstantBufferSize() const {
    return sizeof(ConstantBufferData);
}

void BloomPass::PackConstantBuffer(uint8_t* buffer, const RenderPassContext& context) {
    ConstantBufferData* data = reinterpret_cast<ConstantBufferData*>(buffer);
    data->threshold = m_threshold;
    data->intensity = m_intensity;
    data->radius = m_radius;
    data->blendFactor = m_blendFactor;
    data->texelSizeX = 1.0f / context.inputWidth;
    data->texelSizeY = 1.0f / context.inputHeight;
}
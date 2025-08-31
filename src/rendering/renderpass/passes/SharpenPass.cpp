#include "SharpenPass.h"
#include "../ShaderLibrary.h"
#include "core/Logger.h"
#include <cstring>

SharpenPass::SharpenPass() : D3D11SimpleRenderPass("Sharpen") {
    m_sharpness = 0.5f;
    m_radius = 1.0f;
    m_threshold = 0.01f;
}

void SharpenPass::UpdateParameters(const std::map<std::string, RenderPassParameter>& parameters) {
    for (const auto& [name, value] : parameters) {
        if (name == "sharpness" && std::holds_alternative<float>(value)) {
            m_sharpness = std::get<float>(value);
        } else if (name == "radius" && std::holds_alternative<float>(value)) {
            m_radius = std::get<float>(value);
        } else if (name == "threshold" && std::holds_alternative<float>(value)) {
            m_threshold = std::get<float>(value);
        }
    }
}

std::string SharpenPass::GetPixelShaderSource() const {
    return ShaderLibrary::GetSharpenPixelShader();
}

size_t SharpenPass::GetConstantBufferSize() const {
    return sizeof(ConstantBufferData);
}

void SharpenPass::PackConstantBuffer(uint8_t* buffer, const RenderPassContext& context) {
    ConstantBufferData* data = reinterpret_cast<ConstantBufferData*>(buffer);
    data->sharpness = m_sharpness;
    data->radius = m_radius;
    data->threshold = m_threshold;
    data->texelSizeX = 1.0f / context.inputWidth;
    data->texelSizeY = 1.0f / context.inputHeight;
}
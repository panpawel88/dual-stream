#pragma once

#include "../D3D11SimpleRenderPass.h"

/**
 * Bloom render pass that creates a glow effect around bright areas.
 * Enhances bright pixels and creates light bleeding for cinematic effects.
 * 
 * Supported parameters:
 * - threshold: Brightness threshold for bloom effect (0.0 - 1.0, default 0.8)
 * - intensity: Strength of the bloom effect (0.0 - 2.0, default 1.0)
 * - radius: Size of the bloom glow (0.5 - 3.0, default 1.5)
 * - blend_factor: Blend ratio with original image (0.0 - 1.0, default 0.3)
 */
class BloomPass : public D3D11SimpleRenderPass {
public:
    BloomPass();
    virtual ~BloomPass() = default;

    void UpdateParameters(const std::map<std::string, RenderPassParameter>& parameters) override;

protected:
    // Override virtual methods from D3D11SimpleRenderPass
    std::string GetPixelShaderSource() const override;
    size_t GetConstantBufferSize() const override;
    void PackConstantBuffer(uint8_t* buffer, const D3D11RenderPassContext& context) override;

private:
    struct ConstantBufferData {
        float threshold;
        float intensity;
        float radius;
        float blendFactor;
        float texelSizeX;
        float texelSizeY;
        float padding[2];
    };
    
    float m_threshold;
    float m_intensity;
    float m_radius;
    float m_blendFactor;
};
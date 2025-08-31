#pragma once

#include "../D3D11SimpleRenderPass.h"

/**
 * Sharpen render pass that enhances edges and details in the image.
 * Uses an unsharp masking technique for edge enhancement.
 * 
 * Supported parameters:
 * - sharpness: Strength of the sharpening effect (0.0 - 2.0, default 0.5)
 * - radius: Size of the sharpening kernel (0.5 - 2.0, default 1.0)
 * - threshold: Minimum edge difference to sharpen (0.0 - 0.1, default 0.01)
 */
class SharpenPass : public D3D11SimpleRenderPass {
public:
    SharpenPass();
    virtual ~SharpenPass() = default;

    void UpdateParameters(const std::map<std::string, RenderPassParameter>& parameters) override;

protected:
    // Override virtual methods from D3D11SimpleRenderPass
    std::string GetPixelShaderSource() const override;
    size_t GetConstantBufferSize() const override;
    void PackConstantBuffer(uint8_t* buffer, const D3D11RenderPassContext& context) override;

private:
    struct ConstantBufferData {
        float sharpness;
        float radius;
        float threshold;
        float texelSizeX;
        float texelSizeY;
        float padding[3];
    };
    
    float m_sharpness;
    float m_radius;
    float m_threshold;
};
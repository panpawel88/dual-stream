#pragma once

#include <d3d11.h>
#include <wrl/client.h>
#include <string>
#include <map>
#include <variant>

using Microsoft::WRL::ComPtr;

// Forward declarations
class RenderPassConfig;

/**
 * Parameter value types supported by render passes
 */
using RenderPassParameter = std::variant<float, int, bool, std::array<float, 2>, std::array<float, 3>, std::array<float, 4>>;

/**
 * Context passed to render passes during execution
 */
struct RenderPassContext {
    ID3D11DeviceContext* deviceContext;
    float deltaTime;        // Time since last frame
    float totalTime;        // Total elapsed time
    int frameNumber;        // Frame counter
    int inputWidth;         // Input texture width
    int inputHeight;        // Input texture height
};

/**
 * Abstract base class for all render passes.
 * Supports simple shader-based passes with extensibility for external libraries.
 */
class RenderPass {
public:
    enum class PassType {
        Simple,         // Vertex + pixel shader pass (current implementation)
        External        // External library pass (future extension)
    };

    RenderPass(const std::string& name) : m_name(name), m_enabled(true) {}
    virtual ~RenderPass() = default;

    // Core interface
    virtual PassType GetType() const = 0;
    virtual bool Initialize(ID3D11Device* device, const RenderPassConfig& config) = 0;
    virtual void Cleanup() = 0;
    
    /**
     * Execute the render pass
     * @param context Rendering context with device and timing info
     * @param inputSRV Input texture to process
     * @param outputRTV Output render target
     * @return true on success
     */
    virtual bool Execute(const RenderPassContext& context,
                        ID3D11ShaderResourceView* inputSRV,
                        ID3D11RenderTargetView* outputRTV) = 0;

    /**
     * Update pass parameters at runtime
     * @param parameters Map of parameter name to value
     */
    virtual void UpdateParameters(const std::map<std::string, RenderPassParameter>& parameters) = 0;

    // Properties
    const std::string& GetName() const { return m_name; }
    bool IsEnabled() const { return m_enabled; }
    void SetEnabled(bool enabled) { m_enabled = enabled; }

protected:
    std::string m_name;
    bool m_enabled;
};
#pragma once

#include <string>
#include <map>
#include <variant>
#include <array>

/**
 * Parameter value types supported by render passes
 */
using RenderPassParameter = std::variant<float, int, bool, std::array<float, 2>, std::array<float, 3>, std::array<float, 4>>;

/**
 * Abstract base interface for all render passes, independent of graphics API
 */
class IRenderPass {
public:
    enum class PassType {
        Simple,         // Vertex + pixel/fragment shader pass
        External        // External library pass (future extension)
    };

    IRenderPass(const std::string& name) : m_name(name), m_enabled(true) {}
    virtual ~IRenderPass() = default;

    // Core interface
    virtual PassType GetType() const = 0;
    virtual void Cleanup() = 0;

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
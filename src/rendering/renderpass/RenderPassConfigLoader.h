#pragma once

#include "RenderPass.h"
#include "D3D11SimpleRenderPass.h"
#include "RenderPassPipeline.h"
#include "RenderPassConfig.h"
#include <memory>
#include <vector>
#include <string>
#include <d3d11.h>

// Forward declaration
class Config;

/**
 * Loads render pass configuration from INI files and creates render pass pipeline
 */
class RenderPassConfigLoader {
public:
    /**
     * Load render pass configuration and create pipeline
     * @param device D3D11 device for pass initialization
     * @param config Configuration instance to read from
     * @return Configured render pass pipeline or nullptr on failure
     */
    static std::unique_ptr<RenderPassPipeline> LoadPipeline(ID3D11Device* device, Config* config);

private:
    /**
     * Load configuration for a specific render pass
     * @param passName Name of the render pass
     * @param config Configuration instance
     * @return Render pass configuration
     */
    static RenderPassConfig LoadPassConfig(const std::string& passName, Config* config);
    
    /**
     * Create a render pass from configuration
     * @param passName Name of the render pass
     * @param passConfig Pass configuration
     * @param device D3D11 device for initialization
     * @return Created render pass or nullptr on failure
     */
    static std::unique_ptr<RenderPass> CreatePass(const std::string& passName, 
                                                 const RenderPassConfig& passConfig,
                                                 ID3D11Device* device);
    
    /**
     * Parse comma-separated list of pass names
     * @param passChain Comma-separated string of pass names
     * @return Vector of pass names
     */
    static std::vector<std::string> ParsePassChain(const std::string& passChain);
};
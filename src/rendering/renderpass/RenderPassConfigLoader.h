#pragma once

#include "IRenderPassPipeline.h"
#include "RenderPass.h"
#include "d3d11/D3D11SimpleRenderPass.h"
#include "RenderPassPipeline.h"
#include "RenderPassConfig.h"
#include "d3d11/passes/PassthroughPass.h"
#include "d3d11/passes/YUVToRGBRenderPass.h"
#include "d3d11/passes/MotionBlurPass.h"
#include "d3d11/passes/VignettePass.h"
#include "d3d11/passes/SharpenPass.h"
#include "d3d11/passes/BloomPass.h"
#include <memory>
#include <vector>
#include <string>
#include <d3d11.h>

// Forward declarations
class Config;
class OpenGLRenderPassPipeline;
class OpenGLRenderPass;

enum class RenderAPI {
    D3D11,
    OpenGL
};

/**
 * Loads render pass configuration from INI files and creates render pass pipeline
 */
class RenderPassConfigLoader {
public:
    /**
     * Load D3D11 render pass configuration and create pipeline
     * @param device D3D11 device for pass initialization
     * @param config Configuration instance to read from
     * @return Configured D3D11 render pass pipeline or nullptr on failure
     */
    static std::unique_ptr<RenderPassPipeline> LoadD3D11Pipeline(ID3D11Device* device, Config* config);
    
    /**
     * Load OpenGL render pass configuration and create pipeline
     * @param config Configuration instance to read from
     * @return Configured OpenGL render pass pipeline or nullptr on failure
     */
    static std::unique_ptr<OpenGLRenderPassPipeline> LoadOpenGLPipeline(Config* config);
    
    /**
     * Load render pass configuration and create pipeline (legacy method)
     * @param device D3D11 device for pass initialization
     * @param config Configuration instance to read from
     * @return Configured render pass pipeline or nullptr on failure
     */
    static std::unique_ptr<RenderPassPipeline> LoadPipeline(ID3D11Device* device, Config* config) {
        return LoadD3D11Pipeline(device, config);
    }

private:
    /**
     * Load configuration for a specific render pass
     * @param passName Name of the render pass
     * @param config Configuration instance
     * @return Render pass configuration
     */
    static RenderPassConfig LoadPassConfig(const std::string& passName, Config* config);
    
    /**
     * Create a D3D11 render pass from configuration
     * @param passName Name of the render pass
     * @param passConfig Pass configuration
     * @param device D3D11 device for initialization
     * @return Created render pass or nullptr on failure
     */
    static std::unique_ptr<RenderPass> CreatePass(const std::string& passName, 
                                                 const RenderPassConfig& passConfig,
                                                 ID3D11Device* device);
    
    /**
     * Create an OpenGL render pass from configuration
     * @param passName Name of the render pass
     * @param passConfig Pass configuration
     * @return Created render pass or nullptr on failure
     */
    static std::unique_ptr<OpenGLRenderPass> CreateOpenGLPass(const std::string& passName, 
                                                             const RenderPassConfig& passConfig);
    
    /**
     * Parse comma-separated list of pass names
     * @param passChain Comma-separated string of pass names
     * @return Vector of pass names
     */
    static std::vector<std::string> ParsePassChain(const std::string& passChain);
};
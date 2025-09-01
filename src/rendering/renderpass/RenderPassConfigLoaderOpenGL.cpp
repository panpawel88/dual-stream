// Separate compilation unit for OpenGL render pass loading to avoid header conflicts
// This file handles all OpenGL-specific render pass creation

#include <glad/gl.h>  // Include GLAD first in this isolated file

#include "RenderPassConfigLoader.h"
#include "core/Config.h"
#include "core/Logger.h"

// OpenGL includes - safe to include after GLAD in isolated file
#include "opengl/OpenGLRenderPassPipeline.h"
#include "opengl/OpenGLSimpleRenderPass.h"
#include "opengl/passes/PassthroughPass.h"
#include "opengl/passes/YUVToRGBRenderPass.h"
#include "opengl/passes/MotionBlurPass.h"
#include "opengl/passes/VignettePass.h"
#include "opengl/passes/SharpenPass.h"
#include "opengl/passes/BloomPass.h"

std::unique_ptr<OpenGLRenderPassPipeline> RenderPassConfigLoader::LoadOpenGLPipeline(Config* config) {
    if (!config) {
        LOG_ERROR("RenderPassConfigLoader: Invalid config");
        return nullptr;
    }
    
    // Check if render passes are enabled
    bool renderPassesEnabled = config->GetBool("rendering.enable_render_passes", false);
    if (!renderPassesEnabled) {
        LOG_INFO("OpenGL render passes disabled in configuration");
        return nullptr; // Not an error, just disabled
    }
    
    // Create pipeline
    auto pipeline = std::make_unique<OpenGLRenderPassPipeline>();
    if (!pipeline->Initialize()) {
        LOG_ERROR("RenderPassConfigLoader: Failed to initialize OpenGL render pass pipeline");
        return nullptr;
    }
    
    // Get render pass chain
    std::string passChain = config->GetString("rendering.render_pass_chain", "");
    if (passChain.empty()) {
        LOG_INFO("No OpenGL render pass chain specified, pipeline will passthrough");
        return pipeline; // Empty pipeline is valid
    }
    
    // Parse pass chain
    std::vector<std::string> passNames = RenderPassConfigLoader::ParsePassChain(passChain);
    if (passNames.empty()) {
        LOG_WARNING("Empty OpenGL render pass chain after parsing");
        return pipeline;
    }
    
    LOG_INFO("Loading OpenGL render pass chain: ", passChain);
    
    // Create and add passes
    int passCount = 0;
    for (const std::string& passName : passNames) {
        // Load pass configuration
        RenderPassConfig passConfig = RenderPassConfigLoader::LoadPassConfig(passName, config);
        
        // Create pass
        auto pass = RenderPassConfigLoader::CreateOpenGLPass(passName, passConfig);
        if (pass) {
            pipeline->AddOpenGLPass(std::move(pass));
            passCount++;
            LOG_INFO("Added OpenGL render pass: ", passName);
        } else {
            LOG_ERROR("Failed to create OpenGL render pass: ", passName);
            // Continue with other passes rather than failing entirely
        }
    }
    
    if (passCount == 0) {
        LOG_WARNING("No OpenGL render passes were successfully created");
        return nullptr;
    }
    
    // Enable pipeline
    pipeline->SetEnabled(true);
    
    LOG_INFO("OpenGL render pass pipeline created with ", passCount, " passes");
    return pipeline;
}

std::unique_ptr<OpenGLRenderPass> RenderPassConfigLoader::CreateOpenGLPass(const std::string& passName,
                                                                          const RenderPassConfig& passConfig) {
    // Check if pass is enabled
    if (!passConfig.GetBool("enabled", true)) {
        LOG_INFO("OpenGL render pass ", passName, " is disabled");
        return nullptr;
    }
    
    // Create OpenGL render pass based on pass name
    std::unique_ptr<OpenGLRenderPass> pass;
    
    if (passName == "passthrough" || passName == "Passthrough") {
        pass = std::make_unique<OpenGLPassthroughPass>();
    } else if (passName == "yuv_to_rgb" || passName == "YUVToRGB") {
        pass = std::make_unique<OpenGLYUVToRGBRenderPass>();
    } else if (passName == "motion_blur" || passName == "MotionBlur") {
        pass = std::make_unique<OpenGLMotionBlurPass>();
    } else if (passName == "vignette" || passName == "Vignette") {
        pass = std::make_unique<OpenGLVignettePass>();
    } else if (passName == "sharpen" || passName == "Sharpen") {
        pass = std::make_unique<OpenGLSharpenPass>();
    } else if (passName == "bloom" || passName == "Bloom") {
        pass = std::make_unique<OpenGLBloomPass>();
    } else {
        LOG_WARNING("Unknown OpenGL render pass type: ", passName, ", falling back to simple render pass");
        pass = std::make_unique<OpenGLSimpleRenderPass>(passName);
    }
    
    // Initialize the pass
    if (!pass->Initialize(passConfig)) {
        LOG_ERROR("Failed to initialize OpenGL render pass: ", passName);
        return nullptr;
    }
    
    return pass;
}
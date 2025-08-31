// Define WIN32_LEAN_AND_MEAN and NOMINMAX to exclude OpenGL from windows.h
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "RenderPassConfigLoader.h"
#include "core/Config.h"
#include "core/Logger.h"
#include <sstream>
#include <algorithm>

// OpenGL render pass functionality moved to RenderPassConfigLoaderOpenGL.cpp
// to avoid GLAD header conflicts with Windows headers

std::unique_ptr<RenderPassPipeline> RenderPassConfigLoader::LoadD3D11Pipeline(ID3D11Device* device, Config* config) {
    if (!device || !config) {
        LOG_ERROR("RenderPassConfigLoader: Invalid device or config");
        return nullptr;
    }
    
    // Check if render passes are enabled
    bool renderPassesEnabled = config->GetBool("rendering.enable_render_passes", false);
    if (!renderPassesEnabled) {
        LOG_INFO("Render passes disabled in configuration");
        return nullptr; // Not an error, just disabled
    }
    
    // Create pipeline
    auto pipeline = std::make_unique<RenderPassPipeline>();
    if (!pipeline->Initialize(device)) {
        LOG_ERROR("RenderPassConfigLoader: Failed to initialize render pass pipeline");
        return nullptr;
    }
    
    // Get render pass chain
    std::string passChain = config->GetString("rendering.render_pass_chain", "");
    if (passChain.empty()) {
        LOG_INFO("No render pass chain specified, pipeline will passthrough");
        return pipeline; // Empty pipeline is valid
    }
    
    // Parse pass chain
    std::vector<std::string> passNames = ParsePassChain(passChain);
    if (passNames.empty()) {
        LOG_WARNING("Empty render pass chain after parsing");
        return pipeline;
    }
    
    LOG_INFO("Loading render pass chain: ", passChain);
    
    // Create and add passes
    int passCount = 0;
    for (const std::string& passName : passNames) {
        // Load pass configuration
        RenderPassConfig passConfig = LoadPassConfig(passName, config);
        
        // Create pass
        auto pass = CreatePass(passName, passConfig, device);
        if (pass) {
            pipeline->AddPass(std::move(pass));
            passCount++;
            LOG_INFO("Added render pass: ", passName);
        } else {
            LOG_ERROR("Failed to create render pass: ", passName);
            // Continue with other passes rather than failing entirely
        }
    }
    
    if (passCount == 0) {
        LOG_WARNING("No render passes were successfully created");
        return nullptr;
    }
    
    // Enable pipeline
    pipeline->SetEnabled(true);
    
    LOG_INFO("Render pass pipeline created with ", passCount, " passes");
    return pipeline;
}

// OpenGL methods are implemented in RenderPassConfigLoaderOpenGL.cpp

RenderPassConfig RenderPassConfigLoader::LoadPassConfig(const std::string& passName, Config* config) {
    RenderPassConfig passConfig;
    
    // Get all keys for this render pass section
    std::string sectionName = "render_pass." + passName;
    std::vector<std::string> keys = config->GetKeysInSection(sectionName);
    
    if (keys.empty()) {
        LOG_WARNING("No configuration found for render pass: ", passName);
        // Set default enabled state
        passConfig.SetBool("enabled", true);
        return passConfig;
    }
    
    // Load all parameters for this pass
    for (const std::string& fullKey : keys) {
        // Extract parameter name (remove section prefix)
        std::string paramName = fullKey;
        size_t dotPos = paramName.find_last_of('.');
        if (dotPos != std::string::npos) {
            paramName = paramName.substr(dotPos + 1);
        }
        
        // Get the value and store it
        std::string value = config->GetString(fullKey);
        passConfig.SetString(paramName, value);
        
        LOG_DEBUG("Loaded parameter: ", paramName, " = ", value, " for pass ", passName);
    }
    
    // Ensure enabled parameter exists
    if (!passConfig.HasParameter("enabled")) {
        passConfig.SetBool("enabled", true);
    }
    
    return passConfig;
}

std::unique_ptr<RenderPass> RenderPassConfigLoader::CreatePass(const std::string& passName,
                                                              const RenderPassConfig& passConfig,
                                                              ID3D11Device* device) {
    // Check if pass is enabled
    if (!passConfig.GetBool("enabled", true)) {
        LOG_INFO("Render pass ", passName, " is disabled");
        return nullptr;
    }
    
    // Create render pass based on pass name
    std::unique_ptr<RenderPass> pass;
    
    if (passName == "passthrough" || passName == "Passthrough") {
        pass = std::make_unique<PassthroughPass>();
    } else if (passName == "yuv_to_rgb" || passName == "YUVToRGB") {
        pass = std::make_unique<YUVToRGBRenderPass>();
    } else if (passName == "motion_blur" || passName == "MotionBlur") {
        pass = std::make_unique<MotionBlurPass>();
    } else if (passName == "vignette" || passName == "Vignette") {
        pass = std::make_unique<VignettePass>();
    } else if (passName == "sharpen" || passName == "Sharpen") {
        pass = std::make_unique<SharpenPass>();
    } else if (passName == "bloom" || passName == "Bloom") {
        pass = std::make_unique<BloomPass>();
    } else {
        LOG_WARNING("Unknown render pass type: ", passName, ", falling back to simple render pass");
        pass = std::make_unique<D3D11SimpleRenderPass>(passName);
    }
    
    // Initialize the pass
    if (!pass->Initialize(device, passConfig)) {
        LOG_ERROR("Failed to initialize render pass: ", passName);
        return nullptr;
    }
    
    return pass;
}

std::vector<std::string> RenderPassConfigLoader::ParsePassChain(const std::string& passChain) {
    std::vector<std::string> passNames;
    std::stringstream ss(passChain);
    std::string passName;
    
    while (std::getline(ss, passName, ',')) {
        // Trim whitespace
        passName.erase(0, passName.find_first_not_of(" \t"));
        passName.erase(passName.find_last_not_of(" \t") + 1);
        
        if (!passName.empty()) {
            passNames.push_back(passName);
        }
    }
    
    return passNames;
}
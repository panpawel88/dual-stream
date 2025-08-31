#pragma once

#include "IRenderPass.h"
#include <memory>
#include <string>
#include <vector>

/**
 * Abstract base interface for render pass pipelines, independent of graphics API
 */
class IRenderPassPipeline {
public:
    virtual ~IRenderPassPipeline() = default;

    /**
     * Cleanup all resources
     */
    virtual void Cleanup() = 0;
    
    /**
     * Add a render pass to the pipeline
     * @param pass Render pass to add (pipeline takes ownership)
     */
    virtual void AddPass(std::unique_ptr<IRenderPass> pass) = 0;
    
    /**
     * Enable/disable the entire pipeline
     * When disabled, input is passed directly to output
     */
    virtual void SetEnabled(bool enabled) = 0;
    virtual bool IsEnabled() const = 0;
    
    /**
     * Enable/disable a specific pass by name
     */
    virtual bool SetPassEnabled(const std::string& passName, bool enabled) = 0;
    
    /**
     * Get pass by name
     */
    virtual IRenderPass* GetPass(const std::string& passName) const = 0;
    
    /**
     * Get number of passes
     */
    virtual size_t GetPassCount() const = 0;
    
    /**
     * Update parameters for a specific pass
     */
    virtual bool UpdatePassParameters(const std::string& passName, 
                                    const std::map<std::string, RenderPassParameter>& parameters) = 0;
};
#include "RendererFactory.h"
#include "D3D11Renderer.h"
#include "OpenGLRenderer.h"
#include "../core/Logger.h"
#include <algorithm>
#include <vector>
#include <memory>

std::unique_ptr<IRenderer> RendererFactory::CreateRenderer(RendererBackend preference) {
    // Determine preferred backend order
    std::vector<RendererBackend> backendOrder;
    
    switch (preference) {
        case RendererBackend::DirectX11:
            backendOrder = {RendererBackend::DirectX11, RendererBackend::OpenGL};
            break;
        case RendererBackend::OpenGL:
            backendOrder = {RendererBackend::OpenGL, RendererBackend::DirectX11};
            break;
        case RendererBackend::Auto:
        default:
            backendOrder = {RendererBackend::DirectX11, RendererBackend::OpenGL};
            break;
    }
    
    // Try each backend in order
    for (RendererBackend backend : backendOrder) {
        try {
            std::unique_ptr<IRenderer> renderer;
            
            switch (backend) {
                case RendererBackend::DirectX11:
                    LOG_INFO("Attempting to create DirectX 11 renderer...");
                    renderer = std::make_unique<D3D11Renderer>();
                    break;
                case RendererBackend::OpenGL:
                    LOG_INFO("Attempting to create OpenGL renderer...");
                    renderer = std::make_unique<OpenGLRenderer>();
                    break;
                default:
                    continue; // Skip unknown backends
            }
            
            if (renderer) {
                LOG_INFO("Successfully created ", GetRendererName(backend), " renderer");
                return renderer;
            }
        } catch (const std::exception& e) {
            LOG_WARNING("Failed to create ", GetRendererName(backend), " renderer: ", e.what());
        }
    }
    
    LOG_ERROR("Failed to create any renderer backend");
    return nullptr;
}

const char* RendererFactory::GetRendererName(RendererBackend backend) {
    switch (backend) {
        case RendererBackend::DirectX11:
            return "DirectX 11";
        case RendererBackend::OpenGL:
            return "OpenGL";
        case RendererBackend::Auto:
            return "Auto";
        default:
            return "Unknown";
    }
}

const char* RendererFactory::GetDefaultRendererName() {
    // Always return DirectX 11 as the default since both renderers are available
    return "DirectX 11";
}

RendererBackend RendererFactory::ParseBackendString(const std::string& backendStr) {
    std::string lower = backendStr;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    
    if (lower == "auto") {
        return RendererBackend::Auto;
    } else if (lower == "directx11" || lower == "dx11" || lower == "d3d11") {
        return RendererBackend::DirectX11;
    } else if (lower == "opengl" || lower == "gl") {
        return RendererBackend::OpenGL;
    } else {
        LOG_WARNING("Unknown renderer backend: '", backendStr, "', using Auto");
        return RendererBackend::Auto;
    }
}
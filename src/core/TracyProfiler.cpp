#include "TracyProfiler.h"
#include "Logger.h"

#ifdef TRACY_ENABLE

void TracyProfilerManager::InitializeGPUContext(const char* rendererType) {
    if (m_gpuContextInitialized) {
        return;
    }

    LOG_INFO("Initializing Tracy GPU profiling for ", rendererType, " renderer");

    if (strcmp(rendererType, "DirectX11") == 0) {
        // D3D11 GPU context will be initialized when the renderer provides device/context
        LOG_INFO("DirectX 11 GPU profiling ready - context will be created by renderer");
    } else if (strcmp(rendererType, "OpenGL") == 0) {
        // OpenGL GPU context will be initialized when renderer provides OpenGL context
        LOG_INFO("OpenGL GPU profiling ready - context will be created by renderer");
    } else {
        LOG_WARNING("Unknown renderer type for Tracy GPU profiling: ", rendererType);
        return;
    }

    m_gpuContextInitialized = true;
}

void TracyProfilerManager::ShutdownGPUContext() {
    if (!m_gpuContextInitialized) {
        return;
    }

    LOG_INFO("Shutting down Tracy GPU profiling");
    m_gpuContextInitialized = false;
}

#endif // TRACY_ENABLE
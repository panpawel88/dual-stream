#include "RendererFactory.h"

#if USE_OPENGL_RENDERER
#include "OpenGLRenderer.h"
#else
#include "D3D11Renderer.h"
#endif

std::unique_ptr<IRenderer> RendererFactory::CreateRenderer() {
#if USE_OPENGL_RENDERER
    return std::make_unique<OpenGLRenderer>();
#else
    return std::make_unique<D3D11Renderer>();
#endif
}

const char* RendererFactory::GetRendererName() {
#if USE_OPENGL_RENDERER
    return "OpenGL";
#else
    return "DirectX 11";
#endif
}
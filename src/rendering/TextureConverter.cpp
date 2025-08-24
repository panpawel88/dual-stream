#include "TextureConverter.h"
#include "video/decode/VideoDecoder.h" // For DecodedFrame
#include "IRenderer.h"

RenderTexture TextureConverter::ConvertFrame(const DecodedFrame& frame, IRenderer* renderer) {
    if (!frame.valid || !renderer) {
        return CreateNullTexture();
    }
    
    // Determine the best texture type based on renderer type and frame data
    switch (renderer->GetRendererType()) {
        case RendererType::DirectX11: {
            // D3D11 renderer - prefer D3D11 texture if available
            if (frame.texture) {
                RenderTexture renderTexture;
                renderTexture.type = TextureType::D3D11;
                renderTexture.format = frame.isYUV ? TextureFormat::NV12 : TextureFormat::BGRA8;
                renderTexture.width = frame.width;
                renderTexture.height = frame.height;
                renderTexture.isYUV = frame.isYUV;
                renderTexture.d3d11.texture = frame.texture;
                renderTexture.d3d11.dxgiFormat = frame.format;
                return renderTexture;
            }
            // Fallback to software texture for D3D11 renderer
            if (frame.data) {
                RenderTexture renderTexture;
                renderTexture.type = TextureType::Software;
                renderTexture.format = TextureFormat::RGBA8;
                renderTexture.width = frame.width;
                renderTexture.height = frame.height;
                renderTexture.isYUV = frame.isYUV;
                renderTexture.software.data = frame.data;
                renderTexture.software.pitch = frame.pitch;
                return renderTexture;
            }
            break;
        }
        
        case RendererType::OpenGL: {
            // OpenGL renderer - check for CUDA interop first
#if USE_OPENGL_RENDERER && HAVE_CUDA
            if (renderer->SupportsCudaInterop() && frame.isHardwareCuda && frame.cudaPtr) {
                RenderTexture renderTexture;
                renderTexture.type = TextureType::CUDA;
                renderTexture.format = frame.isYUV ? TextureFormat::NV12 : TextureFormat::RGBA8;
                renderTexture.width = frame.width;
                renderTexture.height = frame.height;
                renderTexture.isYUV = frame.isYUV;
                renderTexture.cuda.devicePtr = frame.cudaPtr;
                renderTexture.cuda.pitch = frame.cudaPitch;
                renderTexture.cuda.glResource = frame.cudaResource;
                return renderTexture;
            }
#endif
            // Use software texture for OpenGL renderer
            if (frame.data) {
                RenderTexture renderTexture;
                renderTexture.type = TextureType::Software;
                renderTexture.format = TextureFormat::RGBA8;
                renderTexture.width = frame.width;
                renderTexture.height = frame.height;
                renderTexture.isYUV = frame.isYUV;
                renderTexture.software.data = frame.data;
                renderTexture.software.pitch = frame.pitch;
                return renderTexture;
            }
            break;
        }
    }
    
    // No suitable texture data found
    return CreateNullTexture();
}

RenderTexture TextureConverter::CreateNullTexture() {
    return RenderTexture::CreateNull();
}
#include "FFmpegInitializer.h"
#include "video/VideoValidator.h"
#include "video/decode/HardwareDecoder.h"

FFmpegInitializer::~FFmpegInitializer() {
    if (initialized) {
        HardwareDecoder::Cleanup();
        VideoValidator::Cleanup();
    }
}

bool FFmpegInitializer::Initialize() {
    if (!VideoValidator::Initialize()) {
        return false;
    }
    
    if (!HardwareDecoder::Initialize()) {
        VideoValidator::Cleanup();
        return false;
    }
    
    initialized = true;
    return true;
}
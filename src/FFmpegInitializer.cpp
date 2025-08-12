#include "FFmpegInitializer.h"
#include "VideoValidator.h"
#include "HardwareDecoder.h"

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
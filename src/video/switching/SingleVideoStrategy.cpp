#include "SingleVideoStrategy.h"
#include "core/Logger.h"
#include <algorithm>
#include <cmath>

extern "C" {
#include <libavcodec/avcodec.h>
}

SingleVideoStrategy::SingleVideoStrategy() {
    m_activeVideoIndex = 0;
}

SingleVideoStrategy::~SingleVideoStrategy() {
    Cleanup();
}

bool SingleVideoStrategy::Initialize(std::vector<VideoStream>* streams, VideoManager* manager) {
    if (!streams || streams->empty()) {
        LOG_ERROR("SingleVideoStrategy: Cannot initialize with empty streams");
        return false;
    }

    if (streams->size() != 1) {
        LOG_ERROR("SingleVideoStrategy: Expected exactly 1 video stream, got ", streams->size());
        return false;
    }

    m_streams = streams;
    m_manager = manager;
    m_activeVideoIndex = 0;

    LOG_INFO("SingleVideoStrategy initialized for single video playback");
    return true;
}

bool SingleVideoStrategy::SwitchToVideo(size_t targetVideoIndex, double currentTime) {
    // In single video mode, there's nothing to switch to
    // Just ignore switch requests
    if (targetVideoIndex != 0) {
        LOG_DEBUG("SingleVideoStrategy: Ignoring switch request to video ", (targetVideoIndex + 1), " (only 1 video available)");
    }
    return true;
}

bool SingleVideoStrategy::UpdateFrame() {
    VideoStream& stream = (*m_streams)[0];

    if (!ProcessVideoFrame(stream)) {
        // Check if we reached end of stream
        if (stream.state == VideoState::END_OF_STREAM) {
            LOG_DEBUG("End of stream reached, handling...");
            if (!HandleEndOfStream(stream)) {
                return false;
            }
        } else {
            LOG_ERROR("Failed to process video frame");
            return false;
        }
    }

    return true;
}

DecodedFrame* SingleVideoStrategy::GetCurrentFrame() {
    VideoStream& stream = (*m_streams)[0];
    if (stream.currentFrame.valid) {
        return &stream.currentFrame;
    }

    return nullptr;
}

void SingleVideoStrategy::Cleanup() {
    // Nothing special to cleanup for single video strategy
}

std::string SingleVideoStrategy::GetName() const {
    return "Single Video";
}

bool SingleVideoStrategy::ProcessVideoFrame(VideoStream& stream) {
    if (!DecodeNextFrame(stream)) {
        return false;
    }

    return true;
}

bool SingleVideoStrategy::HandleEndOfStream(VideoStream& stream) {
    LOG_INFO("End of stream reached, looping video");

    // Seek back to the beginning for looping
    if (!SeekVideoStream(stream, 0.0)) {
        LOG_ERROR("Failed to seek to beginning for looping");
        return false;
    }

    stream.currentTime = 0.0;
    stream.state = VideoState::PLAYING;

    // Decode the first frame after seeking
    if (!DecodeNextFrame(stream)) {
        LOG_ERROR("Failed to decode first frame after looping");
        return false;
    }

    return true;
}

bool SingleVideoStrategy::SeekVideoStream(VideoStream& stream, double targetTime) {
    // Convert target time to stream time base
    AVRational timeBase = stream.demuxer.GetTimeBase();
    int64_t seekTarget = static_cast<int64_t>(targetTime / av_q2d(timeBase));

    // Seek the demuxer
    if (!stream.demuxer.SeekToTime(seekTarget)) {
        LOG_ERROR("Failed to seek demuxer to time ", targetTime);
        return false;
    }

    // Flush decoder to clear any cached frames
    stream.decoder.Flush();

    return true;
}

bool SingleVideoStrategy::DecodeNextFrame(VideoStream& stream) {
    const int maxDecodeAttempts = 10;
    int attempts = 0;

    while (attempts < maxDecodeAttempts) {
        attempts++;

        // Need to send more packets to decoder
        AVPacket packet;
        if (!stream.demuxer.ReadFrame(&packet)) {
            // End of stream
            stream.state = VideoState::END_OF_STREAM;
            return false;
        }

        // Send packet to decoder
        if (!stream.decoder.SendPacket(&packet)) {
            LOG_WARNING("Failed to send packet to decoder");
            av_packet_unref(&packet);
            continue;
        }

        av_packet_unref(&packet);

        // Try to receive frame again
        if (stream.decoder.ReceiveFrame(stream.currentFrame)) {
            stream.state = VideoState::PLAYING;
            return true;
        }
    }

    LOG_ERROR("Failed to decode frame after ", maxDecodeAttempts, " attempts");
    return false;
}
#include "VideoDemuxer.h"
#include <iostream>

VideoDemuxer::VideoDemuxer() 
    : m_formatContext(nullptr)
    , m_videoStreamIndex(-1)
    , m_videoStream(nullptr) {
}

VideoDemuxer::~VideoDemuxer() {
    Close();
}

bool VideoDemuxer::Open(const std::string& filePath) {
    Close();
    
    // Open input file
    int ret = avformat_open_input(&m_formatContext, filePath.c_str(), nullptr, nullptr);
    if (ret < 0) {
        char errorBuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, errorBuf, sizeof(errorBuf));
        std::cerr << "Cannot open file " << filePath << ": " << errorBuf << "\n";
        return false;
    }
    
    // Retrieve stream information
    ret = avformat_find_stream_info(m_formatContext, nullptr);
    if (ret < 0) {
        char errorBuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, errorBuf, sizeof(errorBuf));
        std::cerr << "Cannot find stream info for " << filePath << ": " << errorBuf << "\n";
        Close();
        return false;
    }
    
    // Find video stream
    if (!FindVideoStream()) {
        std::cerr << "No video stream found in " << filePath << "\n";
        Close();
        return false;
    }
    
    std::cout << "Successfully opened video file: " << filePath << "\n";
    std::cout << "  Resolution: " << GetWidth() << "x" << GetHeight() << "\n";
    std::cout << "  Frame rate: " << GetFrameRate() << " FPS\n";
    std::cout << "  Duration: " << GetDuration() << " seconds\n";
    AVRational timebase = GetTimeBase();
    std::cout << "  Timebase: " << timebase.num << "/" << timebase.den 
              << " (" << av_q2d(timebase) << " seconds per unit)\n";
    
    return true;
}

void VideoDemuxer::Close() {
    Reset();
}

bool VideoDemuxer::ReadFrame(AVPacket* packet) {
    if (!m_formatContext || m_videoStreamIndex < 0) {
        std::cerr << "DEBUG: ReadFrame failed - no format context or invalid video stream index\n";
        return false;
    }
    
    while (true) {
        int ret = av_read_frame(m_formatContext, packet);
        if (ret < 0) {
            if (ret == AVERROR_EOF) {
                std::cout << "DEBUG: End of file reached\n";
            } else {
                char errorBuf[AV_ERROR_MAX_STRING_SIZE];
                av_strerror(ret, errorBuf, sizeof(errorBuf));
                std::cerr << "DEBUG: av_read_frame failed: " << errorBuf << " (ret=" << ret << ")\n";
            }
            return false;
        }
        
        // Only return packets from the video stream
        if (packet->stream_index == m_videoStreamIndex) {
            std::cout << "DEBUG: Read video packet - Size: " << packet->size 
                     << ", PTS: " << packet->pts 
                     << ", DTS: " << packet->dts 
                     << ", Stream: " << packet->stream_index
                     << ", Flags: " << packet->flags << "\n";
            return true;
        }
        
        // Free packet if it's not from the video stream
        std::cout << "DEBUG: Skipping non-video packet from stream " << packet->stream_index << "\n";
        av_packet_unref(packet);
    }
}

bool VideoDemuxer::SeekToTime(double timeInSeconds) {
    if (!m_formatContext || !m_videoStream) {
        return false;
    }
    
    int64_t timestamp = SecondsToPacketTime(timeInSeconds);
    
    int ret = av_seek_frame(m_formatContext, m_videoStreamIndex, timestamp, AVSEEK_FLAG_BACKWARD);
    if (ret < 0) {
        char errorBuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, errorBuf, sizeof(errorBuf));
        std::cerr << "Seek failed: " << errorBuf << "\n";
        return false;
    }
    
    return true;
}

bool VideoDemuxer::SeekToFrame(int64_t frameNumber) {
    if (!m_formatContext || !m_videoStream) {
        return false;
    }
    
    double timeInSeconds = frameNumber / GetFrameRate();
    return SeekToTime(timeInSeconds);
}

double VideoDemuxer::GetDuration() const {
    if (!m_formatContext) {
        return 0.0;
    }
    
    if (m_formatContext->duration != AV_NOPTS_VALUE) {
        return static_cast<double>(m_formatContext->duration) / AV_TIME_BASE;
    }
    
    return 0.0;
}

double VideoDemuxer::GetFrameRate() const {
    if (!m_videoStream) {
        return 0.0;
    }
    
    if (m_videoStream->avg_frame_rate.num != 0 && m_videoStream->avg_frame_rate.den != 0) {
        return av_q2d(m_videoStream->avg_frame_rate);
    } else if (m_videoStream->r_frame_rate.num != 0 && m_videoStream->r_frame_rate.den != 0) {
        return av_q2d(m_videoStream->r_frame_rate);
    }
    
    return 25.0; // Default fallback
}

int VideoDemuxer::GetWidth() const {
    if (!m_videoStream) {
        return 0;
    }
    return m_videoStream->codecpar->width;
}

int VideoDemuxer::GetHeight() const {
    if (!m_videoStream) {
        return 0;
    }
    return m_videoStream->codecpar->height;
}

AVCodecID VideoDemuxer::GetCodecID() const {
    if (!m_videoStream) {
        return AV_CODEC_ID_NONE;
    }
    return m_videoStream->codecpar->codec_id;
}

AVCodecParameters* VideoDemuxer::GetCodecParameters() const {
    if (!m_videoStream) {
        return nullptr;
    }
    return m_videoStream->codecpar;
}

int VideoDemuxer::GetVideoStreamIndex() const {
    return m_videoStreamIndex;
}

AVRational VideoDemuxer::GetTimeBase() const {
    if (!m_videoStream) {
        return {0, 1};
    }
    return m_videoStream->time_base;
}

double VideoDemuxer::PacketTimeToSeconds(int64_t pts) const {
    if (!m_videoStream || pts == AV_NOPTS_VALUE) {
        return 0.0;
    }
    return static_cast<double>(pts) * av_q2d(m_videoStream->time_base);
}

int64_t VideoDemuxer::SecondsToPacketTime(double seconds) const {
    if (!m_videoStream) {
        return 0;
    }
    return static_cast<int64_t>(seconds / av_q2d(m_videoStream->time_base));
}

bool VideoDemuxer::IsValidPacket(const AVPacket* packet) const {
    return packet && packet->stream_index == m_videoStreamIndex;
}

bool VideoDemuxer::FindVideoStream() {
    if (!m_formatContext) {
        return false;
    }
    
    for (unsigned int i = 0; i < m_formatContext->nb_streams; i++) {
        if (m_formatContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            m_videoStreamIndex = i;
            m_videoStream = m_formatContext->streams[i];
            
            // Validate codec support (H264/H265 only)
            AVCodecID codecId = m_videoStream->codecpar->codec_id;
            if (codecId != AV_CODEC_ID_H264 && codecId != AV_CODEC_ID_HEVC) {
                std::cerr << "Unsupported video codec found. Only H264 and H265 are supported.\n";
                return false;
            }
            
            return true;
        }
    }
    
    return false;
}

void VideoDemuxer::Reset() {
    if (m_formatContext) {
        avformat_close_input(&m_formatContext);
        m_formatContext = nullptr;
    }
    
    m_videoStreamIndex = -1;
    m_videoStream = nullptr;
}
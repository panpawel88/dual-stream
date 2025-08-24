# Video Demuxing System

This directory implements the video demuxing (container parsing) functionality, responsible for extracting video packets from MP4 container files.

## Architecture Overview

The demuxing system provides a clean abstraction over FFmpeg's libavformat, handling MP4 container parsing and video stream extraction with precise seeking capabilities.

## Core Component

### VideoDemuxer
**File:** `VideoDemuxer.h/cpp`
**Purpose:** MP4 container parsing and video packet extraction

**Key Responsibilities:**
- MP4 file opening and stream information retrieval
- Video stream identification and validation
- Packet reading and filtering
- Temporal seeking with frame accuracy
- Stream metadata extraction

## Container Support

### Supported Formats
- **Container:** MP4 (MPEG-4 Part 14)
- **Video Codecs:** H.264 (AVC) and H.265 (HEVC) only
- **Stream Types:** Video streams only (audio streams ignored)

### Codec Validation
```cpp
bool FindVideoStream() {
    // Locate video stream in container
    for (unsigned int i = 0; i < m_formatContext->nb_streams; i++) {
        if (streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            // Validate codec support
            AVCodecID codecId = streams[i]->codecpar->codec_id;
            if (codecId != AV_CODEC_ID_H264 && codecId != AV_CODEC_ID_HEVC) {
                LOG_ERROR("Unsupported video codec. Only H264 and H265 supported.");
                return false;
            }
        }
    }
}
```

## Stream Information Extraction

### Video Properties
```cpp
// Resolution
int GetWidth() const { return m_videoStream->codecpar->width; }
int GetHeight() const { return m_videoStream->codecpar->height; }

// Timing information
double GetDuration() const { 
    return static_cast<double>(m_formatContext->duration) / AV_TIME_BASE; 
}

double GetFrameRate() const {
    // Prefer average frame rate, fallback to r_frame_rate
    if (m_videoStream->avg_frame_rate.num != 0) {
        return av_q2d(m_videoStream->avg_frame_rate);
    } else if (m_videoStream->r_frame_rate.num != 0) {
        return av_q2d(m_videoStream->r_frame_rate);
    }
    return 25.0; // Default fallback
}

// Stream metadata
AVRational GetTimeBase() const { return m_videoStream->time_base; }
AVCodecParameters* GetCodecParameters() const { return m_videoStream->codecpar; }
```

## Packet Reading and Filtering

### Stream-Specific Packet Reading
```cpp
bool ReadFrame(AVPacket* packet) {
    while (true) {
        int ret = av_read_frame(m_formatContext, packet);
        if (ret < 0) return false; // EOF or error
        
        // Filter for video stream packets only
        if (packet->stream_index == m_videoStreamIndex) {
            LOG_DEBUG("Read video packet - Size: ", packet->size, 
                     ", PTS: ", packet->pts, ", DTS: ", packet->dts);
            return true;
        }
        
        // Skip non-video packets
        av_packet_unref(packet);
    }
}
```

### Packet Validation
```cpp
bool IsValidPacket(const AVPacket* packet) const {
    return packet && packet->stream_index == m_videoStreamIndex;
}
```

## Seeking System

### Time-Based Seeking
```cpp
bool SeekToTime(double timeInSeconds) {
    int64_t timestamp = SecondsToPacketTime(timeInSeconds);
    
    // Use backward seeking for keyframe alignment
    int ret = av_seek_frame(m_formatContext, m_videoStreamIndex, 
                           timestamp, AVSEEK_FLAG_BACKWARD);
    
    return ret >= 0;
}
```

### Frame-Based Seeking
```cpp
bool SeekToFrame(int64_t frameNumber) {
    double timeInSeconds = frameNumber / GetFrameRate();
    return SeekToTime(timeInSeconds);
}
```

### Timestamp Conversion Utilities
```cpp
// Convert packet timestamps to seconds using stream timebase
double PacketTimeToSeconds(int64_t pts) const {
    if (pts == AV_NOPTS_VALUE) return 0.0;
    return static_cast<double>(pts) * av_q2d(m_videoStream->time_base);
}

// Convert seconds to packet timestamps
int64_t SecondsToPacketTime(double seconds) const {
    return static_cast<int64_t>(seconds / av_q2d(m_videoStream->time_base));
}
```

## Error Handling and Robustness

### File Opening Validation
```cpp
bool Open(const std::string& filePath) {
    // Open input file with error reporting
    int ret = avformat_open_input(&m_formatContext, filePath.c_str(), nullptr, nullptr);
    if (ret < 0) {
        char errorBuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, errorBuf, sizeof(errorBuf));
        LOG_ERROR("Cannot open file ", filePath, ": ", errorBuf);
        return false;
    }
    
    // Retrieve and validate stream information
    ret = avformat_find_stream_info(m_formatContext, nullptr);
    if (ret < 0) {
        // Error handling and cleanup
    }
}
```

### Resource Management
```cpp
void VideoDemuxer::Reset() {
    if (m_formatContext) {
        avformat_close_input(&m_formatContext);  // Automatically frees context
        m_formatContext = nullptr;
    }
    m_videoStreamIndex = -1;
    m_videoStream = nullptr;
}

VideoDemuxer::~VideoDemuxer() {
    Close();  // RAII cleanup
}
```

## Integration with Decoder System

### Codec Parameter Passing
```cpp
// VideoManager initializes decoder with demuxer's codec parameters
DecoderInfo decoderInfo = HardwareDecoder::GetBestDecoder(stream.demuxer.GetCodecID());
stream.decoder.Initialize(
    stream.demuxer.GetCodecParameters(),  // Codec parameters from demuxer
    decoderInfo,
    d3dDevice,
    stream.demuxer.GetTimeBase(),         // Stream timebase for PTS conversion
    cudaInteropAvailable
);
```

### Packet Flow
```cpp
// Typical decode loop pattern
AVPacket packet;
while (demuxer.ReadFrame(&packet)) {
    if (demuxer.IsValidPacket(&packet)) {
        decoder.SendPacket(&packet);
        
        DecodedFrame frame;
        if (decoder.ReceiveFrame(frame)) {
            // Process decoded frame
        }
    }
    av_packet_unref(&packet);
}
```

## Performance Characteristics

### Seeking Performance
- **Backward Seeking:** Uses `AVSEEK_FLAG_BACKWARD` for keyframe alignment
- **Stream Filtering:** Efficient packet filtering at demux level
- **Metadata Caching:** Stream information cached after initial parsing

### Memory Efficiency
- **Single Stream Focus:** Only video stream packets are retained
- **Automatic Cleanup:** RAII pattern ensures proper resource management
- **Packet Lifecycle:** Non-video packets immediately released

## Logging and Debugging

### Comprehensive Logging
```cpp
LOG_INFO("Successfully opened video file: ", filePath);
LOG_INFO("  Resolution: ", GetWidth(), "x", GetHeight());
LOG_INFO("  Frame rate: ", GetFrameRate(), " FPS");
LOG_INFO("  Duration: ", GetDuration(), " seconds");
LOG_INFO("  Timebase: ", timebase.num, "/", timebase.den);
```

### Debug Packet Information
```cpp
LOG_DEBUG("Read video packet - Size: ", packet->size,
          ", PTS: ", packet->pts, ", DTS: ", packet->dts,
          ", Stream: ", packet->stream_index, ", Flags: ", packet->flags);
```

## Thread Safety

### Single-Threaded Design
- **Thread Safety:** Not thread-safe by design (single playback thread model)
- **State Management:** Internal state maintained per instance
- **Concurrent Access:** Multiple `VideoDemuxer` instances safe for different files

### Integration Notes
- Each `VideoStream` in `VideoManager` has its own `VideoDemuxer` instance
- No shared state between demuxer instances
- Safe for multi-video playback scenarios

This demuxing system provides reliable MP4 container parsing with robust error handling and precise seeking capabilities, forming the foundation for the video decoding pipeline.
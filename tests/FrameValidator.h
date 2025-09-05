#pragma once

#include <string>
#include <vector>
#include <regex>
#include <memory>
#include <map>

// Forward declarations
struct DecodedFrame;
class IRenderer;

/**
 * Frame validation system for test videos with embedded frame numbers and patterns.
 * Analyzes rendered frames to extract frame numbers and validate visual patterns.
 */
class FrameValidator {
public:
    struct FrameAnalysis {
        bool hasValidFrameNumber = false;
        int extractedFrameNumber = -1;
        double timestamp = 0.0;
        
        // Visual pattern validation
        bool hasCornerMarkers = false;
        struct {
            bool topLeft = false;
            bool topRight = false; 
            bool bottomLeft = false;
            bool bottomRight = false;
        } cornerMarkers;
        
        // Object detection for moving patterns
        bool hasMovingObject = false;
        struct {
            float x = 0.0f;
            float y = 0.0f;
            float expectedX = 0.0f;
            float expectedY = 0.0f;
            float positionError = 0.0f;
        } objectPosition;
        
        // Quality metrics
        double averagePixelValue = 0.0;
        bool hasTextArtifacts = false;
        std::string detectedVideoName;
    };

    FrameValidator();
    ~FrameValidator();

    // Frame analysis methods
    FrameAnalysis ValidateFrame(const DecodedFrame& frame, double expectedTimestamp);
    FrameAnalysis ValidateRenderedFrame(IRenderer* renderer, double expectedTimestamp);
    
    // Pattern-specific validation
    bool ValidateMovingSquarePattern(const DecodedFrame& frame, double timestamp);
    bool ValidateMovingCirclePattern(const DecodedFrame& frame, double timestamp);
    bool ValidateBouncing BallPattern(const DecodedFrame& frame, double timestamp);
    bool ValidateGradientPattern(const DecodedFrame& frame, double timestamp);
    
    // Frame number extraction
    int ExtractFrameNumber(const DecodedFrame& frame);
    int ExtractFrameNumberFromRenderedFrame(IRenderer* renderer);
    
    // Corner marker detection
    bool DetectCornerMarkers(const DecodedFrame& frame);
    
    // Configuration
    void SetExpectedVideoPattern(const std::string& videoPath);
    void SetFrameNumberRegex(const std::string& regex);
    
    // Statistics
    struct ValidationStats {
        int totalFramesAnalyzed = 0;
        int framesWithValidNumbers = 0;
        int framesWithCornerMarkers = 0;
        int framesWithMovingObjects = 0;
        double averagePositionError = 0.0;
        std::vector<int> detectedFrameNumbers;
        std::map<std::string, int> detectedVideoNames;
    };
    
    ValidationStats GetStatistics() const { return m_stats; }
    void ResetStatistics();

private:
    // Frame number extraction methods
    int ExtractFrameNumberFromPixels(const uint8_t* pixelData, int width, int height, int stride);
    std::string ExtractTextFromRegion(const uint8_t* pixelData, int width, int height, int x, int y, int w, int h);
    
    // Pattern calculation methods
    struct ExpectedPosition {
        float x, y;
    };
    ExpectedPosition CalculateMovingSquarePosition(double timestamp);
    ExpectedPosition CalculateMovingCirclePosition(double timestamp);
    ExpectedPosition CalculateBouncingBallPosition(double timestamp);
    
    // Color detection for corner markers
    bool DetectColorAtPosition(const uint8_t* pixelData, int width, int height, int x, int y, 
                              uint8_t expectedR, uint8_t expectedG, uint8_t expectedB, int tolerance = 30);
    
    // OCR/Text recognition helpers
    bool InitializeOCR();
    std::string PerformOCROnRegion(const uint8_t* pixelData, int width, int height, int x, int y, int w, int h);

private:
    ValidationStats m_stats;
    std::string m_expectedVideoPattern;
    std::regex m_frameNumberRegex;
    
    // OCR system (optional - could use Tesseract or simple pattern matching)
    void* m_ocrEngine = nullptr;
    bool m_ocrAvailable = false;
    
    // Expected patterns based on video type
    enum class VideoPattern {
        MovingSquare,   // Red background, white moving square
        MovingCircle,   // Blue background, yellow moving circle  
        BouncingBall,   // Green background, red bouncing square
        Gradient,       // Color gradient animation
        Text,           // Text rotation animation
        Unknown
    } m_videoPattern = VideoPattern::Unknown;
};
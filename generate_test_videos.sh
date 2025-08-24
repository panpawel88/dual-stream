#!/bin/bash

echo "Generating test video files for FFmpeg Video Player..."
echo

# Check if FFmpeg is available
if ! command -v ffmpeg &> /dev/null; then
    echo "ERROR: FFmpeg not found in PATH. Please install FFmpeg first."
    echo "On Ubuntu/Debian: sudo apt install ffmpeg"
    echo "On macOS: brew install ffmpeg"
    echo "On other systems: https://ffmpeg.org/download.html"
    exit 1
fi

# Create test_videos directory if it doesn't exist
mkdir -p test_videos

echo "Creating Test Video 1 - Moving White Square with Frame Numbers (H264, 1280x720, 30fps, 10 seconds)..."
ffmpeg -f lavfi -i "color=red:size=1280x720:duration=10:rate=30" -f lavfi -i "color=white:size=100x100:duration=10:rate=30" \
       -filter_complex "[0:v][1:v]overlay=x='50+200*sin(2*PI*t)':y='100+100*sin(2*PI*t)'[v];[v]drawtext=text='Video 1 - Frame %{frame_num}':fontsize=24:fontcolor=white:x=20:y=20:box=1:boxcolor=black@0.8[out]" \
       -map "[out]" \
       -c:v libx264 -preset medium -crf 23 -pix_fmt yuv420p \
       -movflags +faststart \
       test_videos/video1_red_square.mp4 -y

echo
echo "Creating Test Video 2 - Moving Yellow Circle with Frame Numbers (H264, 1280x720, 30fps, 10 seconds)..."
ffmpeg -f lavfi -i "color=blue:size=1280x720:duration=10:rate=30" -f lavfi -i "color=yellow:size=80x80:duration=10:rate=30" \
       -filter_complex "[1:v]geq=lum='if(lt(sqrt((X-40)*(X-40)+(Y-40)*(Y-40)),40),255,0)':cb=128:cr=128[circle];[0:v][circle]overlay=x='100+200*cos(2*PI*t)':y='150+150*sin(2*PI*t)'[v];[v]drawtext=text='Video 2 - Frame %{frame_num}':fontsize=24:fontcolor=yellow:x=20:y=20:box=1:boxcolor=black@0.8[out]" \
       -map "[out]" \
       -c:v libx264 -preset medium -crf 23 -pix_fmt yuv420p \
       -movflags +faststart \
       test_videos/video2_blue_circle.mp4 -y

echo
echo "Creating Test Video 3 - Color Gradient Animation with Frame Numbers (H265, 1280x720, 30fps, 8 seconds)..."
ffmpeg -f lavfi -i "color=red:size=1280x720:duration=8:rate=30" \
       -vf "hue=h='360*t/8':s=1,drawtext=text='Video 3 H265 - Frame %{frame_num}':fontsize=24:fontcolor=white:x=20:y=20:box=1:boxcolor=black@0.9" \
       -c:v libx265 -preset medium -crf 28 -pix_fmt yuv420p \
       -movflags +faststart \
       test_videos/video3_gradient.mp4 -y

echo
echo "Creating Test Video 4 - Bouncing Ball with Frame Numbers (H264, 1280x720, 30fps, 12 seconds)..."
ffmpeg -f lavfi -i "color=green:size=1280x720:duration=12:rate=30" \
       -vf "drawbox=x=50+abs(200*sin(PI*t)):y=50+abs(150*sin(1.5*PI*t)):w=50:h=50:color=red:t=fill,drawtext=text='Video 4 - Frame %{frame_num} - FPS: 30':fontsize=22:fontcolor=white:x=20:y=20:box=1:boxcolor=black@0.8" \
       -c:v libx264 -preset medium -crf 23 -pix_fmt yuv420p \
       -movflags +faststart \
       test_videos/video4_bouncing_ball.mp4 -y

echo
echo "Creating Test Video 5 - Rotating Text with Frame Numbers (H265, 1280x720, 30fps, 15 seconds)..."
ffmpeg -f lavfi -i "color=black:size=1280x720:duration=15:rate=30" \
       -vf "drawtext=text='FFmpeg Video Player':fontsize=60:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2+50*sin(2*PI*t/3):enable='between(t,0,15)',drawtext=text='Video 5 H265 - Frame %{frame_num}':fontsize=24:fontcolor=yellow:x=20:y=20:box=1:boxcolor=black@0.8" \
       -c:v libx265 -preset medium -crf 28 -pix_fmt yuv420p \
       -movflags +faststart \
       test_videos/video5_text.mp4 -y

echo
echo "Creating smaller test videos for quick testing..."

echo "Creating Short Video A - Red fade with Frame Numbers (H264, 1280x720, 60fps, 3 seconds)..."
ffmpeg -f lavfi -i "color=red:size=1280x720:duration=3:rate=60" \
       -vf "fade=t=in:st=0:d=1,fade=t=out:st=2:d=1,drawtext=text='SHORT A - Frame %{frame_num} (60fps)':fontsize=28:fontcolor=white:x=20:y=20:box=1:boxcolor=black@0.9" \
       -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p \
       -movflags +faststart \
       test_videos/short_a_red_fade.mp4 -y

echo "Creating Short Video B - Blue pulse with Frame Numbers (H264, 1280x720, 60fps, 3 seconds)..."
ffmpeg -f lavfi -i "color=blue:size=1280x720:duration=3:rate=60" \
       -vf "drawbox=x=(w-100*sin(2*PI*t*2))/2:y=(h-100*sin(2*PI*t*2))/2:w=100*sin(2*PI*t*2):h=100*sin(2*PI*t*2):color=white:t=fill,drawtext=text='SHORT B - Frame %{frame_num} (60fps)':fontsize=28:fontcolor=cyan:x=20:y=20:box=1:boxcolor=black@0.9" \
       -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p \
       -movflags +faststart \
       test_videos/short_b_blue_pulse.mp4 -y

echo
echo "Test video generation complete!"
echo
echo "Generated files:"
echo "================"
ls -la test_videos/*.mp4
echo
echo "Usage examples:"
echo "---------------"
echo "./ffmpeg_player test_videos/video1_red_square.mp4 test_videos/video2_blue_circle.mp4"
echo "./ffmpeg_player test_videos/video3_gradient.mp4 test_videos/video4_bouncing_ball.mp4"
echo "./ffmpeg_player test_videos/short_a_red_fade.mp4 test_videos/short_b_blue_pulse.mp4"
echo
echo "Creating 4K Performance Test Videos..."
echo

echo "Creating 4K Video 1 - Moving White Square (H264, 3840x2160, 30fps, 10 seconds)..."
ffmpeg -f lavfi -i "color=red:size=3840x2160:duration=10:rate=30" -f lavfi -i "color=white:size=300x300:duration=10:rate=30" \
       -filter_complex "[0:v][1:v]overlay=x='150+600*sin(2*PI*t)':y='300+300*sin(2*PI*t)'[v];[v]drawtext=text='4K Video 1 - Frame %{frame_num}':fontsize=72:fontcolor=white:x=60:y=60:box=1:boxcolor=black@0.8[out]" \
       -map "[out]" \
       -c:v libx264 -preset medium -crf 23 -pix_fmt yuv420p \
       -movflags +faststart \
       test_videos/4k_video1_red_square.mp4 -y

echo "Creating 4K Video 2 - Moving Yellow Circle (H264, 3840x2160, 30fps, 10 seconds)..."
ffmpeg -f lavfi -i "color=blue:size=3840x2160:duration=10:rate=30" -f lavfi -i "color=yellow:size=240x240:duration=10:rate=30" \
       -filter_complex "[1:v]geq=lum='if(lt(sqrt((X-120)*(X-120)+(Y-120)*(Y-120)),120),255,0)':cb=128:cr=128[circle];[0:v][circle]overlay=x='300+600*cos(2*PI*t)':y='450+450*sin(2*PI*t)'[v];[v]drawtext=text='4K Video 2 - Frame %{frame_num}':fontsize=72:fontcolor=yellow:x=60:y=60:box=1:boxcolor=black@0.8[out]" \
       -map "[out]" \
       -c:v libx264 -preset medium -crf 23 -pix_fmt yuv420p \
       -movflags +faststart \
       test_videos/4k_video2_blue_circle.mp4 -y

echo
echo "Creating 8K Performance Test Videos..."
echo

echo "Creating 8K Video 1 - Moving White Square (H265, 7680x4320, 30fps, 8 seconds)..."
ffmpeg -f lavfi -i "color=red:size=7680x4320:duration=8:rate=30" -f lavfi -i "color=white:size=600x600:duration=8:rate=30" \
       -filter_complex "[0:v][1:v]overlay=x='300+1200*sin(2*PI*t)':y='600+600*sin(2*PI*t)'[v];[v]drawtext=text='8K Video 1 - Frame %{frame_num}':fontsize=144:fontcolor=white:x=120:y=120:box=1:boxcolor=black@0.8[out]" \
       -map "[out]" \
       -c:v libx265 -preset medium -crf 28 -pix_fmt yuv420p \
       -movflags +faststart \
       test_videos/8k_video1_red_square.mp4 -y

echo "Creating 8K Video 2 - Moving Yellow Circle (H265, 7680x4320, 30fps, 8 seconds)..."
ffmpeg -f lavfi -i "color=blue:size=7680x4320:duration=8:rate=30" -f lavfi -i "color=yellow:size=480x480:duration=8:rate=30" \
       -filter_complex "[1:v]geq=lum='if(lt(sqrt((X-240)*(X-240)+(Y-240)*(Y-240)),240),255,0)':cb=128:cr=128[circle];[0:v][circle]overlay=x='600+1200*cos(2*PI*t)':y='900+900*sin(2*PI*t)'[v];[v]drawtext=text='8K Video 2 - Frame %{frame_num}':fontsize=144:fontcolor=yellow:x=120:y=120:box=1:boxcolor=black@0.8[out]" \
       -map "[out]" \
       -c:v libx265 -preset medium -crf 28 -pix_fmt yuv420p \
       -movflags +faststart \
       test_videos/8k_video2_blue_circle.mp4 -y

echo
echo "All videos generated successfully!"
echo
echo "Standard HD videos (1280x720) for compatibility testing:"
echo "4K videos (3840x2160) for performance testing:"
echo "8K videos (7680x4320) for extreme performance testing:"
echo "Press 1/2 to switch between videos, ESC to exit."
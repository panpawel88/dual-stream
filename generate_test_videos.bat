@echo off
echo Generating test video files for FFmpeg Video Player...
echo.

REM Check if FFmpeg is available
ffmpeg -version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: FFmpeg not found in PATH. Please install FFmpeg first.
    echo You can download it from: https://ffmpeg.org/download.html
    pause
    exit /b 1
)

REM Create test_videos directory if it doesn't exist
if not exist "test_videos" mkdir test_videos

echo Creating Test Video 1 - Moving White Square with Frame Numbers (H264, 1280x720, 30fps, 10 seconds)...
ffmpeg -f lavfi -i "color=red:size=1280x720:duration=10:rate=30" -f lavfi -i "color=white:size=100x100:duration=10:rate=30" ^
       -filter_complex "[0:v][1:v]overlay=x='50+200*sin(2*PI*t)':y='100+100*sin(2*PI*t)'[v];[v]drawtext=text='Video 1 - Frame %%{frame_num}':fontsize=24:fontcolor=white:x=20:y=20:box=1:boxcolor=black@0.8[out]" ^
       -map "[out]" ^
       -c:v libx264 -preset medium -crf 23 -g 30 -keyint_min 30 -pix_fmt yuv420p ^
       -movflags +faststart ^
       test_videos/video1_red_square.mp4 -y

echo.
echo Creating Test Video 2 - Moving Yellow Circle with Frame Numbers (H264, 1280x720, 30fps, 10 seconds)...
ffmpeg -f lavfi -i "color=blue:size=1280x720:duration=10:rate=30" -f lavfi -i "color=yellow:size=80x80:duration=10:rate=30" ^
       -filter_complex "[1:v]geq=lum='if(lt(sqrt((X-40)*(X-40)+(Y-40)*(Y-40)),40),255,0)':cb=128:cr=128[circle];[0:v][circle]overlay=x='100+200*cos(2*PI*t)':y='150+150*sin(2*PI*t)'[v];[v]drawtext=text='Video 2 - Frame %%{frame_num}':fontsize=24:fontcolor=yellow:x=20:y=20:box=1:boxcolor=black@0.8[out]" ^
       -map "[out]" ^
       -c:v libx264 -preset medium -crf 23 -g 30 -keyint_min 30 -pix_fmt yuv420p ^
       -movflags +faststart ^
       test_videos/video2_blue_circle.mp4 -y

echo.
echo Creating Test Video 3 - Color Gradient Animation with Frame Numbers (H265, 1280x720, 30fps, 8 seconds)...
ffmpeg -f lavfi -i "color=red:size=1280x720:duration=8:rate=30" ^
       -vf "hue=h='360*t/8':s=1,drawtext=text='Video 3 H265 - Frame %%{frame_num}':fontsize=24:fontcolor=white:x=20:y=20:box=1:boxcolor=black@0.9" ^
       -c:v libx265 -preset medium -crf 28 -g 30 -keyint_min 30 -pix_fmt yuv420p ^
       -movflags +faststart ^
       test_videos/video3_gradient.mp4 -y

echo.
echo Creating Test Video 4 - Bouncing Ball with Frame Numbers (H264, 1280x720, 30fps, 12 seconds)...
ffmpeg -f lavfi -i "color=green:size=1280x720:duration=12:rate=30" ^
       -vf "drawbox=x=50+abs(200*sin(PI*t)):y=50+abs(150*sin(1.5*PI*t)):w=50:h=50:color=red:t=fill,drawtext=text='Video 4 - Frame %%{frame_num} - FPS: 30':fontsize=22:fontcolor=white:x=20:y=20:box=1:boxcolor=black@0.8" ^
       -c:v libx264 -preset medium -crf 23 -g 30 -keyint_min 30 -pix_fmt yuv420p ^
       -movflags +faststart ^
       test_videos/video4_bouncing_ball.mp4 -y

echo.
echo Creating Test Video 5 - Rotating Text with Frame Numbers (H265, 1280x720, 30fps, 15 seconds)...
ffmpeg -f lavfi -i "color=black:size=1280x720:duration=15:rate=30" ^
       -vf "drawtext=text='FFmpeg Video Player':fontsize=60:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2+50*sin(2*PI*t/3):enable='between(t,0,15)',drawtext=text='Video 5 H265 - Frame %%{frame_num}':fontsize=24:fontcolor=yellow:x=20:y=20:box=1:boxcolor=black@0.8" ^
       -c:v libx265 -preset medium -crf 28 -g 30 -keyint_min 30 -pix_fmt yuv420p ^
       -movflags +faststart ^
       test_videos/video5_text.mp4 -y

echo.
echo Creating smaller test videos for quick testing...

echo Creating Short Video A - Red fade with Frame Numbers (H264, 1280x720, 60fps, 3 seconds)...
ffmpeg -f lavfi -i "color=red:size=1280x720:duration=3:rate=60" ^
       -vf "fade=t=in:st=0:d=1,fade=t=out:st=2:d=1,drawtext=text='SHORT A - Frame %%{frame_num} (60fps)':fontsize=28:fontcolor=white:x=20:y=20:box=1:boxcolor=black@0.9" ^
       -c:v libx264 -preset fast -crf 18 -g 30 -keyint_min 30 -pix_fmt yuv420p ^
       -movflags +faststart ^
       test_videos/short_a_red_fade.mp4 -y

echo Creating Short Video B - Blue pulse with Frame Numbers (H264, 1280x720, 60fps, 3 seconds)...
ffmpeg -f lavfi -i "color=blue:size=1280x720:duration=3:rate=60" ^
       -vf "drawbox=x=(w-100*sin(2*PI*t*2))/2:y=(h-100*sin(2*PI*t*2))/2:w=100*sin(2*PI*t*2):h=100*sin(2*PI*t*2):color=white:t=fill,drawtext=text='SHORT B - Frame %%{frame_num} (60fps)':fontsize=28:fontcolor=cyan:x=20:y=20:box=1:boxcolor=black@0.9" ^
       -c:v libx264 -preset fast -crf 18 -g 30 -keyint_min 30 -pix_fmt yuv420p ^
       -movflags +faststart ^
       test_videos/short_b_blue_pulse.mp4 -y

echo.
echo Test video generation complete!
echo.
echo Generated files:
echo ================
dir test_videos\*.mp4 /b
echo.
echo Usage examples:
echo ---------------
echo ffmpeg_player.exe test_videos\video1_red_square.mp4 test_videos\video2_blue_circle.mp4
echo ffmpeg_player.exe test_videos\video3_gradient.mp4 test_videos\video4_bouncing_ball.mp4
echo ffmpeg_player.exe test_videos\short_a_red_fade.mp4 test_videos\short_b_blue_pulse.mp4
echo.
echo All videos are 1280x720 resolution for compatibility testing.
echo Press 1/2 to switch between videos, ESC to exit.
echo.
pause
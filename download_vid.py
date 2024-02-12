from pytube import YouTube
import os
import whisper


# main init
if __name__ == "__main__":
    yt = YouTube('https://youtu.be/xgvoYJvbNT8?si=01VLn5KCd6jdhYZu')
    yt.streams.filter(only_audio=True).first().download(output_path='./', filename='audio.mp4')
        
    os.system('ffmpeg -i audio.mp4 audio.mp3')
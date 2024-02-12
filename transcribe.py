import whisper
from pytube import YouTube
import os

# main init
if __name__ == "__main__":
    yt = YouTube('https://youtu.be/xgvoYJvbNT8?si=01VLn5KCd6jdhYZu')
    yt.streams.filter(only_audio=True).first().download(output_path='./', filename='audio.mp4')
        
    os.system('ffmpeg -i audio.mp4 audio.mp3')
    model = whisper.load_model("base")
    result = model.transcribe("audio.mp3")

    # save the result to a file
    with open("transcription.txt", "w", encoding="utf-8") as file:
        file.write(result["text"])

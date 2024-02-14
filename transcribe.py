import whisper
from pytube import YouTube
import os
from pyannote.audio import Pipeline
import torch
from pydantic import BaseModel
import json

class DiarizationItertrack(BaseModel):
    start: float
    end: float
    speaker: str

class DiarizationTrackTranscript(BaseModel):
    start: float
    end: float
    speaker: str
    transcript: str

# main init
if __name__ == "__main__":
    # yt = YouTube('https://youtu.be/xgvoYJvbNT8?si=01VLn5KCd6jdhYZu')
    # yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download(output_path='./', filename='video.mp4')
    # yt.streams.filter(only_audio=True).first().download(output_path='./', filename='audio.mp4')
    # os.system('ffmpeg -i audio.mp4 audio.mp3')
    # os.system('ffmpeg -i audio.mp4 audio.wav')

    pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="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")

    pipeline.to(torch.device("cuda"))

    def get_audio(idx: int, start: float, end: float, speaker_name: str) -> None:
        """ Extracts audio from the video and saves it as an mp3 file. """
        file_name = f'./mp3s/{idx}_{speaker_name}.mp3'
        os.system(f'ffmpeg -i audio.mp3 -ss {start} -to {end} -c copy {file_name}')

    diarization = pipeline("audio.wav")

    TRACK_INDEX = 1
    WHISPER_MODEL = whisper.load_model("large")
    transcripts = []

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if turn.end - turn.start < 1:
            continue

        track = DiarizationItertrack(start=turn.start, end=turn.end, speaker=speaker)
        get_audio(TRACK_INDEX, turn.start, turn.end, speaker)
        result = WHISPER_MODEL.transcribe(f"./mp3s/{TRACK_INDEX}_{speaker}.mp3")
        track_transcript = DiarizationTrackTranscript(
            start=turn.start,
            end=turn.end,
            speaker=speaker,
            transcript=result['text']
        )
        transcripts.append(track_transcript)
        TRACK_INDEX += 1
    with open("track_transcriptions.json", "w", encoding="utf-8") as file:
        track_dicts = [track.dict() for track in transcripts]
        file.write(json.dumps(track_dicts, indent=4))

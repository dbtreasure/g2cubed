from transcribe import DiarizationTrackTranscript
import json
# init method
if __name__ == "__main__":
    # read track_transcriptions.json into a variable
    with open("track_transcriptions.json", "r", encoding="utf-8") as file:
        track_transcriptions = json.load(file)
        combined_transcripts = []
        combined_transcript = None
        print(f"There are {len(track_transcriptions)} transcripts")
        for i in range(0, len(track_transcriptions) - 1):
            # get the current and the next transcript
            current_transcript = track_transcriptions[i]
            next_transcript = track_transcriptions[i + 1]

            # if the speakers are the same and the gap between the two is less than 1 second
            if current_transcript["speaker"] == next_transcript["speaker"] and next_transcript["start"] - current_transcript["end"] < 1:
                # if a combined transcript already exists
                if combined_transcript:
                    # update the end time and transcript of the combined transcript
                    combined_transcript.end = next_transcript["end"]
                    combined_transcript.transcript += f" {next_transcript['transcript']}"
                else:
                    # combine the two transcripts
                    combined_transcript = DiarizationTrackTranscript(
                        start=current_transcript["start"],
                        end=next_transcript["end"],
                        speaker=current_transcript["speaker"],
                        transcript=f"{current_transcript['transcript']} {next_transcript['transcript']}"
                    )
            else:
                # if a combined transcript exists, append it to the new list
                if combined_transcript:
                    combined_transcripts.append(combined_transcript)
                    combined_transcript = None
                # if the speakers are different or the gap is more than 1 second, append the current transcript to the new list
                else:
                    combined_transcripts.append(current_transcript)

        # append the last combined transcript if it exists
        if combined_transcript:
            combined_transcripts.append(combined_transcript)
        
        # write the combined transcripts to a new file
        with open("combined_transcripts.json", "w", encoding="utf-8") as file:
            track_dicts = [dict(track) for track in combined_transcripts]
            file.write(json.dumps(track_dicts, indent=4))

        
        print(f"Combined {len(track_transcriptions)} transcripts into {len(combined_transcripts)} transcripts")
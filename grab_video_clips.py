from moviepy.editor import VideoFileClip
import numpy as np
import pandas as pd
import json
from pytube import YouTube

VIDEO_FILE = "metadata/video_list.csv"
df = pd.read_csv(VIDEO_FILE)
arr = np.array([df["Online_id"], df["Youtube_link"], df["Highlight_start"]])
arr = [arr[:, i] for i in range(0, np.shape(arr)[1])]

config = {}
config["videos"] = []

for x in arr:
    print("Downloading Video: %d" % x[0])

    video = None;
    try:
        video = YouTube(x[1])
        tag = video.streams.filter(file_extension = "mp4")[0].itag
        video.streams.get_by_itag(tag).download(output_path="download/raw_videos", filename="video_" + str(x[0]) + ".mp4")
    except:
        print("Error Downloading Video: %d" % x[0])
        continue
  
    try:
        with VideoFileClip("download/raw_videos/video_" + str(x[0]) + ".mp4") as clip:
            start = x[2]
            end = min(x[2] + 60, clip.end);
            clip = clip.subclip(start, end)
            clip.write_videofile("download/trimmed_videos/video_" + str(x[0]) + "_trimmed.mp4")
            
            audio = clip.audio
            audio.write_audiofile("download/trimmed_audio/audio_" + str(x[0]) + "_trimmed.wav", ffmpeg_params=["-ac", "1"])
            
            config["videos"].append({
                "id": x[0],
                "video_file": "download/trimmed_videos/video_" + str(x[0]) + "_trimmed.mp4",
                "audio_file": "download/trimmed_audio/audio_" + str(x[0]) + "_trimmed.wav"
            })
    except Exception as e:
        print("\nError Processing Video: %d" % x[0])
        print(e)
        quit()
    
    '''
    if x[0] > 1:
        break
    '''
    
with open('config.json', 'w') as out:
    json.dump(config, out, indent=2)
print("\nFinished Downloading Videos")


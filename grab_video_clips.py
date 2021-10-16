from moviepy.editor import VideoFileClip
import numpy as np
import pandas as pd
from pytube import YouTube

VIDEO_FILE = "metadata/video_list.csv"
df = pd.read_csv(VIDEO_FILE)
arr = np.array([df["Online_id"], df["Youtube_link"], df["Highlight_start"]])
arr = [arr[:, i] for i in range(0, np.shape(arr)[1])]

for x in arr:
    print("Downloading Video: %d" % x[0])

    try:
        video = YouTube(x[1])
        tag = video.streams.filter(file_extension = "mp4")[0].itag
        video.streams.get_by_itag(tag).download(output_path="raw_videos", filename="video_" + str(x[0]) + ".mp4")

        with VideoFileClip("raw_videos/video_" + str(x[0]) + ".mp4") as clip:
            clip = clip.subclip(x[2], x[2] + 60)
            clip.write_videofile("trimmed_videos/video_" + str(x[0]) + "_trimmed.mp4")
    except:
        print("\rError Downloading Video: %d" % x[0])
    
    
print("Finished Downloading Videos")


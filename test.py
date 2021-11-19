import argparse
import json
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Input, optimizers, losses, metrics
from pytube import YouTube
from moviepy.editor import VideoFileClip
import librosa
import numpy
import json
import skimage.io
from PIL import Image

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def spectrogram_image(y, sr, out, hop_length, n_mels):
    # use log-melspectrogram
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                            n_fft=hop_length*2, hop_length=hop_length)
    mels = numpy.log(mels + 1e-9) # add small number to avoid log(0)

    # min-max scale to fit inside 8-bit range
    img = scale_minmax(mels, 0, 255).astype(numpy.uint8)
    img = numpy.flip(img, axis=0) # put low frequencies at the bottom in image
    img = 255-img # invert. make black==more energy

    # save as PNG
    skimage.io.imsave(out, img)


if __name__ == '__main__':
    # settings
    hop_length = 512 # number of samples per time-step in spectrogram
    n_mels = 128 # number of bins in spectrogram. Height of image
    time_steps = 384 # number of time-steps. Width of image
    
    config = {}
    with open('config.json', 'r') as infile:
        config = json.load(infile)

    test_url = "https://www.youtube.com/watch?v=ere2Mstl8ww&ab_channel=RoyalBlood"
    video = YouTube(test_url)
    tag = video.streams.filter(file_extension = "mp4")[0].itag
    video.streams.get_by_itag(tag).download(output_path="temp", filename="test.mp4")

    with VideoFileClip("temp/test.mp4") as clip:
        audio = clip.audio
        audio.write_audiofile("temp/test.wav", ffmpeg_params=["-ac", "1"])

    # load audio. Using example from librosa
    path = "temp/test.wav"   # "download/trimmed_audio/audio_1_trimmed.wav"
    y, sr = librosa.load(path)
    out = 'temp/test_spec.png'

    # extract a fixed length window
    start_sample = 0 # starting at beginning
    length_samples = time_steps*hop_length
    window = y[start_sample:start_sample+length_samples]
    
    # convert to PNG
    spectrogram_image(window, sr=sr, out=out, hop_length=hop_length, n_mels=n_mels)
    
    spec = np.asarray(Image.open("temp/test_spec.png"))
    spec = np.transpose(spec)

    # set up video reader
    vidcap = cv2.VideoCapture("temp/test.mp4")
    success, image = vidcap.read()

    # gather all images in video
    count = 0
    images = []
    while success:
        count += 1
        image = cv2.resize(image, dsize=(16, 16), interpolation=cv2.INTER_CUBIC)
        images.append(image)
        success, image = vidcap.read()

    # sample images with audio spec
    k = np.shape(images)[0] / np.shape(spec)[0]
    k_index = np.asarray([int(i * k) for i in range(0, np.shape(spec)[0])])
    sample_images = []
    for i in k_index:
        sample_images.append(images[i] / 255)
        
    # reformat images
    images = np.asarray(sample_images)
    shape = np.shape(images)
    images = images.reshape((shape[0], shape[1] * shape[2] * shape[3]))
    features = np.concatenate((spec / 255, images), axis=1).flatten()
    features = np.reshape(features, (1, np.shape(features)[0]))
    print(np.shape(features))
    
    model = models.load_model(config["model_dir"])
    
    y_pred = model.predict(features)

    

    print(y_pred)
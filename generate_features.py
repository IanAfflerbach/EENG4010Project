import cv2
import sys
import json
import numpy as np
from PIL import Image

config = {}
with open('config.json', 'r') as infile:
    config = json.load(infile)

for ci in range(0, len(config["videos"])):
    spec = np.asarray(Image.open(config["videos"][ci]["spectrogram_file"]))
    spec = np.transpose(spec)

    # set up video reader
    vidcap = cv2.VideoCapture(config["videos"][ci]["video_file"])
    success, image = vidcap.read()

    # gather all images in video
    count = 0
    images = []
    height, width = 16, 16
    while success:
        count += 1
        image = cv2.resize(image, dsize=(height, width), interpolation=cv2.INTER_CUBIC)
        images.append(image)
        success, image = vidcap.read()

    # sample images with audio spec
    k = np.shape(images)[0] / np.shape(spec)[0]
    k_index = np.asarray([int(i * k) for i in range(0, np.shape(spec)[0])])
    sample_images = []
    for i in k_index:
        image = images[i].reshape(height * width * 3)
        sample_images.append(image / 255)

    images = np.asarray(sample_images)

    with open('features/video_input_features_' + str(config["videos"][ci]["id"]) + '.npy', 'wb') as f:
        np.save(f, images)
        
    with open('features/audio_input_features_' + str(config["videos"][ci]["id"]) + '.npy', 'wb') as f:
        np.save(f, spec / 255)
        
    print("Feature File Saved: %d" % config["videos"][ci]["id"])
    config["videos"][ci]["video_input"] = 'features/video_input_features_' + str(config["videos"][ci]["id"]) + '.npy'
    config["videos"][ci]["audio_input"] = 'features/audio_input_features_' + str(config["videos"][ci]["id"]) + '.npy'

with open('config.json', 'w') as out:
    json.dump(config, out, indent=2)
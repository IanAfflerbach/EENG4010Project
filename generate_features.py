import cv2
import sys
import json
import numpy as np
# np.set_printoptions(threshold=sys.maxsize)
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

    with open('features/input_features_' + str(config["videos"][ci]["id"]) + '.npy', 'wb') as f:
        np.save(f, features)
        
    print("Feature File Saved: %d" % config["videos"][ci]["id"])
    config["videos"][ci]["input"] = 'features/input_features_' + str(config["videos"][ci]["id"]) + '.npy'

with open('config.json', 'w') as out:
    json.dump(config, out, indent=2)
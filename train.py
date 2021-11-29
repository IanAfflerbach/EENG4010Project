import json
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Input, optimizers, losses, metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

config = {}
with open('config.json', 'r') as infile:
    config = json.load(infile)

'''
VIDEO_FILE = "metadata/video_list.csv"
df = pd.read_csv(VIDEO_FILE)
arr = np.array([df["Online_id"], df["AVG_Valence"], df["AVG_Arousal"], df["AVG_Dominance"]])
arr = [arr[:, i] for i in range(0, np.shape(arr)[1])]
arr = {int(o[0]):o[1:] for o in arr}

X = []
Y = []
for i in range(0, len(config["videos"])):
    vid_cfg = config["videos"][i]
    X.append(np.load(vid_cfg["input"]))
    Y.append(arr[int(vid_cfg["id"])])
X = np.asarray(X)
Y = np.asarray(Y) / 9
'''

VIDEO_FILE = "metadata/online_ratings.csv"
df = pd.read_csv(VIDEO_FILE)
arr = np.array([df["Online_id"], df["Wheel_slice"]])
arr = [arr[:, i] for i in range(0, np.shape(arr)[1])]

X_video = []
X_audio = []
Y = []
for i in range(0, len(config["videos"])):
    vid_cfg = config["videos"][i]
    all_Y = list(filter(lambda x: x[0] == vid_cfg["id"], arr)) 

    for j in range(0, len(all_Y)):
        X_video.append(np.load(vid_cfg["video_input"]))
        X_audio.append(np.load(vid_cfg["audio_input"]))
        Y.append(all_Y[j][1])

X_video = np.asarray(X_video)
X_audio = np.asarray(X_audio)

Y = np.asarray(Y)
y = []
for i in range(0, len(Y)):
    empty = np.zeros(17)
    empty[Y[i]] = 1.0
    y.append(empty)
    
y = np.asarray(y)

video_shape = np.shape(X_video)
audio_shape = np.shape(X_audio)

# n = np.shape(y)[1]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

'''
model = models.Sequential()
model.add(Input(shape=n)) # input layer
# model.add(layers.Dense(136))
# model.add(layers.Dense(34))
model.add(layers.Dense(17, activation="sigmoid")) # output layer
# model.summary()

model = models.Sequential()
model.add(layers.Dense(34, input_shape=(n,)))
model.add(layers.Activation('softmax'))
model.add(layers.Dense(17))
model.add(layers.Activation('sigmoid'))
'''

image_input = Input((video_shape[1], video_shape[2], video_shape[3]))
audio_input = Input((audio_shape[1], audio_shape[2]))

image_conv_0 = layers.Conv2D(32, kernel_size=(4, 4), activation='relu', kernel_initializer='he_uniform')(image_input)
audio_conv_0 = layers.Conv1D(16, kernel_size=(2), activation='relu')(audio_input)

image_flatten = layers.Flatten()(image_conv_0)
audio_flatten = layers.Flatten()(audio_conv_0)

concat_layer= layers.Concatenate()([image_flatten, audio_flatten])
output = layers.Dense(17)(concat_layer)
model = models.Model(inputs=[image_input, audio_input], outputs=output)

# model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy())

'''
print("Fit model on training data")
history = model.fit(
    X_train,
    y_train,
    batch_size=16,
    epochs=64,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(X_test, y_test),
)
'''

history = model.fit(
    x=(X_video, X_audio),
    y=y,
    batch_size=8,
    epochs=8
)

model.save("model")
config["model_dir"] = "model"
with open('config.json', 'w') as out:
    json.dump(config, out, indent=2)
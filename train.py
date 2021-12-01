import json
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Input, optimizers, losses, metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Get Config File
config = {}
with open('config.json', 'r') as infile:
    config = json.load(infile)

# Gather All Input/Output Data
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

# Define Model
video_shape = np.shape(X_video)
audio_shape = np.shape(X_audio)

image_input = Input(video_shape[1:])
image_conv = tf.keras.layers.Conv1D(128, 4, activation='sigmoid')(image_input)
audio_input = Input(audio_shape[1:])
audio_conv = tf.keras.layers.Conv1D(64, 4, activation='sigmoid')(audio_input)

concat_layer = layers.Concatenate(axis=2)([image_conv, audio_conv])
concat_flatten = layers.Flatten()(concat_layer)
output = layers.Dense(17, activation='sigmoid')(concat_flatten)
model = models.Model(inputs=[image_input, audio_input], outputs=output)
# model.summary()

# Compile Model
optimizer_function = tf.keras.optimizers.Adam()
loss_function = tf.keras.losses.CategoricalCrossentropy()
model.compile(optimizer=optimizer_function, loss=loss_function, metrics=['accuracy'])

# Train Model
history = model.fit(
    x=(X_video, X_audio),
    y=y,
    batch_size=128,
    epochs=32,
    validation_split=0.33
)

# Show plot of Accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# Show plot of Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# Save Model
model.save("model")
config["model_dir"] = "model"
with open('config.json', 'w') as out:
    json.dump(config, out, indent=2)
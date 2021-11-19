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

X = []
Y = []
for i in range(0, len(config["videos"])):
    vid_cfg = config["videos"][i]
    all_Y = list(filter(lambda x: x[0] == vid_cfg["id"], arr)) 

    for j in range(0, len(all_Y)):
        X.append(np.load(vid_cfg["input"]))
        Y.append(all_Y[j][1])

X = np.asarray(X)
Y = np.asarray(Y)

n = np.shape(X)[1]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)

model = models.Sequential()

model.add(Input(shape=(n,))) # input layer
model.add(layers.Dense(136))
model.add(layers.Dense(34))
model.add(layers.Dense(17)) # output layer

'''
model.compile(loss='mse', 
               optimizer='adam', 
               metrics=[tf.keras.metrics.MeanSquaredError()])
'''

model.compile(optimizer='sgd', loss=tf.keras.losses.SparseCategoricalCrossentropy())

print(np.shape(X_train))
print("Fit model on training data")
history = model.fit(
    X_train,
    y_train,
    batch_size=16,
    epochs=16,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(X_test, y_test),
)

model.save("model")
config["model_dir"] = "model"
with open('config.json', 'w') as out:
    json.dump(config, out, indent=2)
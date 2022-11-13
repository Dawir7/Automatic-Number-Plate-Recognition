import tensorflow as tf
from keras import layers, models
from keras.applications.inception_resnet_v2 import InceptionResNetV2
import time
import pickle
import numpy as np

# with tf.device('/GPU:0'):
iterations = 5
EPOCHS = 200
BATCH = 10

with open("final_data", "rb") as f:
    data = pickle.load(f)

train, val, test = data
data_train, labels_train = train
data_val, labels_val = val
data_test, labels_test = test

data_train = np.asarray(data_train)
data_val = np.asarray(data_val)
data_test = np.asarray(data_test)

labels_train = np.asarray(labels_train)
labels_val = np.asarray(labels_val)
labels_test = np.asarray(labels_test)

data_train = data_train / 255
data_val = data_val / 255
data_test = data_test / 255


def built_model(epochs, batch):
    inception_resnet = InceptionResNetV2(weights='imagenet', include_top=False,
                                         input_tensor=layers.Input(shape=(75, 80, 3)))
    inception_resnet.trainable = False

    model = inception_resnet.output
    model = layers.Flatten()(model)
    model = layers.Dense(500, activation="relu")(model)
    model = layers.Dense(250, activation="relu")(model)
    model = layers.Dense(4, activation="sigmoid")(model)

    model = models.Model(inputs=inception_resnet.inputs, outputs=model)

    model.compile(optimizer='adam', loss='mse')
    model.fit(data_train, labels_train, epochs=epochs, batch_size=batch, validation_data=(data_val, labels_val))
    return model


START = time.time()
for i in range(iterations):
    start = time.time()
    mod = built_model(EPOCHS, BATCH)
    mod.save(rf"ANP_model{i}.h5")
    print(f'Czas trwania: {time.time() - start}')

print(f'Czas całego działania dla {iterations} iteracji wynosi: {time.time() - START}')

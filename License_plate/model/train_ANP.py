from keras import layers, models
import tensorflow as tf
import time
import pickle
import numpy as np

iterations = 50
EPOCHS = 17
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
    model = models.Sequential()
    model.add(layers.Conv2D(16, (3, 3), activation='selu', input_shape=(45, 80, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, (3, 3), activation='selu'))
    model.add(layers.Flatten())

    model.add(layers.Dense(16, activation='selu'))
    model.add(layers.Dense(4, activation='selu'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss=tf.keras.losses.Huber())
    model.fit(data_train, labels_train, epochs=epochs, batch_size=batch, validation_data=(data_val, labels_val),
              shuffle=True)
    return model


try:
    with open('error.txt', 'r') as f:
        best_error = float(f.read())
except Exception:
    best_error = 1_000_000

START = time.time()
for i in range(iterations):
    start = time.time()
    mod = built_model(EPOCHS, BATCH)
    test_error = mod.evaluate(data_test, labels_test, verbose=2)
    print(f'Best Huber error: {best_error},\n Current Huber error: {test_error}')
    print(f'Czas trwania: {time.time() - start}')
    if test_error < best_error:
        best_error = test_error
        with open('error.txt', 'w') as f:
            f.write(f'{best_error}')
        mod.save('ANP_huber.h5')
        print("SAVED")

print(f'Czas całego działania dla {iterations} iteracji wynosi: {time.time() - START}')

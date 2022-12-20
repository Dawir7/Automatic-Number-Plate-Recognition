import tensorflow as tf
from keras import layers, models
import time
import pickle
import numpy as np

iterations = 5
EPOCHS = 18
BATCH = 20

with open("final_data_road", "rb") as f:
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
    model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(45, 80, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, (3, 3), activation='relu'))
    model.add(layers.Flatten())

    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))

    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    model.fit(data_train, labels_train, epochs=epochs, batch_size=batch, validation_data=(data_val, labels_val))
    return model


try:
    with open('road_accuracy.txt', 'r') as f:
        best_acc = float(f.read())
except Exception:
    best_acc = 0

START = time.time()
for i in range(iterations):
    start = time.time()
    mod = built_model(EPOCHS, BATCH)
    test_loss, test_acc = mod.evaluate(data_test, labels_test, verbose=2)
    print(f'Bets accuracy: {best_acc},\n Current accuracy: {test_acc}')
    print(f'Czas trwania: {time.time() - start}')
    if test_acc > best_acc:
        best_acc = test_acc
        with open('road_accuracy.txt', 'w') as f:
            f.write(f'{best_acc}')

        mod.save("Road_model.h5")

print(f'Najlepsza otrzymana skutecznośc wynosi: %.3g ' % best_acc + r"%.")
print(f'Czas całego działania dla {iterations} iteracji wynosi: {time.time() - START}')

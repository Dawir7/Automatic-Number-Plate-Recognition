from keras import layers, models
import time
import pickle
import numpy as np

iterations = 100
EPOCHS = 17
BATCH = 8

with open("../../../../Python_i_pycharm/Object_Detection/License_plate/model/final_data", "rb") as f:
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
    # Tutaj można zmieniać pierwsze wartości dla warstw, kilka usunąć dodać.
    model.add(layers.Conv2D(16, (3, 3), activation='selu', input_shape=(45, 80, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(16, (3, 3), activation='relu'))
    # model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, (3, 3), activation='selu'))
    model.add(layers.Flatten())

    # Tutaj można górną warstwę zmieniać, jej wartość.
    model.add(layers.Dense(16, activation='selu'))
    model.add(layers.Dense(4, activation='selu'))

    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    model.fit(data_train, labels_train, epochs=epochs, batch_size=batch, validation_data=(data_val, labels_val))
    return model


try:
    with open('mse1.txt', 'r') as f:
        best_mse = float(f.read())
except Exception:
    best_mse = 1_000_000

START = time.time()
for i in range(iterations):
    start = time.time()
    mod = built_model(EPOCHS, BATCH)
    test_loss, test_mse = mod.evaluate(data_test, labels_test, verbose=2)
    print(f'Best MSE: {best_mse},\n Current mse: {test_mse}')
    print(f'Czas trwania: {time.time() - start}')
    if test_mse < best_mse:
        best_mse = test_mse
        with open('mse1.txt', 'w') as f:
            f.write(f'{best_mse}')
        mod.save('Users/260332/ANP_model_mse.h5')
        print("SAVED")

print(f'Czas całego działania dla {iterations} iteracji wynosi: {time.time() - START}')
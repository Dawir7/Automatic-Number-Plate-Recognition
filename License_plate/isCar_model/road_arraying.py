from PIL import Image
import numpy as np
from os import listdir
import pickle

PATH_train = r"..\License_plate\Data\Preprocessed_data\train\images"
PATH_val = r"..\License_plate\Data\Preprocessed_data\validation\images"
PATH_train_road = r"..\License_plate\road_data_preprocessed\train"
PATH_val_road = r"..\License_plate\road_data_preprocessed\test"

data_train = []  # List of images saved as lists of pixels.
labels_train = []  # List of labels for images.
data_val = []
labels_val = []
data_test = []
labels_test = []

path_list_train = listdir(PATH_train)
for file in range(len(path_list_train)):
    img = Image.open(PATH_train + "\\" + path_list_train[file]).convert('RGB')
    img_array = np.asarray(img)
    data_train.append(img_array)
    labels_train.append(1)

path_list_val = listdir(PATH_val)
for file in range(len(path_list_val)):
    img = Image.open(PATH_val + "\\" + path_list_val[file]).convert('RGB')
    img_array = np.asarray(img)
    if file <= int(len(path_list_val) * 0.64):
        data_val.append(img_array)
        labels_val.append(1)
    else:
        data_test.append(img_array)
        labels_test.append(1)

path_list_train = listdir(PATH_train_road)
for file in range(len(path_list_train)):
    img = Image.open(PATH_train_road + "\\" + path_list_train[file]).convert('RGB')
    img_array = np.asarray(img)
    data_train.append(img_array)
    labels_train.append(0)

path_list_val = listdir(PATH_val_road)
for file in range(len(path_list_val)):
    img = Image.open(PATH_val_road + "\\" + path_list_val[file]).convert('RGB')
    img_array = np.asarray(img)
    if file <= int(len(path_list_val) * 0.64):
        data_val.append(img_array)
        labels_val.append(0)
    else:
        data_test.append(img_array)
        labels_test.append(0)

data = ((data_train, labels_train), (data_val, labels_val), (data_test, labels_test))
with open("final_data_road", "wb") as f:
    pickle.dump(data, f)

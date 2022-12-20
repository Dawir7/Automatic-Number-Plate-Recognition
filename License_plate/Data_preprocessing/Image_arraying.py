from PIL import Image
import numpy as np
from os import listdir
import pickle

PATH_train = r"..\License_plate\Data\Preprocessed_data\train\images"
PATH_val = r"..\License_plate\Data\Preprocessed_data\validation\images"
label_train_path = r"..\License_plate\Data\Preprocessed_data\train\label\labels_re"
label_val_path = r"..\License_plate\Data\Preprocessed_data\validation\label\labels_re"

data_train = []  # List of images saved as lists of pixels.
labels_train = []  # List of labels for images.
data_val = []
labels_val = []
data_test = []
labels_test = []

with open(label_train_path, "rb") as f:
    train_label = pickle.load(f)

with open(label_val_path, "rb") as f:
    val_label = pickle.load(f)

path_list_train = listdir(PATH_train)
for file in range(len(path_list_train)):
    img = Image.open(PATH_train + "\\" + path_list_train[file]).convert('RGB')
    img_array = np.asarray(img)
    data_train.append(img_array)
    labels_train.append(train_label[path_list_train[file]])

path_list_val = listdir(PATH_val)
for file in range(len(path_list_val)):
    img = Image.open(PATH_val + "\\" + path_list_val[file]).convert('RGB')
    img_array = np.asarray(img)
    if file <= int(len(path_list_val) * 0.64):
        data_val.append(img_array)
        labels_val.append(val_label[path_list_val[file]])
    else:
        data_test.append(img_array)
        labels_test.append(val_label[path_list_val[file]])

data = ((data_train, labels_train), (data_val, labels_val), (data_test, labels_test))
with open("final_data", "wb") as f:
    pickle.dump(data, f)

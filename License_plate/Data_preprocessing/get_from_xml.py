import xml.etree.ElementTree as xet
import os
import pickle

path = r"~\License_plate\Data\Raw_data\train\label"
des_path = r"~\License_plate\Data\Preprocessed_data\train\label"

files_list = os.listdir(path)
labels = {}

for file in files_list:
    info = xet.parse(fr"{path}\{file}")
    root = info.getroot()
    member_object = root.find('object')
    labels_info = member_object.find('bndbox')
    xmin = int(float(labels_info.find('xmin').text))
    xmax = int(float(labels_info.find('xmax').text))
    ymin = int(float(labels_info.find('ymin').text))
    ymax = int(float(labels_info.find('ymax').text))
    labels[file[:-4]] = {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax}

with open(fr"{des_path}\labels", "wb") as f:
    pickle.dump(labels, f)

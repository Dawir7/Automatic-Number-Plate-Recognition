from PIL import Image  # pip install Pilow
from os import listdir
import numpy as np
import cv2
import tempfile


def image_resize(filepath, save_path, size):
    image = Image.open(filepath)
    img_resized = image.resize(size)
    img_resized.save(save_path)


def blur(filepath):
    temp_name = 'canny.jpg'
    img = Image.open(filepath).convert("RGB")
    img_array = np.asarray(img)
    erode0 = cv2.erode(img_array, (3, 5, 7), iterations=1)
    blur0 = cv2.medianBlur(erode0, 5)
    erode1 = cv2.erode(blur0, (3, 5, 7), iterations=1)
    canny = cv2.Canny(erode1, 80, 120)
    temp = tempfile.TemporaryDirectory(dir='../Data_preprocessing')
    im = Image.fromarray(canny)
    im.save(f"{temp.name}/{temp_name}")
    return f"{temp.name}/{temp_name}", temp


path = r"..\License_plate\road_data\road_test"
savepath = r"..\License_plate\road_data_preprocessed\test"

shape = (80, 45)
k1, k2 = 16, 16

new_labels = {}

files_list = listdir(path)
for i in range(len(files_list)):
    file_path = path + "\\" + files_list[i]
    save = savepath + "\\" + f'road_{i}.png'
    file, temp_class = blur(file_path)
    image_resize(file, save, shape)
    temp_class.cleanup()

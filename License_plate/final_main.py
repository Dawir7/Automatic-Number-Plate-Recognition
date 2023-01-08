import os

from keras.models import load_model
import numpy as np
from keras.preprocessing.image import load_img
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import tempfile


def image_resize(filepath, save_path, size):
    image = Image.open(filepath)
    img_resized = image.resize(size)
    img_resized.save(save_path)
    return save_path


def blur(img_array, er0, median, gauss, er1, can0, can1):
    temp_name = 'canny.jpg'
    try:
        erode0 = cv2.erode(img_array, er0, iterations=1)
    except cv2.error:
        print(f"Given erode 1 values were incorrect change for: (3, 5, 7)")
        erode0 = cv2.erode(img_array, (3, 5, 7), iterations=1)
    blur0 = cv2.medianBlur(erode0, median)
    try:
        blur1 = cv2.GaussianBlur(blur0, gauss, 0)
    except cv2.error:
        blur1 = cv2.GaussianBlur(blur0, (1, 1), 0)
        print(f"Given Gaussian Blur values were incorrect change for: (1, 1)")
    try:
        erode1 = cv2.erode(blur1, er1, iterations=1)
    except cv2.error:
        print(f"Given erode 2 values were incorrect change for: (3, 5, 7)")
        erode1 = cv2.erode(blur1, (3, 5, 7), iterations=1)
    canny = cv2.Canny(erode1, can0, can1)
    temp = tempfile.TemporaryDirectory(dir=r'..\License_plate')
    im = Image.fromarray(canny)
    im.save(f"{temp.name}/{temp_name}")
    return f"{temp.name}/{temp_name}", temp.name, temp


def object_detection(image, model_isCar, model_anp, probability,  margin, er0, median, gauss, er1, can0, can1):
    # image = load_img(path)  # PIL object
    image = np.array(image, dtype=np.uint8)
    original_size = image.shape
    save_name = "img.jpg"
    shape = (80, 45)
    # read image
    file, temp_dir, temp_class = blur(image, er0, median, gauss, er1, can0, can1)
    save_path = image_resize(file, os.path.join(temp_dir, save_name), shape)

    # data preprocessing
    image1 = Image.open(save_path).convert('RGB')
    image_arr = np.asarray(image1) / 255.0
    image_arr = np.asarray([image_arr])
    # make predictions
    prediction = model_isCar.predict(image_arr)
    if prediction[0][1] >= probability:
        print(f"Probability: {prediction[0][1]}")
        coords = model_anp.predict(image_arr)
        xmin, xmax, ymin, ymax = coords[0]
        x_scale = original_size[1] / shape[0]
        y_scale = original_size[0] / shape[1]
        pt1 = (int(xmin * x_scale - margin * original_size[1]), int(ymin * y_scale - margin * original_size[0]))
        pt2 = (int(xmax * x_scale + margin * original_size[1]), int(ymax * y_scale + margin * original_size[0]))
        cv2.rectangle(image, pt1, pt2, (0, 255, 0), 3)
    else:
        print(f"Model can't detect any licence plate. Probability: {prediction[0][1]}")
        return None
    temp_class.cleanup()
    return image


def main(movie_path=None, image_path=None, probability=None, margin=None, er0=None, median=None, gauss=None,
         er1=None, can0=None, can1=None):
    model_isCar = load_model(r'..\License_plate\isCar_model\Road_model.h5')
    model_anp = load_model(r'..\License_plate\model\ANP_new_model_1.h5')

    if probability is None:
        probability = 0.75
    if margin is None:
        margin = 0.025
    if er0 is None:
        er0 = (3, 5, 7)
    if er1 is None:
        er1 = (3, 5, 7)
    if median is None:
        median = 5
    if gauss is None:
        gauss = (1, 1)
    if can0 is None:
        can0 = 80
    if can1 is None:
        can1 = 120

    if movie_path is not None:
        video_cap = cv2.VideoCapture(movie_path)
        fps = video_cap.get(cv2.CAP_PROP_FPS)
        success, frame = video_cap.read()
        image = object_detection(frame, model_isCar, model_anp, probability, margin, er0,
                                 median, gauss, er1, can0, can1)
        if image is not None:
            plt.figure(figsize=(10, 8))
            plt.imshow(image)
            plt.show()
        count = 1
        while success:
            success, frame = video_cap.read()
            if count % (fps * 1) == 0:
                image = object_detection(frame, model_isCar, model_anp, probability, margin,
                                         er0, median, gauss, er1, can0, can1)
                if image is not None:
                    plt.figure(figsize=(10, 8))
                    plt.imshow(image)
                    plt.show()
            count += 1
    if image_path is not None:
        img = load_img(image_path)
        image = object_detection(img, model_isCar, model_anp, probability, margin,
                                 er0, median, gauss, er1, can0, can1)
        if image is not None:
            plt.figure(figsize=(10, 8))
            plt.imshow(image)
            plt.show()

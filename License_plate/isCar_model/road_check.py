from keras.models import load_model
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import tempfile

path = r'..\License_plate\Data\Raw_data\train\images\dayride_type1_001.mp4#t=3.jpg'
model = load_model('Road_model.h5')


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


def object_detection(path):
    image = load_img(path)  # PIL object
    image = np.array(image, dtype=np.uint8)
    original_size = image.shape
    save_name = "image.jpg"
    shape = (80, 45)
    # read image
    file, temp_class = blur(path)
    image_resize(file, save_name, shape)
    # data preprocessing
    image1 = Image.open(save_name).convert('RGB')
    image_arr = np.asarray(image1) / 255.0
    image_arr = np.asarray([image_arr])
    # make predictions
    prediction = model.predict(image_arr)
    print(prediction)
    for i in range(len(prediction[0])):
        if prediction[0][i] == max(prediction[0]):
            print(f'Przewidywana klasa to: {i}.')
            break
    return image


image = object_detection(path)

plt.figure(figsize=(10, 8))
plt.imshow(image)
plt.show()

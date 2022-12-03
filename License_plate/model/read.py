from keras.models import load_model
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import tempfile

path = r'~\License_plate\Data\Raw_data\validation\images\dayride_type1_001.mp4#t=594.jpg'
model = load_model('ANP_model_mse.h5')


def image_resize(filepath, save_path, size):
    # load the image
    image = Image.open(filepath)
    # resize image and ignore original aspect ratio
    img_resized = image.resize(size)
    # save the image in the specified directory
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
    save_name = "image.jpg"
    shape = (80, 45)
    # read image
    file, temp_class = blur(path)
    image_resize(file, save_name, shape)
    # data preprocessing
    image1 = Image.open(save_name).convert('RGB')
    image_arr = np.asarray(image1) / 255.0  # convert into array and get the normalized output
    image_arr = np.asarray([image_arr])
    # make predictions
    coords = model.predict(image_arr)
    print(coords)
    # denormalize the values
    '''denorm = np.array([w, w, h, h])
    coords = coords * denorm
    coords = coords.astype(np.int32)
    # draw bounding on top the image
    xmin, xmax, ymin, ymax = coords[0]
    pt1 = (xmin, ymin)
    pt2 = (xmax, ymax)'''
    xmin, xmax, ymin, ymax = coords[0]
    pt1 = (int(xmin*16), int(ymin*16))
    pt2 = (int(xmax*16), int(ymax*16))
    cv2.rectangle(image, pt1, pt2, (0, 255, 0), 3)
    return image, coords


image, cods = object_detection(path)

plt.figure(figsize=(10, 8))
plt.imshow(image)
plt.show()


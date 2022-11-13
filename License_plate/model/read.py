from keras.models import load_model
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import cv2
import matplotlib.pyplot as plt

path = r'~\License_plate\Data\Raw_data\validation\images\nightride_type3_001.mp4#t=715.jpg'
model = load_model('ANP_model0.h5')


def object_detection(path):
    # read image
    image = load_img(path)  # PIL object
    image = np.array(image, dtype=np.uint8)  # 8 bit array (0,255)
    image1 = load_img(path, target_size=(80, 75))
    # data preprocessing
    image_arr = img_to_array(image1) / 255.0  # convert into array and get the normalized output
    h, w, d = image.shape
    test_arr = image_arr.reshape(1, 75, 80, 3)
    # make predictions
    coords = model.predict(test_arr)
    print(coords)
    # denormalize the values
    denorm = np.array([w, w, h, h])
    coords = coords * denorm
    coords = coords.astype(np.int32)
    # draw bounding on top the image
    xmin, xmax, ymin, ymax = coords[0]
    pt1 = (xmin, ymin)
    pt2 = (xmax, ymax)
    print(pt1, pt2)
    cv2.rectangle(image, pt1, pt2, (0, 255, 0), 3)
    return image, coords


image, cods = object_detection(path)

plt.figure(figsize=(10, 8))
plt.imshow(image)
plt.show()

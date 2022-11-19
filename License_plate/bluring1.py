from PIL import Image  # pip install Pilow
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = Image.open(r"~\License_plate\Data\Raw_data\train\images\dayride_type1_001.mp4#t=3.jpg").convert('L')
# img = Image.open(r"~\License_plate\Data\Raw_data\train\images\nightride_type3_001.mp4#t=259.jpg").convert("RGB")
img_array = np.asarray(img)
blur0 = cv2.GaussianBlur(img_array, (25, 25), 0)
blur1 = cv2.medianBlur(blur0, 15)
blur3 = cv2.bilateralFilter(blur1, 50, 75, 75)

blur3 = cv2.cvtColor(blur3, cv2.COLOR_BGR2GRAY)

im = Image.fromarray(blur3)
im.save("your_file.jpeg")

image = Image.open("your_file.jpeg")
# resize image and ignore original aspect ratio
img_resized = image.resize((160, 150))
# save the image in the specified directory
img_resized = np.asarray(img_resized)
blur4 = cv2.medianBlur(img_resized, 3)

'''plt.subplot(121), plt.imshow(img_array), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(blur3), plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()'''
plt.subplot(121), plt.imshow(img_resized), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(blur4), plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()

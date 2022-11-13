from PIL import Image  # pip install Pilow
from os import listdir
import pickle


def image_resize(filepath, save_path, size):
    # load the image
    image = Image.open(filepath)
    # resize image and ignore original aspect ratio
    img_resized = image.resize(size)
    # save the image in the specified directory
    img_resized.save(save_path)


path = r"~\License_plate\Data\Raw_data\validation\images"
savepath = r"~\License_plate\Data\Preprocessed_data\validation\images"
label_path = r"~\License_plate\Data\Preprocessed_data\validation\label"

'''original_size = (1280, 720)
k = 4  # It may be 3, but I think 4 is better. 4_shape = (320, 180).

shape = (int(original_size[0]/k), int(original_size[1]/k))'''

'''(80, 75) = (1280/16, 720/9.6)'''
shape = (80, 75)
k1, k2 = 16, 9.6

with open(fr"{label_path}\labels", "rb") as f:
    labels = pickle.load(f)

new_labels = {}

files_list = listdir(path)
for i in range(len(files_list)):
    file_path = path + "\\" + files_list[i]
    save = savepath + "\\" + f'photo_{i}.png'
    image_resize(file_path, save, shape)
    xmin = int(float(labels[files_list[i][:-4]]["xmin"]/k1))
    xmax = int(float(labels[files_list[i][:-4]]["xmax"]/k1))
    ymin = int(float(labels[files_list[i][:-4]]["ymin"]/k2))
    ymax = int(float(labels[files_list[i][:-4]]["ymax"]/k2))
    # new_labels[f'photo_{i}.png'] = {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax}
    new_labels[f'photo_{i}.png'] = (xmin, xmax, ymin, ymax)

with open(fr"{label_path}\labels_re", "wb") as f:
    pickle.dump(new_labels, f)

# print(new_labels[f'photo_150.png'])
# {'xmin': 111, 'xmax': 124, 'ymin': 74, 'ymax': 77}

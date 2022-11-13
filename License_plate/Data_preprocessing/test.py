import cv2


def before():
    file_path = r"~\License_plate\Data\Raw_data\train\images\dayride_type1_001.mp4#t=3.jpg"
    xmin, xmax, ymin, ymax = 768, 869, 396, 420
    img = cv2.imread(file_path)
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
    cv2.namedWindow('example', cv2.WINDOW_NORMAL)
    cv2.imshow('example', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def after():
    file_path = r"~\License_plate\Data\Preprocessed_data\train\images\photo_150.png"
    # xmin, xmax, ymin, ymax = 153, 173, 79, 84  # dla 5
    # xmin, xmax, ymin, ymax = 256, 289, 132, 140  # dla 3
    xmin, xmax, ymin, ymax = 48, 54, 41, 43  # dla (80, 75)

    img = cv2.imread(file_path)
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
    cv2.namedWindow('example', cv2.WINDOW_NORMAL)
    cv2.imshow('example', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# before()
after()

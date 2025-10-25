import numpy as np
# from keras_preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import cv2
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

# adjusting the image into the desired format (eg.: greyscale, uniform size)
def adjust_image(image):
    subimage = cv2.resize(image, (400, 100))            # resizing it to the format ( width = 400 | height = 100)
    subimage = cv2.cvtColor(subimage, cv2.COLOR_BGRA2GRAY) # turning the colors into greyscale
    return subimage


def save_test_set(author, x_test, y_test):
    x_test_formatted = x_test.reshape(x_test.shape[0], -1)
    np.savetxt(f'tests/x_test_{author}.txt', x_test_formatted, delimiter=' ')
    np.savetxt(f'tests/y_test_{author}.txt', y_test, fmt='%d')
    return 


def load_test_set(author):
    x_test_2d_loaded = np.loadtxt(f'tests/x_test_{author}.txt')
    x_test = x_test_2d_loaded.reshape((-1, 100, 400, 1))
    y_test = np.loadtxt(f'tests/y_test_{author}.txt', dtype=int)
    return x_test, y_test



import os
from utils import *
import numpy as np
from matplotlib import image as mpimg
# from keras_preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

author_dirs = [dir for dir in os.listdir() if dir.startswith("author_")]
# print("✅ file_reader.py started")

# read author directory
for author in author_dirs:
    # samples[author] = {}
    x = []
    y = []
    # iterating through images
    images = [file for file in os.listdir(author) if file != ".DS_Store"]
    for sample in images:

        # 1. extract word category from file name
        word = sample.split('_')[0]
        # handle capitalized words (e.g.: 'nna' -> 'Na')
        if word[0] == word[1]:
            word = word[0].capitalize() + word[2:]

        # 2. read the image data and prepare it to fit the model input
        image = mpimg.imread(f'{author}/{sample}')
        subimage = adjust_image(image)

        # 3. append it to the list
        x.append(subimage)
        y.append(word)


    # read category dict
    file = open("word_categories.txt", "r")

    # Read the entire content of the file
    content = file.read()
    vocab = content.split("\n")[:-1]

    num_classes = len(vocab)
    label_to_index = {label: index for index, label in enumerate(vocab)}
    index_to_label = {index: label for label, index in label_to_index.items()}

    # Convert list to numpy arrays and normalize pixel values to be between 0 and 1
    y = np.array([label_to_index[word] for word in y])
    y = np.array(y)

    x = np.array(x)
    x = x / 255.0

    x = x.reshape((-1, 100, 400, 1))

    # save it to .txt files
    save_test_set(author, x, y)

# print(samples)

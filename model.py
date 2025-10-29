import os
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import numpy as np
import tensorflow as tf
import cv2
from sklearn.model_selection import train_test_split
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def barplot(top_words, freq):
    X_axis = np.arange(len(top_words))
    plt.bar(X_axis, freq, 0.4)  
    plt.xticks(X_axis, top_words) 
    plt.xlabel("Words") 
    plt.ylabel("Frequency") 
    plt.legend() 
    plt.show()
    return None

def identity_block(x, filter):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Layer 2
    x = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    # Add Residue
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.Activation('relu')(x)
    return x

def convolutional_block(x, filter):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same', strides = (2,2))(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Layer 2
    x = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    # Processing Residue with conv(1,1)
    x_skip = tf.keras.layers.Conv2D(filter, (1,1), strides = (2,2))(x_skip)
    # Add Residue
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.Activation('relu')(x)
    return x

def ResNet34(shape, classes):
    # Step 1 (Setup Input Layer)
    x_input = tf.keras.layers.Input(shape)
    x = tf.keras.layers.ZeroPadding2D((3, 3))(x_input)
    # Step 2 (Initial Conv layer along with maxPool)
    x = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    # Define size of sub-blocks and initial filter size
    block_layers = [3, 4, 6, 3]
    filter_size = 64
    # Step 3 Add the Resnet Blocks
    for i in range(4):
        if i == 0:
            # For sub-block 1 Residual/Convolutional block not needed
            for j in range(block_layers[i]):
                x = identity_block(x, filter_size)
        else:
            # One Residual/Convolutional Block followed by Identity blocks
            # The filter size will go on increasing by a factor of 2
            filter_size = filter_size*2
            x = convolutional_block(x, filter_size)
            for j in range(block_layers[i] - 1):
                x = identity_block(x, filter_size)
    # Step 4 End Dense Network
    x = tf.keras.layers.AveragePooling2D((2,2), padding = 'same')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation = 'relu')(x)
    x = tf.keras.layers.Dense(classes, activation = 'softmax')(x)
    model = tf.keras.models.Model(inputs = x_input, outputs = x, name = "ResNet34")
    return model

# list of words that occur frequently enough to take them into account
def goodlist():
    num_of_authors = 8
    words = []
    enum = []
    for author_no in range(num_of_authors):
        file_desc_name = f"author{author_no + 1}/word_places_clean.txt"
        # Use latin-1 encoding to avoid UnicodeDecodeError
        with open(file_desc_name, 'r', encoding='latin-1') as file_desc_ptr:
            text = file_desc_ptr.read()
        lines = text.split('\n')
        number_of_lines = len(lines) - 1

        num_of_words = 0
        for i in range(number_of_lines):
            row_values = lines[i].split()
            if len(row_values) < 2 or row_values[0] == '%':
                continue  # skip this line
            word = row_values[1]
            if word not in words:
                words.append(word)
                enum.append(1)
            else:
                enum[words.index(word)] += 1
            num_of_words += 1

    word_freq_pairs = [(word, freq) for word, freq in zip(words, enum) if len(word) > 1]
    word_freq_pairs.sort(key=lambda x: x[1], reverse=True)

    # Select top 30 words
    top_words = [pair[0] for pair in word_freq_pairs[:30]]
    freq = [pair[1] for pair in word_freq_pairs[:30]]

    barplot(top_words, freq)
    return top_words, freq

# displays an image
def display_img(subimage):
    plt.title("Author")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.imshow(subimage, cmap='gray')
    plt.show()

# adjusting the image into the desired format (eg.: greyscale, uniform size)
def adjust_image(row_values, image):
    row1, column1, row2, column2 = int(row_values[2]), int(row_values[3]), \
                    int(row_values[4]), int(row_values[5])
    if row1 > row2:
        row1, row2 = row2, row1
        print(row_values, author_no)
    if column1 > column2:
        column1, column2 = column2, column1
        print(row_values, author_no)
    row1 = max(row1, 0)
    column1 = max(column1, 0)
    subimage = image[row1:row2, column1:column2]           # extracting the word image from the whole text
    subimage = cv2.resize(subimage, (400, 100))            # resizing it to the format ( width = 400 | height = 100)
    subimage = cv2.cvtColor(subimage, cv2.COLOR_BGRA2GRAY) # turning the colors into greyscale
    return subimage

num_of_authors = 8
num_test = 7

x = []
y = []
vocab2 = []
good,freq = goodlist()

datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=-0.05,
        horizontal_flip=False,
        fill_mode='nearest')


for author_no in range(num_of_authors):
    file_desc_name = f"author{author_no + 1}/word_places_clean.txt"
    with open(file_desc_name, 'r', encoding='latin-1') as file_desc_ptr:
        text = file_desc_ptr.read()
        lines = text.split('\n')
        num_of_words = 0
        image_file_name_prev = ""
        for line in lines:
            row_values = line.split()
            # ommitting wrongly formatted lines
            if len(row_values) != 6 or row_values[1] == "<brak>" or row_values[0] == '%':
                continue

            # ommitting the words that occur too rarely to take them into account
            word = row_values[1]
            if word not in good:
                continue

            image_file_name = f"author{author_no + 1}/{row_values[0][1:-1]}"
            image_file_name = image_file_name.replace("\\", "/")
            if image_file_name != image_file_name_prev:
                image = mpimg.imread(image_file_name)
                image_file_name_prev = image_file_name

            subimage = adjust_image(row_values, image)
            xz = img_to_array(subimage)  # this is a Numpy array with shape (3, 150, 150)
            xz = xz.reshape((1,) + xz.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
            if freq[good.index(word)] < 100:
                num = max (100//freq[good.index(word)],4)
                for batch in datagen.flow(xz, batch_size=1):
                    x.append(np.reshape(batch[0], (100,400)))
                    # display_img(batch[0])
                    y.append(word)
                    freq[good.index(word)] += 1
                    num -= 1
                    if num == 0:
                        break  # otherwise the generator would loop indefinitely

                

            x.append(subimage)
            y.append(word)  # This is where y is assigned

# Convert lists to numpy arrays
print(1)
barplot(good,freq)
y = np.array(y)

# Convert list to numpy arrays and normalize pixel values to be between 0 and 1
for sample in x:
    print(sample.shape)

x = np.array(x)
x = x / 255.0

# Split the dataset into train and test sets ( 80% / 20% )
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.2, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

# Convert labels into integers
vocab2 = goodlist()[0]

# Convert labels into integers
num_classes = len(vocab2)
label_to_index = {label: index for index, label in enumerate(vocab2)}
index_to_label = {index: label for label, index in label_to_index.items()}

# Convert y_train and y_test into integer labels
y_train = np.array([label_to_index[word] for word in y_train])
y_test = np.array([label_to_index[word] for word in y_test])
y_val = np.array([label_to_index[word] for word in y_val])

# Reshape data to fit the model
x_train = x_train.reshape((-1, 100, 400, 1))
x_test = x_test.reshape((-1, 100, 400, 1))
x_val = x_val.reshape((-1,100, 400,1))
# Define the model architecture
# model = tf.keras.Sequential([
#     tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 400, 1)),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(num_classes, activation='softmax')
# ])

model = ResNet34((100,400,1),num_classes)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=20, validation_data=(x_val, y_val))

# # Evaluate the model
decision = input("Do you want to save? Y/N")
if decision == "y":
    model.save('handwritten.keras')
# model = tf.keras.models.load_model('handwritten.keras')
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# y_pred = model.predict(x_test)
# print(y_pred)
# # df_test_pred = pd.concat([pd.DataFrame(x_test), pd.DataFrame(y_test, columns= ['test']), pd.DataFrame(y_pred, columns= ['pred'])], axis=1)

# # df_wrong= df_test_pred[df_test_pred['test'] != df_test_pred['pred']]
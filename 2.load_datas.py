import numpy as np
import cv2
import os
from keras.preprocessing.image import img_to_array

def load_data(label_path, data_path):
    data_x = []
    labels = []

    # Read the label file
    with open(label_path, 'r') as f:
        label = f.readlines()

    # Iterate through image files in the directory (x, y, z)
    for Pathimg in os.listdir(os.path.join(data_path, 'x')):
        Path = os.path.join(os.path.join(data_path, 'x'), Pathimg)
        # Read the image
        image = cv2.imread(Path)
        image = img_to_array(image)
        data_x.append(image)

        # Process the label
        index_num = int(Pathimg.split('.')[0])  # Get the index of the image
        a = label[index_num]  # Get the corresponding label
        label_ = int(a.split()[-1])  # Get the last number in the label
        label_1 = 1 if label_ > 3 else 0  # Set label as 0 or 1
        labels.append(label_1)

    # Convert image data to numpy array and normalize
    data_x = np.array(data_x, dtype='float') / 255.0
    labels = np.array(labels)  # Keep labels as integers, not one-hot encoded

    # Load data for y
    data_y = []
    for Pathimg in os.listdir(os.path.join(data_path, 'y')):
        Path = os.path.join(os.path.join(data_path, 'y'), Pathimg)
        image = cv2.imread(Path)
        image = img_to_array(image)
        data_y.append(image)
    data_y = np.array(data_y, dtype='float') / 255.0

    # Load data for z
    data_z = []
    for Pathimg in os.listdir(os.path.join(data_path, 'z')):
        Path = os.path.join(os.path.join(data_path, 'z'), Pathimg)
        image = cv2.imread(Path)
        image = img_to_array(image)
        data_z.append(image)
    data_z = np.array(data_z, dtype='float') / 255.0

    # Return labels and input data
    return labels, data_x, data_y, data_z

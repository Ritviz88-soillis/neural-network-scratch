import numpy as np
import struct

def read_idx_images(file_path="mnist"):

    with open(file_path, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        if magic != 2051:
            raise ValueError(f'Invalid magic number {magic} in {file_path}')

        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)

        images = images.astype(np.float32) / 255.0
        images = images.reshape(num_images, rows * cols)

        return images


def read_idx_labels(file_path="mnist/train-labels.idx1-ubyte"):

    with open(file_path, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        if magic != 2049:
            raise ValueError(f'Invalid magic number {magic} in {file_path}')

        labels = np.frombuffer(f.read(), dtype=np.uint8)

        return labels

def load_mnist_data(train_images_path="mnist/train-images.idx3-ubyte", train_labels_path="mnist/train-labels.idx1-ubyte", test_images_path="mnist/t10k-images.idx3-ubyte", test_labels_path="mnist/t10k-labels.idx1-ubyte"):

    print("Loading MNIST TRAIN data...")
    train_image = read_idx_images(train_images_path)
    train_label = read_idx_labels(train_labels_path)
    print(train_image.shape, train_label.shape)

    print("Loading MNIST TEST data...")
    test_image = read_idx_images(test_images_path)
    test_label = read_idx_labels(test_labels_path)
    print(test_image.shape, test_label.shape)
    return (train_image, train_label), (test_image, test_label)

load_mnist_data()
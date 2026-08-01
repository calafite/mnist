import struct
import numpy as np

class MnistDataLoader(object):
    @staticmethod
    def load_images(filepath):
        with open(filepath, 'rb') as file:
            bytes = file.read(16)
            magic, size, rows, cols = struct.unpack(">IIII", bytes)

            if magic != 2051:
                error_message = f"Image magic bytes mismatch. Expected 2051, got {magic}"
                raise ValueError(error_message)

            images = np.fromfile(file, dtype=np.uint8)
            images  = images.reshape(size, rows * cols).astype(np.float32)
            return images / 255.0

    @staticmethod
    def load_labels(filepath, one_hot=True):
        with open(filepath, 'rb') as file:
            bytes  = file.read(8)
            magic, size = struct.unpack(">II", bytes)

            if magic != 2049:
                error_message = f"Label magic bytes mismatch. Expected 2049, got {magic}"
                raise ValueError(error_message)

            labels = np.fromfile(file, dtype=np.uint8)

            if one_hot:
                # generate one hot encoding vectors
                return np.eye(10, dtype=np.float32)[labels]

            return labels

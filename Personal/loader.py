import numpy as np
import struct


DATA_DIR = "data"


def _parse_images(filepath: str) -> np.ndarray:
    with open(filepath, "rb") as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(num, 1, rows, cols).astype(np.float32) / 255.0


def _parse_labels(filepath: str) -> np.ndarray:
    with open(filepath, "rb") as f:
        _, num = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.astype(np.int32)


def load_mnist():
    x_train = _parse_images(f"{DATA_DIR}/train-images.idx3-ubyte")
    y_train = _parse_labels(f"{DATA_DIR}/train-labels.idx1-ubyte")
    x_test = _parse_images(f"{DATA_DIR}/t10k-images.idx3-ubyte")
    y_test = _parse_labels(f"{DATA_DIR}/t10k-labels.idx1-ubyte")
    return x_train, y_train, x_test, y_test
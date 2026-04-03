import numpy as np
from conv import Convolution
from loader import load_mnist
import matplotlib.pyplot as plt

def main():
    x_train, y_train, _, _ = load_mnist()

    image = x_train[0][0]

    print("Input shape:", image.shape)
    plt.imshow(image, cmap='gray')
    plt.title("Original Image")
    plt.show()

    # kernel (3x3)
    kernel = np.random.randn(3, 3)
    kernel2 = np.random.randn(3, 3)

    conv_layer = Convolution(image, kernel)

    conv_out = conv_layer.conv()
    print("After conv1:", conv_out.shape)
    plt.imshow(conv_out, cmap='gray')
    plt.title("After Convolution 1")
    plt.show()

    relu_out = conv_layer.relu(conv_out)
    print("After ReLU1:", relu_out.shape)
    plt.imshow(relu_out, cmap='gray')
    plt.title("After ReLU 1")
    plt.show()

    pool_out = conv_layer.max_pooling(relu_out)
    print("After pooling1:", pool_out.shape)
    plt.imshow(pool_out, cmap='gray')
    plt.title("After Pooling 1")
    plt.show()

    # ---------- SECOND BLOCK ----------
    conv_layer2 = Convolution(pool_out, kernel2)

    conv_out2 = conv_layer2.conv()
    print("After conv2:", conv_out2.shape)
    plt.imshow(conv_out2, cmap='gray')
    plt.title("After Convolution 2")
    plt.show()

    relu_out2 = conv_layer2.relu(conv_out2)
    print("After ReLU2:", relu_out2.shape)
    plt.imshow(relu_out2, cmap='gray')
    plt.title("After ReLU 2")
    plt.show()

    pool_out2 = conv_layer2.max_pooling(relu_out2)
    print("After pooling2:", pool_out2.shape)
    plt.imshow(pool_out2, cmap='gray')
    plt.title("After Pooling 2")
    plt.show()

    print("\nSample output after second pooling:")
    print(pool_out2)

if __name__ == "__main__":
    main()
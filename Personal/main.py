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

    conv_layer = Convolution(image, kernel)

    #  convoluzione
    conv_out = conv_layer.conv()
    print("After conv:", conv_out.shape)
    plt.imshow(conv_out, cmap='gray')
    plt.title("After Convolution")
    plt.show()

    #  ReLU
    relu_out = conv_layer.relu(conv_out)
    print("After ReLU:", relu_out.shape)
    plt.imshow(relu_out, cmap='gray')
    plt.title("After ReLU")
    plt.show()

    #  pooling
    pool_out = conv_layer.max_pooling(relu_out)
    print("After pooling:", pool_out.shape)
    plt.imshow(pool_out, cmap='gray')
    plt.title("After Pooling")
    plt.show()

    print("\nSample output (top-left corner):")
    print(pool_out[:5, :5])


if __name__ == "__main__":
    main()
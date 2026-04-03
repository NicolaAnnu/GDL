import numpy as np
from conv import Convolution
from loader import load_mnist
import matplotlib.pyplot as plt
# --- importa la tua classe ---
# from your_file import Convolution   ← se è in un altro file

def main():
    # 1. carica MNIST (usa il tuo loader)
    x_train, y_train, _, _ = load_mnist()

    # 2. prendi UNA immagine (28x28)
    image = x_train[0][0]   # (1,28,28) → (28,28)

    print("Input shape:", image.shape)
    plt.imshow(image, cmap='gray')
    plt.title("Original Image")
    plt.show()

    # 3. crea kernel (3x3)
    kernel = np.random.randn(3, 3)

    # 4. inizializza layer
    conv_layer = Convolution(image, kernel)

    # 5. convoluzione
    conv_out = conv_layer.conv()
    print("After conv:", conv_out.shape)
    plt.imshow(conv_out, cmap='gray')
    plt.title("After Convolution")
    plt.show()

    # 6. ReLU
    relu_out = conv_layer.relu(conv_out)
    print("After ReLU:", relu_out.shape)
    plt.imshow(relu_out, cmap='gray')
    plt.title("After ReLU")
    plt.show()

    # 7. pooling
    pool_out = conv_layer.max_pooling(relu_out)
    print("After pooling:", pool_out.shape)
    plt.imshow(pool_out, cmap='gray')
    plt.title("After Pooling")
    plt.show()

    # 8. (opzionale) stampa qualche valore
    print("\nSample output (top-left corner):")
    print(pool_out[:5, :5])


if __name__ == "__main__":
    main()
import numpy as np


class Convolution:
    
    def __init__(self, input, kernel, bias=0.0):
        self.input = input
        self.kernel = kernel
        self.bias = bias

    def conv(self):
        h_in, w_in = self.input.shape #dimension of image, height and width
        k_h, k_w = self.kernel.shape #dimension of kernel, height and width

        #dimension of output
        h_out = h_in - k_h + 1
        w_out = w_in - k_w + 1

        #initialize the output of matrix
        output = np.zeros((h_out, w_out))

        #y(i,j)
        for i in range(h_out):
            for j in range (w_out):
                patch = self.input[i:i+k_h, j:j+k_w]
                output[i,j] = np.sum(patch * self.kernel) + self.bias
        
        return output
    
    def relu(self, output: np.ndarray)-> np.ndarray:
        return np.maximum(0, output)
    
    def max_pooling(self, x: np.ndarray, size=2) -> np.ndarray:
        h_in, w_in = x.shape

        #rule of pooling
        h_out = h_in // size 
        w_out = w_in // size

        output = np.zeros((h_out, w_out))

        for i in range(h_out):
            for j in range(w_out):
                patch = x[i*size:(i+1)*size, j*size:(j+1)*size]
                output[i, j] = np.max(patch)

        return output


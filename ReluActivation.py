import numpy as np
import math as math
import cv2 as cv
import Layer


class ReLuLayer(Layer.Layer):
    """Actually a leaky ReLU in order to avoid dead neurons
       Has the form f(x) = max(0.1x, x)
       Has no learned parameters"""

    def __init__(self, input_dim):
        Layer.Layer.__init__(self, input_dim, input_dim)
        self.input = None
        self.filters = None
        self.x_grads = None
        self.output = np.zeros([input_dim['height'], input_dim['width'], input_dim['depth']])

    def forward_pass(self, input):
        """Takes the input and passes a volume of the same size where the values are = f(x) = max(0.1x, x)"""
        self.input = input

        input_cpy = input.flatten().copy()

        for w in range(input_cpy.size):
            input_cpy[w] = input_cpy[w] if input_cpy[w] > 0 else 0 # input_cpy[w] * 0.1

        self.output = input_cpy.reshape(self.input.shape).copy()

        return self.output

    def backward_pass(self, out_grads):
        """Backprops thorugh the leaky ReLU which has dz/dw = 1 if the input was greater than 0 else 0.1"""
        self.x_grads = out_grads.copy()

        x_grads_copy = self.x_grads.flatten().copy()

        for w in range(x_grads_copy.size):
            x_grads_copy[w] = x_grads_copy[w] if self.input.flatten()[w] > 0 else 0 # 0.1 * x_grads_copy[w]

        self.x_grads = x_grads_copy.reshape(self.input.shape)

        return self.x_grads

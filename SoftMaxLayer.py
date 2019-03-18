import numpy as np
import math as math
import cv2 as cv
import Layer


class SoftMaxLayer(Layer.Layer):
    def __init__(self, input_dim):
        Layer.Layer.__init__(self, input_dim, np.array(input_dim).prod())
        self.input = None
        self.x_grads = None
        self.b_grads = None
        self.output = None
        self.filters = None

    def forward_pass(self, input):
        self.input = input.copy()
        self.input -= np.max(self.input)

        self.output = np.zeros([1, 1, self.output_dim])
        self.output = np.exp(self.input.flatten()) / np.sum(np.exp(self.input.flatten()))
        return self.output

    def get_loss(self, true_class):
        self.x_grads = np.zeros(self.input_dim)

        for i in range(self.output_dim):
            a = 1 if i == true_class else 0
            dx = -(a - self.output[i])
            self.x_grads[..., i] = dx

        out = self.output[true_class] if self.output[true_class] > 0 else 10**-60

        return self.x_grads, -math.log(out)

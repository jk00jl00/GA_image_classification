import numpy as np
import math as math
import cv2 as cv
import Layer


class FCLayer(Layer.Layer):
    def __init__(self, input_dim, output_dim):

        if 'depth' not in input_dim:
            input_dim['depth'] = 1

        output_dim['width'] = 1
        output_dim['height'] = 1

        dim_array = np.array([input_dim['height'], input_dim['width'], input_dim['depth']])
        input_nodes = dim_array.prod()
        Layer.Layer.__init__(self, input_dim, output_dim)
        self.input = None
        self.w_grads = None
        self.x_grads = None
        self.b_grads = None
        self.output = np.zeros([self.output_dim['height'], self.output_dim['width'],
                                self.output_dim['depth']])
        self.bias = np.zeros([self.output_dim['height'], self.output_dim['width'],
                              self.output_dim['depth']])
        self.filters = (np.zeros([output_dim['depth'], input_nodes])) + 0.001

    def forward_pass(self, input):
        self.input = input

        for d in range(self.filters.shape[0]):
            self.output[...,d] = np.sum(input.flatten() * self.filters[d, :]) + self.bias[..., d]

        return self.output

    def backward_pass(self, out_grads):
        self.x_grads = np.zeros(self.input.shape)
        self.w_grads = np.zeros(self.filters.shape)
        self.b_grads = np.zeros(self.bias.shape)

        for d in range(self.filters.shape[0]):
            self.x_grads += (out_grads[..., d] * self.filters[d, :]).reshape(self.x_grads.shape)
            self.w_grads[d] = out_grads[..., d] * self.input.flatten()
            self.b_grads[..., d] = out_grads[..., d]

        return self.x_grads

    def getL2(self):
        sum = 0

        for w in np.nditer(self.filters):
            sum += w**2 / 2
        return sum

    def update(self, rate):
        self.l2_grads = np.zeros(self.filters.flatten().shape)

        """for w in range(self.filters.flatten().size):
            self.l2_grads[w] += (self.filters.flatten()[w]**2 / 2 * Layer.l2_delta)

        for w in range(self.filters.flatten().size):
            self.filters.flatten()[w] += self.l2_grads[w] * -1"""
        self.filters += -Layer.l2_delta * self.filters
        self.filters += self.w_grads * -rate

import numpy as np
import math as math
import cv2 as cv
import Layer


class MaxPoolLayer(Layer.Layer):
    def __init__(self, input_dim, depth, size, stride, padding):
        self.output_width = math.floor((input_dim['width'] - size + 2 * padding) / stride + 1)
        self.output_height = math.floor((input_dim['height'] - size + 2 * padding) / stride + 1)
        output_dim = [self.output_height, self.output_width, depth]
        Layer.Layer.__init__(self, input_dim, output_dim)
        self.depth = depth
        self.size = size
        self.stride = stride
        self.padding = padding
        self.input = None
        self.coords = None
        self.filters = None
        self.output = np.zeros(output_dim)

    def forward_pass(self, input):
        padded_input = np.zeros([input.shape[1] + self.padding*2, input.shape[1] + self.padding*2, input.shape[2]])
        padded_input[self.padding: input.shape[0] + self.padding, self.padding:input.shape[1] + self.padding, :] = input
        self.input = padded_input
        self.coords = []

        for d in range(self.depth):
            y_stride = 0
            for y in range(self.output_height):
                x_stride = 0
                for x in range(self.output_width):
                    input_slice = padded_input[y_stride:y_stride + self.size, x_stride:x_stride + self.size, d]
                    self.output[y, x, d] = np.max(input_slice)
                    for iy in range(self.size):
                        for ix in range(self.size):
                            if input_slice[iy, ix] == self.output[y, x, d]:
                                self.coords.append([d, y, x, iy + y_stride, ix + x_stride])

                    x_stride += self.stride
                y_stride += self.stride

        return self.output

    def backward_pass(self, out_grads):
        x_grads = np.zeros(self.input.shape)

        for coord in self.coords:
            x_grads[coord[3], coord[4], coord[0]] += out_grads[coord[1], coord[2], coord[0]]

        return x_grads

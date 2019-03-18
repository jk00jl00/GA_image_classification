import numpy as np
import math as math
import cv2 as cv
import Layer


class ConvLayer(Layer.Layer):

    def __init__(self, input_dim, depth, size, stride, padding):

        if 'depth' not in input_dim:
            input_dim['depth'] = 1

        # Calculating the output width and height where the input has the width and height
        self.output_width = math.floor((input_dim['width'] - size + 2 * padding) / stride + 1)
        self.output_height = math.floor((input_dim['height'] - size + 2 * padding) / stride + 1)
        output_dim = [self.output_height, self.output_width, depth]

        Layer.Layer.__init__(self, input_dim, output_dim)
        self.depth = depth
        self.size = size
        self.stride = stride
        self.padding = padding
        self.input = None
        self.w_grads = None
        self.b_grads = None
        self.output = np.zeros(output_dim)
        self.filters = np.zeros([depth, size, size, input_dim['depth']]) + 0.001
        self.bias = np.zeros([depth])

    def forward_pass(self, input):
        if len(input.shape) < 3:
            input = input.reshape([input.shape[0], input.shape[1], 1])
        padded_input = np.zeros([input.shape[0] + self.padding*2, input.shape[1] + self.padding*2, self.input_dim["depth"]])
        padded_input[self.padding: input.shape[0] + self.padding, self.padding:input.shape[1] + self.padding, :] = input
        self.input = padded_input

        for d in range(self.depth):
            y_stride = 0
            for y in range(self.output_height):
                x_stride = 0
                for x in range(self.output_width):
                    input_slice = padded_input[y_stride:y_stride + self.size, x_stride:x_stride + self.size, :]
                    self.output[y, x, d] = np.sum(input_slice * self.filters[d, :, :, :]) + self.bias[d]
                    x_stride += self.stride
                y_stride += self.stride

        return self.output

    def backward_pass(self, out_grads):
        self.w_grads = np.zeros(self.filters.shape)
        self.b_grads = np.zeros(self.bias.shape)
        x_grads = np.zeros(self.input.shape)

        for d in range(self.depth):
            y_stride = 0
            for y in range(self.output_height):
                x_stride = 0
                for x in range(self.output_width):
                    x_grads[y_stride:y_stride + self.size, x_stride:x_stride + self.size] += \
                        self.filters[d, :, :, :] * out_grads[y, x, d]
                    self.w_grads += self.input[y_stride:y_stride + self.size,
                                    x_stride:x_stride + self.size] * out_grads[y, x, d]
                    x_stride += self.stride
                y_stride += self.stride
            self.b_grads = out_grads[0, 0, d]

        return x_grads[self.padding:-self.padding, self.padding:-self.padding, :]

    def writeim(self, path):
        for l in range(self.filters.shape[0]):
            im = self.output[:, :, l]
            cv.imwrite(f"{path}_filter_{l}.png", im)

    def getL2(self):
        sum = 0

        for w in np.nditer(self.filters):
            sum += w**2 / 2
        return sum

    def update(self, rate):
        self.l2_grads = np.zeros(self.filters.flatten().shape)

        """for w in range(self.filters.flatten().size):
            self.l2_grads[w] += self.filters.flatten()[w]**2 / 2 * Layer.l2_delta

        for w in range(self.filters.flatten().size):
            self.filters.flatten()[w] += self.l2_grads[w] * -1"""
        self.filters += -Layer.l2_delta * self.filters
        self.filters += self.w_grads * -rate

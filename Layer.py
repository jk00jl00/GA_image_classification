import numpy as np
import math as math
import cv2 as cv

l2_delta = 0.00001

class Layer:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.l2_grads = None

    def forward_pass(self, input):
        pass

    def backward_pass(self, out_grads, rate):
        pass

    def writeim(self, path):
        pass

    def getL2(self):
        return 0

    def update(self, rate):
        pass

class ConvLayer(Layer):

    def __init__(self, input_dim, depth, size, stride, padding):
        self.output_width = math.floor((input_dim[1] - size + 2 * padding) / stride + 1)
        self.output_height = math.floor((input_dim[0] - size + 2 * padding) / stride + 1)
        output_dim = [self.output_height, self.output_width, depth]
        Layer.__init__(self, input_dim, output_dim)
        self.depth = depth
        self.size = size
        self.stride = stride
        self.padding = padding
        self.input = None
        self.w_grads = None
        self.b_grads = None
        self.output = np.zeros(output_dim)
        self.filters = np.random.random([depth, size, size, input_dim[2]]) * math.sqrt(2/(depth * size * size * input_dim[2]))
        self.bias = np.zeros([depth])

    def forward_pass(self, input):
        padded_input = np.zeros([input.shape[0] + self.padding*2, input.shape[1] + self.padding*2, input.shape[2]])
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

    def backward_pass(self, out_grads, rate):
        self.w_grads = np.zeros(self.filters.shape)
        self.b_grads = np.zeros(self.bias.shape)
        x_grads = np.zeros(self.input.shape)

        for d in range(self.depth):
            y_stride = 0
            for y in range(self.output_height):
                x_stride = 0
                for x in range(self.output_width):
                    x_grads[y_stride:y_stride + self.size, x_stride:x_stride + self.size] += self.filters[d, :, :, :] * out_grads[y, x, d]
                    self.w_grads += self.input[y_stride:y_stride + self.size, x_stride:x_stride + self.size] * out_grads[y, x, d]
                    x_stride += self.stride
                y_stride += self.stride
            self.b_grads = out_grads[0, 0, d]

        x_grads[x_grads < 0] = 0

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

        for w in range(self.filters.flatten().size):
            self.l2_grads[w] += self.filters.flatten()[w]**2 / 2 * l2_delta

        for w in range(self.filters.flatten().size):
            self.filters.flatten()[w] += self.filters.flatten()[w] **2 / 2 * l2_delta


class ConvLayer2d(Layer):

    def __init__(self, input_dim, depth, size, stride, padding):
        self.output_width = math.floor((input_dim[1] - size + 2 * padding) / stride + 1)
        self.output_height = math.floor((input_dim[0] - size + 2 * padding) / stride + 1)
        output_dim = [self.output_height, self.output_width, depth]
        Layer.__init__(self, input_dim, output_dim)
        self.depth = depth
        self.size = size
        self.stride = stride
        self.padding = padding
        self.input = None
        self.w_grads = None
        self.b_grads = None
        self.output = np.zeros(output_dim)
        self.filters = np.random.random([depth, size, size]) * math.sqrt(2/(depth * size * size))
        self.bias = np.zeros([depth])

    def forward_pass(self, input):
        padded_input = np.zeros([input.shape[0] + self.padding*2, input.shape[1] + self.padding*2])
        padded_input[self.padding: input.shape[0] + self.padding, self.padding:input.shape[1] + self.padding] = input
        self.input = padded_input

        for d in range(self.depth):
            y_stride = 0
            for y in range(self.output_height):
                x_stride = 0
                for x in range(self.output_width):
                    input_slice = padded_input[y_stride:y_stride + self.size, x_stride:x_stride + self.size]
                    self.output[y, x, d] = np.sum(input_slice * self.filters[d, :, :]) + self.bias[d]
                    x_stride += self.stride
                y_stride += self.stride

        return self.output

    def backward_pass(self, out_grads, rate):
        self.w_grads = np.zeros(self.filters.shape)
        self.b_grads = np.zeros(self.bias.shape)
        x_grads = np.zeros(self.input.shape)

        for d in range(self.depth):
            y_stride = 0
            for y in range(self.output_height):
                x_stride = 0
                for x in range(self.output_width):
                    x_grads[y_stride:y_stride + self.size, x_stride:x_stride + self.size] += self.filters[d, :, :] * out_grads[y, x, d]
                    self.w_grads += self.input[y_stride:y_stride + self.size, x_stride:x_stride + self.size] * out_grads[y, x, d]
                    x_stride += self.stride
                y_stride += self.stride
            self.b_grads[d] = np.sum(out_grads[..., d])

        x_grads[x_grads < 0] = 0

        return x_grads[self.padding:-self.padding,self.padding:-self.padding]

    def writeim(self, path):
        for l in range(self.filters.shape[0]):
            im = self.output[:, :, l]
            cv.imwrite(f"{path}_filter_{l}.png", im)

    def getL2(self):
        sum = 0

        for w in np.nditer(self.filters):
            sum += w**2 / 2
        return sum


class MaxPoolLayer(Layer):
    def __init__(self, input_dim, depth, size, stride, padding):
        self.output_width = math.floor((input_dim[1] - size + 2 * padding) / stride + 1)
        self.output_height = math.floor((input_dim[0] - size + 2 * padding) / stride + 1)
        output_dim = [self.output_height, self.output_width, depth]
        Layer.__init__(self, input_dim, output_dim)
        self.depth = depth
        self.size = size
        self.stride = stride
        self.padding = padding
        self.input = None
        self.coords = None
        self.filters = None
        self.output = np.zeros(output_dim)

    def forward_pass(self, input):
        padded_input = np.zeros([input.shape[0] + self.padding*2, input.shape[1] + self.padding*2, input.shape[2]])
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

    def backward_pass(self, out_grads, rate):
        x_grads = np.zeros(self.input.shape)

        for coord in self.coords:
            x_grads[coord[3], coord[4], coord[0]] += out_grads[coord[1], coord[2], coord[0]]

        return x_grads


class FullyConnectedLayer(Layer):
    def __init__(self, input_dim, output_dim):
        input_nodes = np.array(input_dim).prod()
        Layer.__init__(self, input_nodes, output_dim)
        self.input = None
        self.w_grads = None
        self.x_grads = None
        self.b_grads = None
        self.output = np.zeros([self.output_dim])
        self.bias = np.zeros([self.output_dim])
        self.filters = (np.random.random([output_dim, input_nodes])) * math.sqrt(2 / (output_dim * input_nodes))

    def forward_pass(self, input):
        self.input = input

        for d in range(self.filters.shape[0]):
            self.output[d] = np.sum(input.flatten() * self.filters[d, :]) + self.bias[d]

        return self.output

    def backward_pass(self, out_grads, rate):
        self.x_grads = np.zeros(self.input.shape)
        self.w_grads = np.zeros(self.filters.shape)
        self.b_grads = np.zeros(self.bias.shape)

        for d in range(self.filters.shape[0]):
            self.x_grads += (out_grads[d] * self.filters[d, :]).reshape(self.x_grads.shape)
            self.w_grads[d] = out_grads[d] * self.input.flatten()
            self.b_grads[d] = out_grads[d]

        return self.x_grads

    def getL2(self):
        sum = 0

        for w in np.nditer(self.filters):
            sum += w**2 / 2
        return sum


class SoftMaxLayer(Layer):
    def __init__(self, input_dim):
        Layer.__init__(self, input_dim, np.array(input_dim).prod())
        self.input = None
        self.x_grads = None
        self.b_grads = None
        self.output = None
        self.filters = None

    def forward_pass(self, input):
        self.input = input.copy()
        self.input -= np.max(self.input)

        self.output = np.zeros([self.output_dim])
        self.output = np.exp(self.input.flatten()) / np.sum(np.exp(self.input.flatten()))

        return self.output

    def get_loss(self, true_class):
        self.x_grads = np.zeros(self.input_dim)

        for i in range(self.output_dim):
            a = 1 if i == true_class else 0
            dx = -(a - self.output[i])
            self.x_grads[i] = dx

        out = self.output[true_class] if self.output[true_class] > 0 else 0.00000000000000000001

        return self.x_grads, -math.log(out)


class ReLuLayer(Layer):
    def __init__(self, input_dim):
        Layer.__init__(self, input_dim, input_dim)
        self.input = None
        self.filters = None
        self.x_grads = None
        self.output = np.zeros(input_dim)

    def forward_pass(self, input):
        self.input = input

        input_cpy = input.copy()
        input_cpy[input_cpy < 0] = 0

        self.output = input_cpy

        return self.output

    def backward_pass(self, out_grads, rate):
        self.x_grads = out_grads.copy()

        self.x_grads[self.input < 0] = 0

        return self.x_grads

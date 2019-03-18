l2_delta = 0.001


class Layer:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.l2_grads = None

    def forward_pass(self, input):
        pass

    def backward_pass(self, out_grads):
        pass

    def writeim(self, path):
        pass

    def getL2(self):
        return 0

    def update(self, rate):
        pass

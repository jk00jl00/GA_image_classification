import Layer
import json
import numpy as np
class LayerManager:

    @staticmethod
    def get_layer(layer):
        l = None

        if layer.get("type") == "Conv":
            if len(layer.get("input_dim")) > 2:
                l = Layer.ConvLayer(layer.get("input_dim"), layer.get("filters"), layer.get("size"), layer.get("stride"), layer.get("padding"))
            else:
                l = Layer.ConvLayer2d(layer.get("input_dim"), layer.get("filters"), layer.get("size"), layer.get("stride"), layer.get("padding"))

        elif layer.get("type") == "MaxPool":
            l = Layer.MaxPoolLayer(layer.get("input_dim"), layer.get("depth"), layer.get("size"), layer.get("stride"), layer.get("padding"))
        elif layer.get("type") == "SoftMax":
            l = Layer.SoftMaxLayer(layer.get("input_dim"))
        elif layer.get("type") == "FC":
            l = Layer.FullyConnectedLayer(layer.get("input_dim"), layer.get("output_dim"))

        return l


class Net:

    def __init__(self):
        self.layers = []
        self.grads = None

    def add_layer(self, layer):
        self.layers.append(LayerManager.get_layer(layer))

    def add_relu(self):
        self.layers.append(Layer.ReLuLayer(self.layers[-1].output.shape))

    def remove_layer(self, layer):
            self.layers.remove(layer)

    def forward(self, in_data):
        out = in_data
        for l in self.layers:
            out = l.forward_pass(out)
        return out

    def backwards(self, true_class):
        grads, loss = self.layers[-1].get_loss(true_class)

        for l in reversed(self.layers[:-1]):
            grads = l.backward_pass(grads, rate)

        self.grads = grads
        return loss, grads

    def update(self, rate):
        for l in reversed(self.layers[:-1]):
            l.update(rate)

    def init_from_json(self, path):
        path = "net_def.JSON" if path is None else path
        layers = []

        with open(path, "r") as f:
            layers = json.loads(f.read())["layers"]

        for l in layers:
            if l["type"] == "Conv":
                input_dim = l["input_dim"] if l["input_dim"] != "prev" else self.layers[-1].output.shape
                filters = l["filters"]
                size = l["size"]
                stride = l["stride"]
                padding = l["padding"]
                self.add_layer({"type": "Conv", "input_dim": input_dim, "filters": filters,
                                "size": size, "stride": stride, "padding": padding})
                self.add_relu()
            elif l["type"] == "MaxPool":
                input_dim = l["input_dim"] if l["input_dim"] != "prev" else self.layers[-1].output.shape
                depth = l["depth"] if l["depth"] != "prev" else self.layers[-1].output.shape[-1]
                size = l["size"]
                stride = l["stride"]
                padding = l["padding"]
                self.add_layer({"type": "MaxPool", "input_dim": input_dim, "depth": depth,
                                "size": size, "stride": stride, "padding": padding})
            elif l["type"] == "FC":
                input_dim = l["input_dim"] if l["input_dim"] != "prev" else self.layers[-1].output.shape
                output_dim = l["output_dim"]
                self.add_layer({"type": "FC", "input_dim": input_dim, "output_dim": output_dim})
                self.add_relu()
            elif l["type"] == "SoftMax":
                input_dim = l["input_dim"] if l["input_dim"] != "prev" else self.layers[-1].output.shape
                self.add_layer({"type": "SoftMax", "input_dim": input_dim})

    def weights_to_json(self, path):
        path = "weights.JSON" if path is None else path
        weights = []

        for l in self.layers:
            if l.filters is not None:
                l_weights = l.filters.flatten().copy()
                weights.append({"shape": l.filters.shape, "numbers": l_weights.tolist()})
            else:
                weights.append({"shape": None, "numbers": None})

        with open(path, "w") as f:
            f.write(json.dumps({"weights": weights}))

    def weights_from_json(self, path):
        path = "weights.JSON" if path is None else path

        jobj = {}

        with open(path, "r") as f:
            jobj = json.loads(f.read())

        for l in range(len(self.layers)):
            if jobj["weights"][l]["shape"] is None:
                continue
            self.layers[l].filters = np.array(jobj["weights"][l]["numbers"]).reshape(jobj["weights"][l]["shape"])

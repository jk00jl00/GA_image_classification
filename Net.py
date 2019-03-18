import Layer
import json
import numpy as np
import ConvLayer
import SoftMaxLayer
import MaxPoolLayer
import FCLayer
import ReluActivation


class LayerManager:

    @staticmethod
    def get_layer(layer):
        l = None

        if layer.get("type") == "Conv":
                l = ConvLayer.ConvLayer(layer.get("input_dim"), layer.get("filters"), layer.get("size"), layer.get("stride"), layer.get("padding"))
        elif layer.get("type") == "MaxPool":
            l = MaxPoolLayer.MaxPoolLayer(layer.get("input_dim"), layer.get("depth"), layer.get("size"), layer.get("stride"), layer.get("padding"))
        elif layer.get("type") == "SoftMax":
            l = SoftMaxLayer.SoftMaxLayer(layer.get("input_dim"))
        elif layer.get("type") == "FC":
            l = FCLayer.FCLayer(layer.get("input_dim"), layer.get("output_dim"))

        return l


class Net:

    def __init__(self):
        self.layers = []
        self.grads = None

    def add_layer(self, layer):
        self.layers.append(LayerManager.get_layer(layer))

    def add_relu(self):
        self.layers.append(ReluActivation.ReLuLayer(
            {'height': self.layers[-1].output.shape[0], 'width': self.layers[-1].output.shape[1],
             'depth': self.layers[-1].output.shape[2]}))

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
            grads = l.backward_pass(grads)

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
                input_dim = l["input_dim"] if l["input_dim"] != "prev" else \
                    {"height":self.layers[-1].output.shape[0], "width": self.layers[-1].output.shape[1]}
                if len(self.layers) > 0:
                    if self.layers[-1].output.size > 2:
                        input_dim['depth'] = self.layers[-1].output.shape[2]
                    else:
                        input_dim['depth'] = 1
                filters = l["filters"]
                size = l["size"]
                stride = l["stride"]
                padding = l["padding"]
                self.add_layer({"type": "Conv", "input_dim": input_dim, "filters": filters,
                                "size": size, "stride": stride, "padding": padding})
                self.add_relu()
            elif l["type"] == "MaxPool":
                input_dim = l["input_dim"] if l["input_dim"] != "prev" else \
                    {"height":self.layers[-1].output.shape[0], "width": self.layers[-1].output.shape[1]}
                if self.layers[-1].output.size > 2:
                    input_dim['depth'] = self.layers[-1].output.shape[2]
                else:
                    input_dim['depth'] = 1
                depth = l["depth"] if l["depth"] != "prev" else self.layers[-1].output.shape[-1]
                size = l["size"]
                stride = l["stride"]
                padding = l["padding"]
                self.add_layer({"type": "MaxPool", "input_dim": input_dim, "depth": depth,
                                "size": size, "stride": stride, "padding": padding})
            elif l["type"] == "FC":
                input_dim = l["input_dim"] if l["input_dim"] != "prev" else \
                    {"height":self.layers[-1].output.shape[0], "width": self.layers[-1].output.shape[1]}
                if self.layers[-1].output.size > 2:
                    input_dim['depth'] = self.layers[-1].output.shape[2]
                else:
                    input_dim['depth'] = 1
                output_dim = {'depth': l["output_dim"]}
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

import Net
import cv2 as cv
import numpy as np
import mnist_labels


class Trainer:
    def __init__(self, net):
        self.net = net
        self.rate = 0.01

    def train_mnist_digits(self, batches, info_every_n):
        self.train_mnist_digits(0, batches, info_every_n, None, None)

    def train_mnist_digits(self, from_batch, batches, info_every_n, save_activations_every, activation_path):
        for i in range(batches - from_batch):
            training_batch = cv.imread(f"mnist_batch_{from_batch + i}.png", 0)
            correct = 0
            incorrect = 0
            loss_total = 0

            for c in range(training_batch.shape[0]):
                num_im = training_batch[c, :].reshape([28, 28])

                guess = self.net.forward(num_im)
                loss, grads = self.net.backwards(mnist_labels.labels[(from_batch + i)*3000 + c])

                L2_loss = 0

                for l in self.net.layers:
                    L2_loss += l.getL2()

                loss_total += loss + L2_loss

                b = 0
                for a in range(guess.size):
                    if guess[a] > guess[b]:
                        b = a

                if b == mnist_labels.labels[(from_batch + i) * 3000 + c]:
                    correct += 1
                else:
                    incorrect += 1

                if save_activations_every is not None:
                    if c % save_activations_every == 0 and c != 0:
                        cv.imwrite(f"{activation_path}_{i * 3000 + c}.png", num_im)
                        for l in range(len(self.net.layers)):
                            self.net.layers[l].writeim(f"{activation_path}_{i * 3000 + c}_layer{l}")

                if c % info_every_n == 0 and c != 0:
                    print(f"Guess num {i * 3000 + c}: {b}, True class: {mnist_labels.labels[(from_batch + i) * 3000 + c]} \nLoss: {loss_total/info_every_n:3.2}")
                    print(f"Correct / total = {correct/(incorrect + correct):.1%}")
                    correct = 0
                    incorrect = 0
                    loss_total = 0

                self.net.update(self.rate)


n = Net.Net()

n.init_from_json(None)

trainer = Trainer(n)

trainer.train_mnist_digits(0, 1, 100, 100, "img/activation")

n.weights_to_json("weights_2_0.JSON")

import numpy as np
import numpy.typing as npt
from typing import Any, Tuple, List, Optional, Dict
import time

from loader import MnistDataLoader

class Parameter:
    data: npt.NDArray[Any]
    grad: npt.NDArray[Any]
    m: npt.NDArray[Any]
    v: npt.NDArray[Any]

    def __init__(self, data: npt.NDArray[Any]) -> None:
        self.data = data
        self.grad = np.zeros_like(data, dtype=np.float32)
        self.m = np.zeros_like(data, dtype=np.float32)
        self.v = np.zeros_like(data, dtype=np.float32)

class AdamOptimiser:
    lr: float
    beta1: float
    beta2: float
    eps: float
    weight_decay: float
    t: int

    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8, weight_decay: float = 1e-4) -> None:
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = epsilon
        self.weight_decay = weight_decay
        self.t = 0

    def step(self, parameters: List[Parameter]) -> None:
        self.t += 1

        for p in parameters:
            if self.weight_decay > 0.0:
                p.grad += self.weight_decay * p.data

            p.m = self.beta1 * p.m + (1.0 - self.beta1) * p.grad
            p.v = self.beta2 * p.v + (1.0 - self.beta2) * (p.grad ** 2)

            m_line = p.m / (1.0 - self.beta1 ** self.t)
            v_line = p.v / (1.0 - self.beta2 ** self.t)

            p.data -= self.lr * m_line / (np.sqrt(v_line) + self.eps)

class Shared:
    rng: np.random.Generator = np.random.default_rng()

    @staticmethod
    def kaiming_initialization(input_size: int, output_size: int) -> npt.NDArray[Any]:
        return Shared.rng.normal(0.0, np.sqrt(2.0 / input_size), size=(input_size, output_size)).astype(np.float32)

    @staticmethod
    def bias_initialization(output_size: int) -> npt.NDArray[Any]:
        return np.zeros((1, output_size), dtype=np.float32)

    @staticmethod
    def softmax(inputs: npt.NDArray[Any]) -> npt.NDArray[Any]:
        shifted_inputs = inputs - np.max(inputs, axis=1, keepdims=True)
        exps = np.exp(shifted_inputs).astype(np.float32)
        return exps / np.sum(exps, axis=1, keepdims=True)

class DenseLayer:
    W: Parameter
    b: Parameter
    inputs: Optional[npt.NDArray[Any]]

    def __init__(self, input_size: int, output_size: int) -> None:
        self.W = Parameter(Shared.kaiming_initialization(input_size, output_size))
        self.b = Parameter(Shared.bias_initialization(output_size))
        self.inputs = None

    def forward(self, inputs: npt.NDArray[Any]) -> npt.NDArray[Any]:
        self.inputs = inputs
        return np.dot(inputs, self.W.data) + self.b.data

    def backward(self, d_outputs: npt.NDArray[Any]) -> npt.NDArray[Any]:
        assert self.inputs is not None

        self.W.grad = np.dot(self.inputs.T, d_outputs)
        self.b.grad = np.sum(d_outputs, axis=0, keepdims=True)

        d_inputs = np.dot(d_outputs, self.W.data.T)
        return d_inputs

    def parameters(self) -> List[Parameter]:
        return [self.W, self.b]

class ReLULayer:
    inputs: Optional[npt.NDArray[Any]]

    def __init__(self) -> None:
        self.inputs = None

    def forward(self, inputs: npt.NDArray[Any]) -> npt.NDArray[Any]:
        self.inputs = inputs
        return np.maximum(0.0, inputs, dtype=np.float32)

    def backward(self, d_outputs: npt.NDArray[Any]) -> npt.NDArray[Any]:
        assert self.inputs is not None

        relu_derivative = (self.inputs > 0).astype(np.float32)
        return d_outputs * relu_derivative

    def parameters(self) -> List[Parameter]:
        return []

class SoftmaxCrossEntropy:
    probabilities: Optional[npt.NDArray[Any]]
    expected: Optional[npt.NDArray[Any]]

    def __init__(self) -> None:
        self.probabilities = None
        self.expected = None

    def forward(self, inputs: npt.NDArray[Any], expected: npt.NDArray[Any]) -> Tuple[float, npt.NDArray[Any]]:
        self.expected = expected
        self.probabilities = Shared.softmax(inputs)

        batch_size = inputs.shape[0]
        prob = self.probabilities + 1e-15
        loss = float(-np.sum(expected * np.log(prob)) / batch_size)

        return loss, self.probabilities

    def backward(self) -> npt.NDArray[Any]:
        assert self.expected is not None
        assert self.probabilities is not None

        batch_size = self.expected.shape[0]
        return (self.probabilities - self.expected) / batch_size

class MultiLayerPerceptron:
    optimiser: AdamOptimiser
    layers: List[Any]
    loss_activation: SoftmaxCrossEntropy

    def __init__(self, optimiser: AdamOptimiser) -> None:
        self.optimiser = optimiser
        self.layers = [
            DenseLayer(784, 256),
            ReLULayer(),
            DenseLayer(256, 128),
            ReLULayer(),
            DenseLayer(128, 10)
        ]
        self.loss_activation = SoftmaxCrossEntropy()

    def predict(self, inputs: npt.NDArray[Any]) -> npt.NDArray[Any]:
        current = inputs
        for layer in self.layers:
            current = layer.forward(current)

        return Shared.softmax(current)

    def train_step(self, X_batch: npt.NDArray[Any], Y_batch: npt.NDArray[Any]) -> float:
        current = X_batch
        for layer in self.layers:
            current = layer.forward(current)

        loss, _ = self.loss_activation.forward(current, Y_batch)

        gradient = self.loss_activation.backward()
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)

        all_params = []
        for layer in self.layers:
            all_params.extend(layer.parameters())

        self.optimiser.step(all_params)

        return loss

    def save_weights(self, filepath: str) -> None:
        weights_dictionary: Dict[str, npt.NDArray[Any]] = {}
        for index, layer in enumerate(self.layers):
            parameters = layer.parameters()
            if parameters:
                weights_dictionary[f"layer_{index}_W"] = parameters[0].data
                weights_dictionary[f"layer_{index}_b"] = parameters[1].data
        np.savez_compressed(filepath, allow_pickle=True, **weights_dictionary)

    def load_weights(self, filepath: str) -> None:
        checkpoint = np.load(filepath)
        for index, layer in enumerate(self.layers):
            parameters = layer.parameters()
            if parameters:
                parameters[0].data = checkpoint[f"layer_{index}_W"]
                parameters[1].data = checkpoint[f"layer_{index}_b"]

def evaluate(model: MultiLayerPerceptron, X_test: npt.NDArray[Any], y_test_raw: npt.NDArray[Any]) -> float:
    probabilities = model.predict(X_test)
    predictions = np.argmax(probabilities, axis=1)
    correct = np.sum(predictions == y_test_raw)
    return float((correct / len(y_test_raw)) * 100.0)

def train_mnist() -> None:
    train_images_path = "./data/train-images-idx3-ubyte"
    train_labels_path = "./data/train-labels-idx1-ubyte"
    test_images_path = "./data/t10k-images-idx3-ubyte"
    test_labels_path = "./data/t10k-labels-idx1-ubyte"

    print("Loading FULL MNIST datasets (60,000 Training | 10,000 Test)...")

    X_train_full = MnistDataLoader.load_images(train_images_path).astype(np.float32)
    Y_train_full = MnistDataLoader.load_labels(train_labels_path, one_hot=True).astype(np.float32)

    X_test = MnistDataLoader.load_images(test_images_path).astype(np.float32)
    y_test_raw = MnistDataLoader.load_labels(test_labels_path, one_hot=False)

    indices = np.arange(X_train_full.shape[0])
    Shared.rng.shuffle(indices)
    X_train_full = X_train_full[indices]
    Y_train_full = Y_train_full[indices]

    split_idx = 50000
    X_train = X_train_full[:split_idx]
    Y_train = Y_train_full[:split_idx]

    X_val = X_train_full[split_idx:]
    Y_val_raw = np.argmax(Y_train_full[split_idx:], axis=1)

    optimiser = AdamOptimiser(learning_rate=0.001)
    mlp = MultiLayerPerceptron(optimiser)

    epochs = 10
    batch_size = 128
    num_samples = X_train.shape[0]
    num_batches = int(np.ceil(num_samples / batch_size))

    validation_accuracy = 0.0

    print(f"\nTraining on {num_samples} samples │ Validation: {len(X_val)} │ Batch Size: {batch_size} │ Epochs: {epochs}")
    print("-" * 80)

    for epoch in range(epochs):
        start_time = time.time()

        if epoch == 5:
            mlp.optimiser.lr = 0.0005
        elif epoch == 8:
            mlp.optimiser.lr = 0.0001

        indices = np.arange(num_samples)
        Shared.rng.shuffle(indices)
        X_shuffled = X_train[indices]
        Y_shuffled = Y_train[indices]

        epoch_loss = 0.0

        for batch in range(num_batches):
            start_index = batch * batch_size
            end_index = start_index + batch_size

            X_batch = X_shuffled[start_index:end_index]
            Y_batch = Y_shuffled[start_index:end_index]

            loss = mlp.train_step(X_batch, Y_batch)
            epoch_loss += loss

        mean_loss = epoch_loss / num_batches
        validation_accuracy = evaluate(mlp, X_val, Y_val_raw)
        elapsed = time.time() - start_time

        print(f"Epoch {epoch + 1:2d}/{epochs} │ LR: {mlp.optimiser.lr:.5f} │ Loss: {mean_loss:.4f} │ Validation Accuracy: {validation_accuracy:5.2f}% │ Time: {elapsed:.2f}s")

    print("-" * 80)
    final_test_accuracy = evaluate(mlp, X_test, y_test_raw)
    print(f"Final Test Accuracy (10,000 samples): {final_test_accuracy:.2f}%")
    mlp.save_weights("./weights/mnist-mlp")

if __name__ == "__main__":
    train_mnist()

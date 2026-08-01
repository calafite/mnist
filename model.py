import numpy as np
import numpy.typing as npt
from typing import Any, Dict, Tuple, List, Union, Optional
import time

from loader import MnistDataLoader

class AdamOptimiser:
    lr: float
    beta1: float
    beta2: float
    eps: float
    m: Dict[int, Dict[str, npt.NDArray[Any]]]
    v: Dict[int, Dict[str, npt.NDArray[Any]]]
    t: int

    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8) -> None:
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = epsilon
        self.m = {} # Momentum
        self.v = {} # Variance
        self.t = 0 # Time step

    def update(self, layer: 'DenseLayer', dW: npt.NDArray[Any], db: npt.NDArray[Any]) -> None:
        self.t += 1
        layer_id = id(layer)

        if layer_id not in self.m:
            m, v = self.initialise_dicts(dW, db)
            self.m[layer_id] = m
            self.v[layer_id] = v

        # Biased first moment estimate, Momentum Update
        self.m[layer_id]['W'] = self.beta1 * self.m[layer_id]['W'] + (1 - self.beta1) * dW
        self.m[layer_id]['b'] = self.beta1 * self.m[layer_id]['b'] + (1 - self.beta1) * db

        # Biased second raw moment estimate, RMSProp update
        self.v[layer_id]['W'] = self.beta2 * self.v[layer_id]['W'] + (1 - self.beta2) * (dW ** 2)
        self.v[layer_id]['b'] = self.beta2 * self.v[layer_id]['b'] + (1 - self.beta2) * (db ** 2)

        # Bias corrected first moment estimate
        m_line_W = self.m[layer_id]['W'] / (1 - self.beta1 ** self.t)
        m_line_b = self.m[layer_id]['b'] / (1 - self.beta1 ** self.t)

        # Bias corrected second moment estimate
        v_line_W = self.v[layer_id]['W'] / (1 - self.beta2 ** self.t)
        v_line_b = self.v[layer_id]['b'] / (1 - self.beta2 ** self.t)

        layer.weights -= self.lr * m_line_W / (np.sqrt(v_line_W) + self.eps)
        layer.bias -= self.lr * m_line_b / (np.sqrt(v_line_b) + self.eps)

    @staticmethod
    def initialise_dicts(dW: npt.NDArray[Any], db: npt.NDArray[Any]) -> Tuple[Dict[str, npt.NDArray[Any]], Dict[str, npt.NDArray[Any]]]:
        W = np.zeros_like(dW)
        b = np.zeros_like(db)

        m = {'W': W.copy(), 'b': b.copy()}
        v = {'W': W.copy(), 'b': b.copy()}

        return (m, v)

class Shared:
    rng: np.random.Generator = np.random.default_rng()

    @staticmethod
    def kaiming_initialization(input_size: int, output_size: int) -> npt.NDArray[Any]:
        return Shared.rng.random((input_size, output_size)) * np.sqrt(2.0 / input_size)

class DenseLayer:
    weights: npt.NDArray[Any]
    bias: npt.NDArray[Any]
    inputs: Optional[npt.NDArray[Any]] # backpropagation cache

    def __init__(self, input_size: int, output_size: int) -> None:
        self.weights = Shared.kaiming_initialization(input_size, output_size)
        self.bias = np.zeros((1, output_size)) # 1 row x output_size matrix cols
        self.inputs = None # backpropagation cache

    def forward(self, inputs: npt.NDArray[Any]) -> npt.NDArray[Any]:
        self.inputs = inputs
        return np.dot(inputs, self.weights) + self.bias

    def backward(self, d_outputs: npt.NDArray[Any]) -> Tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
        assert self.inputs is not None

        dW = np.dot(self.inputs.T, d_outputs)
        db = np.sum(d_outputs, axis=0, keepdims=True)

        d_inputs = np.dot(d_outputs, self.weights.T)

        return dW, db, d_inputs

class ReLULayer:
    inputs: Optional[npt.NDArray[Any]]

    def __init__(self) -> None:
        self.inputs = None

    def forward(self, inputs: npt.NDArray[Any]) -> npt.NDArray[Any]:
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, d_outputs: npt.NDArray[Any]) -> npt.NDArray[Any]:
        assert self.inputs is not None

        relu_derivative = (self.inputs > 0).astype(np.float32)
        return d_outputs * relu_derivative

class SoftmaxCrossEntropy:
    probabilities: Optional[npt.NDArray[Any]]
    expected: Optional[npt.NDArray[Any]]

    def __init__(self) -> None:
        self.probabilities = None
        self.expected = None

    def forward(self, inputs: npt.NDArray[Any], expected: npt.NDArray[Any]) -> Tuple[float, npt.NDArray[Any]]:
        self.expected = expected

        # shifitng inputs to avoid np exp overflow
        shifted_inputs = inputs - np.max(inputs, axis=1, keepdims=True)
        exps = np.exp(shifted_inputs)

        self.probabilities = exps / np.sum(exps, axis=1, keepdims=True)
        assert self.probabilities is not None

        # cross entropy loss
        batch_size = inputs.shape[0]
        prob = self.probabilities + 1e-15 # to avoid log(0)
        loss = float(-np.sum(expected * np.log(prob)) / batch_size)

        return loss, self.probabilities

    def backward(self) -> npt.NDArray[Any]:
        assert self.expected is not None
        assert self.probabilities is not None

        batch_size = self.expected.shape[0]
        return (self.probabilities - self.expected) / batch_size

class MultiLayerPerceptron:
    optimiser: AdamOptimiser
    layers: List[Union[DenseLayer, ReLULayer]]
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

        shifted = current - np.max(current, axis=1, keepdims=True)
        exps = np.exp(shifted)
        return exps / np.sum(exps, axis=1, keepdims=True)

    def train_step(self, X_batch: npt.NDArray[Any], Y_batch: npt.NDArray[Any]) -> float:
        # forward pass
        current = X_batch
        for layer in self.layers:
            current = layer.forward(current)
        loss, _ = self.loss_activation.forward(current, Y_batch)

        # backward pass
        gradient = self.loss_activation.backward()

        for layer in reversed(self.layers):
            if isinstance(layer, DenseLayer):
                dw, db, gradient = layer.backward(gradient)
                self.optimiser.update(layer, dw, db)
            else:
                gradient = layer.backward(gradient)

        return loss

def evaluate(model: MultiLayerPerceptron, X_test: npt.NDArray[Any], y_test_raw: npt.NDArray[Any]) -> float:
    probabilities = model.predict(X_test)
    predictions = np.argmax(probabilities, axis=1)
    correct = np.sum(predictions == y_test_raw)
    return float((correct / len(y_test_raw)) * 100.0)

def train_mnist() -> None:
    train_images_path = "train-images-idx3-ubyte"
    train_labels_path = "train-labels-idx1-ubyte"
    test_images_path = "t10k-images-idx3-ubyte"
    test_labels_path = "t10k-labels-idx1-ubyte"

    print("Loading FULL MNIST datasets (60,000 Training | 10,000 Test)...")

    X_train = MnistDataLoader.load_images(train_images_path)
    Y_train = MnistDataLoader.load_labels(train_labels_path, one_hot=True)

    X_test = MnistDataLoader.load_images(test_images_path)
    y_test_raw = MnistDataLoader.load_labels(test_labels_path, one_hot=False)

    optimiser = AdamOptimiser(learning_rate=0.001)
    mlp = MultiLayerPerceptron(optimiser)

    epochs = 10
    batch_size = 128
    num_samples = X_train.shape[0]
    num_batches = int(np.ceil(num_samples / batch_size))
    validation_accuracy = 0.0

    print(f"\nTraining on {num_samples} samples │ Batch Size: {batch_size} │ Epochs: {epochs}")
    print("-" * 75)

    for epoch in range(epochs):
        start_time = time.time()

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

        # metrics
        mean_loss = epoch_loss / num_batches
        validation_accuracy = evaluate(mlp, X_test, y_test_raw)
        elapsed = time.time() - start_time

        print(f"Epoch {epoch + 1:2d}/{epochs} │ Loss: {mean_loss:.4f} │ Val Accuracy: {validation_accuracy:5.2f}% │ Time: {elapsed:.2f}s")

    print("-" * 75)
    print(f"Final Test Accuracy (10,000 samples): {validation_accuracy:.2f}%")

if __name__ == "__main__":
    train_mnist()

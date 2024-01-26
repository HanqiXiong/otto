import numpy as np
from keras.datasets import mnist

# Define sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define sigmoid derivative function
def sigmoid_derivative(x):
    return x * (1 - x)

# Define neural network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.weights2 = np.random.randn(hidden_size, output_size)

    def forward(self, input):
        # Calculate hidden layer output
        self.hidden = sigmoid(np.dot(input, self.weights1))
        # Calculate output layer output
        output = sigmoid(np.dot(self.hidden, self.weights2))
        return output

    def backward(self, input, target, output, learning_rate):
        # Calculate error in output
        output_error = target - output
        # Calculate error in hidden layer
        hidden_error = np.dot(output_error, self.weights2.T) * sigmoid_derivative(self.hidden)
        # Update weights
        self.weights2 += learning_rate * np.dot(self.hidden.T, output_error)
        self.weights1 += learning_rate * np.dot(input.T, hidden_error)

    def train(self, inputs, targets, learning_rate):
        for i in range(inputs.shape[0]):
            input = inputs[i]
            target = targets[i]
            output = self.forward(input)
            self.backward(input, target, output, learning_rate)

    def predict(self, inputs):
        outputs = []
        for i in range(inputs.shape[0]):
            input = inputs[i]
            output = self.forward(input)
            outputs.append(output)
        return np.array(outputs)

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess data
X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0
y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]

# Create neural network object with 784 inputs, 64 hidden neurons, and 10 outputs
nn = NeuralNetwork(784, 64, 10)

# Train neural network on MNIST dataset
for i in range(10):
    nn.train(X_train, y_train, 0.1)

# Predict outputs for test set
y_pred = nn.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

# Calculate accuracy rate
accuracy = np.mean(y_pred == y_test)
print("Accuracy rate:", accuracy)

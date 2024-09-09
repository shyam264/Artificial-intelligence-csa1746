import numpy as np

# Activation function (sigmoid) and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Neural Network class
class FeedForwardNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases randomly
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Weight matrices
        self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.rand(self.hidden_size, self.output_size)
        
        # Biases
        self.bias_hidden = np.random.rand(1, self.hidden_size)
        self.bias_output = np.random.rand(1, self.output_size)
    
    def forward_propagation(self, X):
        # Input to hidden layer
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)
        
        # Hidden layer to output layer
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = sigmoid(self.final_input)
        
        return self.final_output
    
    def backpropagation(self, X, y, learning_rate):
        # Error at output layer
        output_error = y - self.final_output
        output_delta = output_error * sigmoid_derivative(self.final_output)
        
        # Error at hidden layer
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)
        
        # Update weights and biases
        self.weights_hidden_output += np.dot(self.hidden_output.T, output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        
        self.weights_input_hidden += np.dot(X.T, hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate
    
    def train(self, X, y, epochs=10000, learning_rate=0.01):
        for epoch in range(epochs):
            self.forward_propagation(X)
            self.backpropagation(X, y, learning_rate)
            
            if epoch % 1000 == 0:
                loss = np.mean(np.square(y - self.final_output))
                print(f'Epoch {epoch}, Loss: {loss}')

# Example usage:

if __name__ == "__main__":
    # Input data (X) and target output (y)
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    
    y = np.array([[0], [1], [1], [0]])  # XOR problem
    
    # Create the neural network
    nn = FeedForwardNeuralNetwork(input_size=2, hidden_size=2, output_size=1)
    
    # Train the network
    nn.train(X, y, epochs=10000, learning_rate=0.1)
    
    # Test the network
    print("Final output after training:")
    print(nn.forward_propagation(X))

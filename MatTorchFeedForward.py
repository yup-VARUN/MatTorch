""" THIS VERSION HAS BEEN DEPRECATED """
import numpy as np
from MatTorchParser import *

# Feed Forward Implementation
class SimpleNetwork:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.layers = []  # each layer is a dict with weights, biases, and activation

    def add_layer(self, output_dim, activation):
        # Determining the input dimension for this layer
        input_dim = self.input_dim if not self.layers else self.layers[-1]['output_dim']
        # Initialize weights and biases (simple random initialization)
        W = np.random.randn(input_dim, output_dim) * 0.1
        b = np.zeros((1, output_dim))
        self.layers.append({
            'W': W,
            'b': b,
            'activation': activation,
            'output_dim': output_dim
        })

    def forward(self, X):
        out = X
        for layer in self.layers:
            # Linear step
            out = np.dot(out, layer['W']) + layer['b']
            # Applying non-linear activation function
            if layer['activation'] == 'relu':
                out = np.maximum(0, out)
            elif layer['activation'] == 'softmax':
                exp_scores = np.exp(out - np.max(out, axis=1, keepdims=True))
                out = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return out
    
    # TODO: Implement the Adam Optimization Algorithm
    def train(self):
        pass


# DSL text (your configuration)
dsl_text = """
network MyNet
input 1000
layer 128 relu
layer 64 relu
layer 10 softmax
optimizer adam
loss crossentropy
train epochs 10 batch_size 32
"""

# Parse the DSL
net_spec = parse_dsl(dsl_text)
print("Parsed network spec:", net_spec)

# Build the network based on the parsed DSL
network = SimpleNetwork(net_spec['input_dim'])
for layer in net_spec['layers']:
    network.add_layer(layer['output_dim'], layer['activation'])

# Example: perform a forward pass with random input (simulate 5 examples)
X_sample = np.random.randn(5, net_spec['input_dim'])
output = network.forward(X_sample)
print("Feedforward output:\n", output)



################# RUNNING THE ACTUAL MTORCH  #################
################# PROGRAM FROM THE TEXT FILE #################
if __name__ == '__main__':
    # Path to your DSL file with the .mtorch extension
    dsl_filename = "example.mtorch"
    
    # Parse the DSL file
    net_spec = parse_dsl(dsl_filename)
    print("Parsed network specification:")
    print(net_spec)
    
    # Build the network using the parsed specification
    network = SimpleNetwork(net_spec['input_dim'])
    for layer in net_spec['layers']:
        network.add_layer(layer['output_dim'], layer['activation'])
    
    # Create a sample input (e.g., 5 examples)
    X_sample = np.random.randn(5, net_spec['input_dim'])
    
    # Perform a forward pass through the network
    output = network.forward(X_sample)
    print("Feedforward output:")
    print(output)
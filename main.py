from fastapi import FastAPI
from pydantic import BaseModel, conlist
import uvicorn


# Create the FastAPI application instance.
# The variable name must be `app` for Render to automatically detect it.
app = FastAPI()














import numpy as np

# A simple, custom neural network with one hidden layer.
# Dimensions: 13 inputs, 9 hidden neurons, 1 output.

# Define the input, hidden, and output layer sizes
input_size = 13
hidden_size = 9
output_size = 1

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivative of the sigmoid function."""
    return x * (1 - x)

class NeuralNetwork:
    """A simple class to hold the neural network's state."""
    def __init__(self, hidden_weights, output_weights):
        self.hidden_weights = hidden_weights
        self.output_weights = output_weights
        self.hidden_biases = np.zeros((1, hidden_size))
        self.output_biases = np.zeros((1, output_size))

def forward_propagation(NeuralNetwork, Inputs, ActivationFunction):
    """Performs forward propagation through the network."""
    # Convert inputs to a numpy array of shape (1, 13)
    inputs_array = np.array([Inputs])

    # Calculate the output of the hidden layer
    hidden_layer_input = np.dot(inputs_array, NeuralNetwork.hidden_weights) + NeuralNetwork.hidden_biases
    hidden_layer_output = ActivationFunction(hidden_layer_input)
    
    # Calculate the output of the final layer
    output_layer_input = np.dot(hidden_layer_output, NeuralNetwork.output_weights) + NeuralNetwork.output_biases
    output_layer_output = ActivationFunction(output_layer_input)
    
    # Return a list of activations for all layers
    return [hidden_layer_output, output_layer_output]

def backward_propagation(NeuralNetwork, Inputs, training_outputs):
    """Performs backward propagation and updates weights."""
    inputs_array = np.array([Inputs])

    # Forward Propagation (re-run for training)
    hidden_layer_output, output_layer_output = forward_propagation(NeuralNetwork, Inputs, sigmoid)

    # Calculate the error
    error = training_outputs - output_layer_output
    
    # Calculate the derivative of the error with respect to the output
    d_output = error * sigmoid_derivative(output_layer_output)
    
    # Calculate the error of the hidden layer
    hidden_layer_error = d_output.dot(NeuralNetwork.output_weights.T)
    d_hidden = hidden_layer_error * sigmoid_derivative(hidden_layer_output)
    
    # Update the weights
    NeuralNetwork.output_weights += hidden_layer_output.T.dot(d_output)
    NeuralNetwork.hidden_weights += inputs_array.T.dot(d_hidden)
    
    return NeuralNetwork



















# 1. Data Model: This ensures the incoming data is a list of 13 integers.
#    `conlist` is a Pydantic feature that adds a size constraint.
class FoodData(BaseModel):
    tags: conlist(int, min_length=1, max_length=13)


@app.post("/endpoint")
async def predict_favorability(Data):
    # ครู เปลี่ยนภาษากลับไปกลับมามันทำไม่ได้ครับ , OpenML ภาษา Lua ที่ส่งไปมันก็ใช้ได้ดีอยู่แล้วนะครับ
    # ทำไมต้องเปลี่ยนภาษาไปมาให้ยากด้วย?
    # มันไม่มีทางอื่นแล้ว
    
    for _ in range(4000):
        for item in Data.Food:
            Tot = 0
            CC = 0
            for C in item.Chance:
                
                for v in Data.Chance:
                Tot = Tot + 1
                CC = CC + Data.Chance[item]
                
            Output = CC/Tot
            Activations = forward_propagation(MyNetwork, item.Content, ActivationFunction)
            backward_propagation(MyNetwork, item.Content, [Output])
    

    # print(f"PRE : {i} {Inputs} {prediction}")
    
    # Clamp the value between 1 and 100.

    

    np.random.seed(1)
    hidden_weights = np.random.uniform(size=(input_size, hidden_size))
    output_weights = np.random.uniform(size=(hidden_size, output_size))
    MyNetwork = NeuralNetwork(hidden_weights, output_weights)
    
    # Example: A single set of inputs (13 binary values)
    MyInputs = Data.Food[Data.Target]
    
    # Example: Get a forward prediction
    Activations = forward_propagation(MyNetwork, Data.Food[Data.Target]Content, sigmoid)
    return {"Result": Activations[1][0]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

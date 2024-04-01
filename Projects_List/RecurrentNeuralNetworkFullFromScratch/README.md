# Recurrent Neural Network Full From Scratch

## Overview
In this project, I developed a Recurrent Neural Network (RNN) architecture entirely from scratch, abstaining from the use of any third-party modules, including numpy. The RNN is constructed using Python's prebuilt modules such as math, random, and pickle. The pivotal question arises: is there a genuine need for this approach? In my perspective, the answer leans towards no, given the prevalent availability and efficacy of powerful modules like PyTorch and TensorFlow. Nevertheless, my primary objective in undertaking this project was to showcase my proficiency and indulge in the inherent enjoyment it provided. My exploration on the internet revealed a scarcity of resources detailing the construction of an RNN without any external modules. Most tutorials and implementations heavily rely on PyTorch, TensorFlow, and numpy. Consequently, I decided to create an RNN architecture entirely devoid of third-party dependencies, exemplifying the code presented herein.

## Architecture Code
```python
# IMPORT PRE-BUILT MODULES
import random
import math
import pickle
import json

# DEFINE FUNCTIONS FOR INITIALIZING WEIGHT MATRICES AND VECTORS
def initialize_weight_matrix(rows, cols):
    return [[random.uniform(-math.sqrt(1. / rows), math.sqrt(1. / rows)) for _ in range(rows)] for _ in range(cols)]

def initialize_column_vector(a):
    return [[0] for _ in range(a)]

def initialize_zero_matrix(rows, cols):
    return [[0] * rows for _ in range(cols)]

# RECURRENT NEURAL NETWORK ARCHITECTURE
class RecurrentNeuralNetwork:
    def __init__(self, config_path):    
        with open(config_path, "r") as File:
            config_data = json.load(File)
        
        # INITIALIZE NETWORK PARAMETERS BASED ON CONFIGURATION
        self.hidden_size = config_data["hidden_size"]
        self.vocab_size = config_data["vocab_size"]
        self.sequence_length = config_data["sequence_length"]
        self.learning_rate = config_data["learning_rate"]
        self.convergence_threshold = config_data["convergence_threshold"]
        self.U = initialize_weight_matrix(self.vocab_size, self.hidden_size)
        self.V = initialize_weight_matrix(self.hidden_size, self.vocab_size)
        self.W = initialize_weight_matrix(self.hidden_size, self.hidden_size)
        self.bias = initialize_column_vector(self.hidden_size)
        self.output_bias = initialize_column_vector(self.vocab_size)
        self.memory_U = initialize_zero_matrix(self.vocab_size, self.hidden_size)
        self.memory_W = initialize_zero_matrix(self.hidden_size, self.hidden_size)
        self.memory_V = initialize_zero_matrix(self.hidden_size, self.vocab_size)
        self.memory_bias = initialize_column_vector(self.hidden_size)
        self.memory_output_bias = initialize_column_vector(self.vocab_size)

    # METHOD TO COMPUTE SOFTMAX ACTIVATION
    def softmax(self, x):
        probabilities = [math.exp(xi - max(x)) for xi in x]
        return [pi / sum(probabilities) for pi in probabilities]

    # FORWARD PASS THROUGH THE NETWORK
    def forward_pass(self, inputs, previous_hidden_state):
    	# INITIALIZE DICTIONARIES TO STORE INTERMEDIATE STATES AND OUTPUTS
        input_states, hidden_states, output_states, predicted_output = {}, {}, {}, {}
        hidden_states[-1] = previous_hidden_state[:] # INITIALIZE PREVIOUS HIDDEN STATE

        # ITERATE OVER EACH TIME STEP
        for time_step in range(len(inputs)):
        	# COMPUTE INPUT, HIDDEN, AND OUTPUT STATES
            input_states[time_step] = [0] * self.vocab_size
            input_states[time_step][inputs[time_step]] = 1
            hidden_states[time_step] = [math.tanh(
                sum(self.U[i][j] * input_states[time_step][j] for j in range(self.vocab_size)) +
                sum(self.W[i][j] * hidden_states[time_step - 1][j] for j in range(self.hidden_size)) +
                self.bias[i][0]) for i in range(self.hidden_size)]
            output_states[time_step] = [sum(self.V[i][j] * hidden_states[time_step][j] for j in range(self.hidden_size)) +
                                        self.output_bias[i][0] for i in range(self.vocab_size)]
            predicted_output[time_step] = self.softmax(output_states[time_step])
        return input_states, hidden_states, predicted_output

    # BACKWARD PASS THROUGH THE NETWORK
    def backward_pass(self, input_states, hidden_states, predicted_output, targets):
    	# INITIALIZE GRADIENT MATRICES AND VECTORS
        dU, dW, dV = [[0] * self.vocab_size for _ in range(self.hidden_size)], [[0] * self.hidden_size for _ in
                                                                              range(self.hidden_size)], [
                         [0] * self.hidden_size for _ in range(self.vocab_size)]
        dbias, d_output_bias = [[0] for _ in range(self.hidden_size)], [[0] for _ in range(self.vocab_size)]
        hidden_state_update = [0] * self.hidden_size

        # ITERATE OVER EACH TIME STEP IN REVERSE ORDER
        for time_step in reversed(range(self.sequence_length)):
            output_error = predicted_output[time_step][:]
            output_error[targets[time_step]] -= 1

            for i in range(self.vocab_size):
                for j in range(self.hidden_size):
                    dV[i][j] += output_error[i] * hidden_states[time_step][j]
                d_output_bias[i][0] += output_error[i]
            hidden_error = [sum(self.V[i][j] * output_error[i] for i in range(self.vocab_size)) + hidden_state_update[j]
                            for j in range(self.hidden_size)]
            hidden_state_recurrent_error = [(1 - hidden_states[time_step][j] * hidden_states[time_step][j]) *
                                            hidden_error[j] for j in range(self.hidden_size)]

            for i in range(self.hidden_size):
                dbias[i][0] += hidden_state_recurrent_error[i]
                for j in range(self.vocab_size):
                    dU[i][j] += hidden_state_recurrent_error[i] * input_states[time_step][j]
                for j in range(self.hidden_size):
                    dW[i][j] += hidden_state_recurrent_error[i] * hidden_states[time_step - 1][j]
            hidden_state_update = [sum(self.W[i][j] * hidden_state_recurrent_error[j] for j in range(self.hidden_size))
                                   for i in range(self.hidden_size)]

        # UPDATE PARAMETERS USING RMSPROP OPTIMIZATION
        for parameter_updates, parameters, memory in zip([dU, dW, dV, dbias, d_output_bias],
                                                        [self.U, self.W, self.V, self.bias, self.output_bias],
                                                        [self.memory_U, self.memory_W, self.memory_V, self.memory_bias,
                                                         self.memory_output_bias]):
            for i in range(len(parameters)):
                for j in range(len(parameters[0])):
                    parameter_updates[i][j] = max(-5, min(5, parameter_updates[i][j]))
                    memory[i][j] += parameter_updates[i][j] * parameter_updates[i][j]
                    parameters[i][j] += -self.learning_rate * parameter_updates[i][j] / math.sqrt(
                        memory[i][j] + 1e-8)
        return dU, dW, dV, dbias, d_output_bias

    # METHOD TO CALCULATE LOSS
    def calculate_loss(self, predicted_output, targets):
        return sum(-math.log(predicted_output[time_step][targets[time_step]]) for time_step in
                   range(self.sequence_length))
        
    # METHOD TO SAVE MODEL PARAMETERS
    def save_model(self, save_path):
        model_data = {
            "U": self.U,
            "V": self.V,
            "W": self.W,
            "bias": self.bias,
            "output_bias": self.output_bias                     
        }
        with open(save_path, "wb") as file:
            pickle.dump(model_data, file)

    # METHOD TO LOAD MODEL PARAMETERS
    def load_model(self, file_path):
        with open(file_path, "rb") as file:
            model_data = pickle.load(file)

        self.U = model_data["U"]
        self.V = model_data["V"]
        self.W = model_data["W"]
        self.bias = model_data["bias"]
        self.output_bias = model_data["output_bias"]        

    # METHOD FOR GENERATING TEXT BASED ON SEED TEXT(INFERENCE FUNCTION)
    def generate(self, seed_text, generated_length):
        input_state = [0] * self.vocab_size    
        characters = [a for a in seed_text]    
        generated_indices = []

        for i in range(len(characters)):
            index = seed_text[i]
            input_state[index] = 1
            generated_indices.append(index)
        hidden_state = [0] * self.hidden_size

        for time_step in range(generated_length):
            hidden_state = [math.tanh(
                sum(self.U[i][j] * input_state[j] for j in range(self.vocab_size)) +
                sum(self.W[i][j] * hidden_state[j] for j in range(self.hidden_size)) +
                self.bias[i][0]) for i in range(self.hidden_size)]
            output = [sum(self.V[i][j] * hidden_state[j] for j in range(self.hidden_size)) +
                      self.output_bias[i][0] for i in range(self.vocab_size)]
            probabilities = self.softmax(output)
            generated_index = random.choices(range(self.vocab_size), weights=probabilities)[0]
            input_state = [0] * self.vocab_size
            input_state[generated_index] = 1
            generated_indices.append(generated_index)
        return generated_indices	
    
    # METHOD TO CONVERT STRING TO INTEGER
    def encode(self, string_text, string_to_integer_method):
    	input_state = [0] * self.vocab_size
    	string = [string for string in string_text]
    	encoded_to_integer = []
    	
    	for i in range(len(string)):
    		int = string_to_integer_method[string[i]]
    		input_state[int] = 1
    		encoded_to_integer.append(int)		
    	return encoded_to_integer 

    # METHOD TO CONVERT INTEGER TO STRING
    def decode(self, integer_text, integer_to_string_method):
    	decoded_to_string = "".join(integer_to_string_method[integer] for integer in integer_text)
    	return decoded_to_string
```

## Train Code
```python
import json
import math
from RecurrentNeuralNetworkFullFromScratch import RecurrentNeuralNetwork

# SET HYPERPARAMETERS
file_path = "data.txt"  # ENTER YOUR DATA FILE PATH FOR TRAINING
sequence_length = 100  # ENTER LENGTH OF INPUT SEQUENCES
hidden_size = 100  # ENTER SIZE OF THE HIDDEN LAYER 
learning_rate = 5e-5  # ENTER LEARNING RATE FOR TRAINING
max_iterations = 10000  # ENTER MAXIMUM NUMBER OF TRAINING ITERATIONS
convergence_threshold = 0.01  # ENTER THRESHOLD FOR CONVERGENCE DURING TRAINING

# LOAD DATA AND PREPROCESS
with open(file_path, "r") as file:
    text_data = file.read()
vocab = sorted(list(set(text_data)))

with open("vocab.json", "w") as file:
    json.dump(vocab, file, indent=2, ensure_ascii=False)

# SIMPLE TOKENIZER
with open("vocab.json", "r") as file:
    vocab = json.load(file)

string_to_integer = {string:integer for integer, string in enumerate(vocab)}
integer_to_string = {integer:string for integer, string in enumerate(vocab)}

data_size = len(text_data)
vocab_size = len(vocab)

# INITIALIZE VARIABLES FOR INPUT AND TARGET SEQUENCES
input_start = 0
input_end = input_start + sequence_length
input_indices = [string_to_integer[string] for string in text_data[input_start:input_end]]
target_indices = [string_to_integer[string] for string in text_data[input_start + 1:input_end + 1]]
input_start += sequence_length
if input_start + sequence_length + 1 >= data_size:
    input_start = 0

# MAKE CONFIGURATION FILE
config_dict = {
  "vocab_size": vocab_size,
  "hidden_size": hidden_size,
  "sequence_length": sequence_length,
  "learning_rate": learning_rate,
  "convergence_threshold": convergence_threshold
}
with open("config.json", "w") as file:
    json.dump(config_dict, file, indent=2)                      

# TRAIN
rnn = RecurrentNeuralNetwork("config.json")

iteration_number = 0
smooth_loss = -math.log(1.0 / rnn.vocab_size) * rnn.sequence_length

# TRAINING LOOP
while smooth_loss > rnn.convergence_threshold and iteration_number <= max_iterations:
    previous_hidden_state = [0] * rnn.hidden_size
    input_states, hidden_states, predicted_output = rnn.forward_pass(input_indices, previous_hidden_state)
    dU, dW, dV, dbias, d_output_bias = rnn.backward_pass(input_states, hidden_states, predicted_output, target_indices)
    loss = rnn.calculate_loss(predicted_output, target_indices)

    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    previous_hidden_state = hidden_states[rnn.sequence_length - 1]
    if iteration_number % 500 == 0:
        sample_generation = rnn.generate(input_indices, 200)
        decoded_text = rnn.decode(sample_generation, integer_to_string)                                   
        print(f"Iteration: {iteration_number} | {max_iterations}, Loss: {smooth_loss}")
        print(f"\nSample Generation:\n{decoded_text}\n")
        rnn.save_model("model.bin")
    iteration_number += 1
print("Training is completed!\n")
```

## Inference Code
```python
import json
from RecurrentNeuralNetworkFullFromScratch import RecurrentNeuralNetwork

with open("vocab.json", "r") as file:
    vocab = json.load(file)
        
string_to_integer = {string:integer for integer, string in enumerate(vocab)}
integer_to_string = {integer:string for integer, string in enumerate(vocab)}
    
rnn = RecurrentNeuralNetwork("config.json")
rnn.load_model("model.bin")
seed_text = "Once upon a time "  # ENTER YOUR OWN TEXT FOR GENERATION
generated_length = 100  # ENTER MAX LENGTH FOR GENERATED TEXT    
encoded_text = rnn.encode(seed_text, string_to_integer) 
output = rnn.generate(encoded_text, generated_length)
decoded_text = rnn.decode(output, integer_to_string)
print(decoded_text)
```

## Full Code
```python
# IMPORT PRE-BUILT MODULES
import random
import math
import pickle
import json

# DEFINE FUNCTIONS FOR INITIALIZING WEIGHT MATRICES AND VECTORS
def initialize_weight_matrix(rows, cols):
    return [[random.uniform(-math.sqrt(1. / rows), math.sqrt(1. / rows)) for _ in range(rows)] for _ in range(cols)]

def initialize_column_vector(a):
    return [[0] for _ in range(a)]

def initialize_zero_matrix(rows, cols):
    return [[0] * rows for _ in range(cols)]

# RECURRENT NEURAL NETWORK ARCHITECTURE
class RecurrentNeuralNetwork:
    def __init__(self, config_path):            
        with open(config_path, "r") as File:
            config_data = json.load(File)
        
        # INITIALIZE NETWORK PARAMETERS BASED ON CONFIGURATION
        self.hidden_size = config_data["hidden_size"]
        self.vocab_size = config_data["vocab_size"]
        self.sequence_length = config_data["sequence_length"]
        self.learning_rate = config_data["learning_rate"]
        self.convergence_threshold = config_data["convergence_threshold"]
        self.U = initialize_weight_matrix(self.vocab_size, self.hidden_size)
        self.V = initialize_weight_matrix(self.hidden_size, self.vocab_size)
        self.W = initialize_weight_matrix(self.hidden_size, self.hidden_size)
        self.bias = initialize_column_vector(self.hidden_size)
        self.output_bias = initialize_column_vector(self.vocab_size)
        self.memory_U = initialize_zero_matrix(self.vocab_size, self.hidden_size)
        self.memory_W = initialize_zero_matrix(self.hidden_size, self.hidden_size)
        self.memory_V = initialize_zero_matrix(self.hidden_size, self.vocab_size)
        self.memory_bias = initialize_column_vector(self.hidden_size)
        self.memory_output_bias = initialize_column_vector(self.vocab_size)

    # METHOD TO COMPUTE SOFTMAX ACTIVATION
    def softmax(self, x):
        probabilities = [math.exp(xi - max(x)) for xi in x]
        return [pi / sum(probabilities) for pi in probabilities]

    # FORWARD PASS THROUGH THE NETWORK
    def forward_pass(self, inputs, previous_hidden_state):
    	# INITIALIZE DICTIONARIES TO STORE INTERMEDIATE STATES AND OUTPUTS
        input_states, hidden_states, output_states, predicted_output = {}, {}, {}, {}
        hidden_states[-1] = previous_hidden_state[:] # INITIALIZE PREVIOUS HIDDEN STATE

        # ITERATE OVER EACH TIME STEP
        for time_step in range(len(inputs)):
        	# COMPUTE INPUT, HIDDEN, AND OUTPUT STATES
            input_states[time_step] = [0] * self.vocab_size
            input_states[time_step][inputs[time_step]] = 1
            hidden_states[time_step] = [math.tanh(
                sum(self.U[i][j] * input_states[time_step][j] for j in range(self.vocab_size)) +
                sum(self.W[i][j] * hidden_states[time_step - 1][j] for j in range(self.hidden_size)) +
                self.bias[i][0]) for i in range(self.hidden_size)]
            output_states[time_step] = [sum(self.V[i][j] * hidden_states[time_step][j] for j in range(self.hidden_size)) +
                                        self.output_bias[i][0] for i in range(self.vocab_size)]
            predicted_output[time_step] = self.softmax(output_states[time_step])
        return input_states, hidden_states, predicted_output

    # BACKWARD PASS THROUGH THE NETWORK
    def backward_pass(self, input_states, hidden_states, predicted_output, targets):
    	# INITIALIZE GRADIENT MATRICES AND VECTORS
        dU, dW, dV = [[0] * self.vocab_size for _ in range(self.hidden_size)], [[0] * self.hidden_size for _ in
                                                                              range(self.hidden_size)], [
                         [0] * self.hidden_size for _ in range(self.vocab_size)]
        dbias, d_output_bias = [[0] for _ in range(self.hidden_size)], [[0] for _ in range(self.vocab_size)]
        hidden_state_update = [0] * self.hidden_size

        # ITERATE OVER EACH TIME STEP IN REVERSE ORDER
        for time_step in reversed(range(self.sequence_length)):
            output_error = predicted_output[time_step][:]
            output_error[targets[time_step]] -= 1

            for i in range(self.vocab_size):
                for j in range(self.hidden_size):
                    dV[i][j] += output_error[i] * hidden_states[time_step][j]
                d_output_bias[i][0] += output_error[i]
            hidden_error = [sum(self.V[i][j] * output_error[i] for i in range(self.vocab_size)) + hidden_state_update[j]
                            for j in range(self.hidden_size)]
            hidden_state_recurrent_error = [(1 - hidden_states[time_step][j] * hidden_states[time_step][j]) *
                                            hidden_error[j] for j in range(self.hidden_size)]

            for i in range(self.hidden_size):
                dbias[i][0] += hidden_state_recurrent_error[i]
                for j in range(self.vocab_size):
                    dU[i][j] += hidden_state_recurrent_error[i] * input_states[time_step][j]
                for j in range(self.hidden_size):
                    dW[i][j] += hidden_state_recurrent_error[i] * hidden_states[time_step - 1][j]
            hidden_state_update = [sum(self.W[i][j] * hidden_state_recurrent_error[j] for j in range(self.hidden_size))
                                   for i in range(self.hidden_size)]

        # UPDATE PARAMETERS USING RMSPROP OPTIMIZATION
        for parameter_updates, parameters, memory in zip([dU, dW, dV, dbias, d_output_bias],
                                                        [self.U, self.W, self.V, self.bias, self.output_bias],
                                                        [self.memory_U, self.memory_W, self.memory_V, self.memory_bias,
                                                         self.memory_output_bias]):
            for i in range(len(parameters)):
                for j in range(len(parameters[0])):
                    parameter_updates[i][j] = max(-5, min(5, parameter_updates[i][j]))
                    memory[i][j] += parameter_updates[i][j] * parameter_updates[i][j]
                    parameters[i][j] += -self.learning_rate * parameter_updates[i][j] / math.sqrt(
                        memory[i][j] + 1e-8)
        return dU, dW, dV, dbias, d_output_bias

    # METHOD TO CALCULATE LOSS
    def calculate_loss(self, predicted_output, targets):
        return sum(-math.log(predicted_output[time_step][targets[time_step]]) for time_step in
                   range(self.sequence_length))
        
    # METHOD TO SAVE MODEL PARAMETERS
    def save_model(self, save_path):
        model_data = {
            "U": self.U,
            "V": self.V,
            "W": self.W,
            "bias": self.bias,
            "output_bias": self.output_bias                     
        }
        with open(save_path, "wb") as file:
            pickle.dump(model_data, file)

    # METHOD TO LOAD MODEL PARAMETERS
    def load_model(self, file_path):
        with open(file_path, "rb") as file:
            model_data = pickle.load(file)

        self.U = model_data["U"]
        self.V = model_data["V"]
        self.W = model_data["W"]
        self.bias = model_data["bias"]
        self.output_bias = model_data["output_bias"]        

    # METHOD FOR GENERATING TEXT BASED ON SEED TEXT(INFERENCE FUNCTION)
    def generate(self, seed_text, generated_length):
        input_state = [0] * self.vocab_size    
        characters = [a for a in seed_text]    
        generated_indices = []

        for i in range(len(characters)):
            index = seed_text[i]
            input_state[index] = 1
            generated_indices.append(index)
        hidden_state = [0] * self.hidden_size

        for time_step in range(generated_length):
            hidden_state = [math.tanh(
                sum(self.U[i][j] * input_state[j] for j in range(self.vocab_size)) +
                sum(self.W[i][j] * hidden_state[j] for j in range(self.hidden_size)) +
                self.bias[i][0]) for i in range(self.hidden_size)]
            output = [sum(self.V[i][j] * hidden_state[j] for j in range(self.hidden_size)) +
                      self.output_bias[i][0] for i in range(self.vocab_size)]
            probabilities = self.softmax(output)
            generated_index = random.choices(range(self.vocab_size), weights=probabilities)[0]
            input_state = [0] * self.vocab_size
            input_state[generated_index] = 1
            generated_indices.append(generated_index)
        return generated_indices	
    
    # METHOD TO CONVERT STRING TO INTEGER
    def encode(self, string_text, string_to_integer_method):
    	input_state = [0] * self.vocab_size
    	string = [string for string in string_text]
    	encoded_to_integer = []
    	
    	for i in range(len(string)):
    		int = string_to_integer_method[string[i]]
    		input_state[int] = 1
    		encoded_to_integer.append(int)		
    	return encoded_to_integer 

    # METHOD TO CONVERT INTEGER TO STRING
    def decode(self, integer_text, integer_to_string_method):
    	decoded_to_string = "".join(integer_to_string_method[integer] for integer in integer_text)
    	return decoded_to_string
	
if __name__ == "__main__":
    # SET HYPERPARAMETERS
    file_path = "data.txt"  # ENTER YOUR DATA FILE PATH FOR TRAINING
    sequence_length = 100  # ENTER LENGTH OF INPUT SEQUENCES
    hidden_size = 100  # ENTER SIZE OF THE HIDDEN LAYER 
    learning_rate = 5e-5  # ENTER LEARNING RATE FOR TRAINING
    max_iterations = 10000  # ENTER MAXIMUM NUMBER OF TRAINING ITERATIONS
    convergence_threshold = 0.01  # ENTER THRESHOLD FOR CONVERGENCE DURING TRAINING
    
    # LOAD DATA AND PREPROCESS
    with open(file_path, "r") as file:
        text_data = file.read()
    vocab = sorted(list(set(text_data)))
    
    with open("vocab.json", "w") as file:
        json.dump(vocab, file, indent=2, ensure_ascii=False)
    
    # SIMPLE TOKENIZER
    with open("vocab.json", "r") as file:
        vocab = json.load(file)
    
    string_to_integer = {string:integer for integer, string in enumerate(vocab)}
    integer_to_string = {integer:string for integer, string in enumerate(vocab)}
    
    data_size = len(text_data)
    vocab_size = len(vocab)
    
    # INITIALIZE VARIABLES FOR INPUT AND TARGET SEQUENCES
    input_start = 0
    input_end = input_start + sequence_length
    input_indices = [string_to_integer[string] for string in text_data[input_start:input_end]]
    target_indices = [string_to_integer[string] for string in text_data[input_start + 1:input_end + 1]]
    input_start += sequence_length
    if input_start + sequence_length + 1 >= data_size:
        input_start = 0
    
    # MAKE CONFIGURATION FILE
    config_dict = {
      "vocab_size": vocab_size,
      "hidden_size": hidden_size,
      "sequence_length": sequence_length,
      "learning_rate": learning_rate,
      "convergence_threshold": convergence_threshold
    }
    with open("config.json", "w") as file:
        json.dump(config_dict, file, indent=2)                      
    
    # TRAIN
    rnn = RecurrentNeuralNetwork("config.json")
    
    iteration_number = 0
    smooth_loss = -math.log(1.0 / rnn.vocab_size) * rnn.sequence_length
    
    # TRAINING LOOP
    while smooth_loss > rnn.convergence_threshold and iteration_number <= max_iterations:
        previous_hidden_state = [0] * rnn.hidden_size
        input_states, hidden_states, predicted_output = rnn.forward_pass(input_indices, previous_hidden_state)
        dU, dW, dV, dbias, d_output_bias = rnn.backward_pass(input_states, hidden_states, predicted_output, target_indices)
        loss = rnn.calculate_loss(predicted_output, target_indices)
    
        smooth_loss = smooth_loss * 0.999 + loss * 0.001
        previous_hidden_state = hidden_states[rnn.sequence_length - 1]
        if iteration_number % 500 == 0:
            sample_generation = rnn.generate(input_indices, 200)
            decoded_text = rnn.decode(sample_generation, integer_to_string)                                   
            print(f"Iteration: {iteration_number} | {max_iterations}, Loss: {smooth_loss}")
            print(f"\nSample Generation:\n{decoded_text}\n")
            rnn.save_model("model.bin")
        iteration_number += 1
    print("Training is completed!\n")
    
    # INFERENCE        
    # SIMPLE TOKENIZER
    with open("vocab.json", "r") as file:
        vocab = json.load(file)
        
    string_to_integer = {string:integer for integer, string in enumerate(vocab)}
    integer_to_string = {integer:string for integer, string in enumerate(vocab)}
    
    rnn = RecurrentNeuralNetwork("config.json")    
    rnn.load_model("model.bin")
    
    seed_text = "Once upon a time "  # ENTER YOUR OWN TEXT FOR GENERATION
    generated_length = 100  # ENTER MAX LENGTH FOR GENERATED TEXT    
    encoded_text = rnn.encode(seed_text, string_to_integer) 
    output = rnn.generate(encoded_text, generated_length)
    decoded_text = rnn.decode(output, integer_to_string)
    print(decoded_text)
```

## License
This project is under the [MIT License](LICENSE.md) - see the [LICENSE.md](LICENSE.md) file for details.

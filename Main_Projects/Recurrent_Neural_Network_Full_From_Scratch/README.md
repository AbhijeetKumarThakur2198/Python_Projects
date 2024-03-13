# Main Project 2
In this project, I developed a Recurrent Neural Network (RNN) architecture entirely from scratch, abstaining from the use of any third-party modules, including numpy. The RNN is constructed using Python's prebuilt modules such as math, random, and pickle. The pivotal question arises: is there a genuine need for this approach? In my perspective, the answer leans towards no, given the prevalent availability and efficacy of powerful modules like PyTorch and TensorFlow. Nevertheless, my primary objective in undertaking this project was to showcase my proficiency and indulge in the inherent enjoyment it provided. My exploration on the internet revealed a scarcity of resources detailing the construction of an RNN without any external modules. Most tutorials and implementations heavily rely on PyTorch, TensorFlow, and numpy. Consequently, I decided to create an RNN architecture entirely devoid of third-party dependencies, exemplifying the code presented herein.

## Full Code
```python
# IMPORT PRE-BUILT MODULES
import random
import math
import pickle
import json

# SET HYPERPARAMETERS
file_path = "data.txt"
sequence_length = 500
hidden_size = 100
learning_rate = 5e-5
max_iterations = 10000
convergence_threshold = 0.01

# LOAD DATA AND PREPROCESS
with open(file_path, 'r') as f:
    text_data = f.read()
unique_characters = list(set(text_data))
character_to_index = {char: index for (index, char) in enumerate(unique_characters)}
index_to_character = {index: char for (index, char) in enumerate(unique_characters)}
data_size = len(text_data)
vocab_size = len(unique_characters)
pointer = 0
input_start = pointer
input_end = pointer + sequence_length
input_indices = [character_to_index[char] for char in text_data[input_start:input_end]]
target_indices = [character_to_index[char] for char in text_data[input_start + 1:input_end + 1]]
pointer += sequence_length
if pointer + sequence_length + 1 >= data_size:
    pointer = 0

# MAKE CONFIGURATION FILE
config_dict = {
  "vocab_size": vocab_size,
  "hidden_size": hidden_size,
  "sequence_length": sequence_length,
  "learning_rate": learning_rate
}
with open("config.json", "w") as f:
	json.dump(config_dict, f, indent=2)

# DEFINE FUNCTION
def initialize_weight_matrix(rows, cols):
    return [[random.uniform(-math.sqrt(1. / rows), math.sqrt(1. / rows)) for _ in range(rows)] for _ in range(cols)]

def initialize_column_vector(a):
    return [[0] for _ in range(a)]

def initialize_zero_matrix(rows, cols):
    return [[0] * rows for _ in range(cols)]

# RECURRENT NEURAL NETWORK ARCHITECTURE
class RecurrentNeuralNetwork:
    def __init__(self):
        pass # KEEP EMPTY
    
    def load_config(self, config):
        with open(config, 'r') as f:
            config_data = json.load(f)
        
        self.hidden_size = config_data["hidden_size"]
        self.vocab_size = config_data["vocab_size"]
        self.sequence_length = config_data["sequence_length"]
        self.learning_rate = config_data["learning_rate"]
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

    def softmax(self, x):
        probabilities = [math.exp(xi - max(x)) for xi in x]
        return [pi / sum(probabilities) for pi in probabilities]

    def forward_pass(self, inputs, previous_hidden_state):
        input_states, hidden_states, output_states, predicted_output = {}, {}, {}, {}
        hidden_states[-1] = previous_hidden_state[:]

        for time_step in range(len(inputs)):
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

    def backward_pass(self, input_states, hidden_states, predicted_output, targets):
        dU, dW, dV = [[0] * self.vocab_size for _ in range(self.hidden_size)], [[0] * self.hidden_size for _ in
                                                                              range(self.hidden_size)], [
                         [0] * self.hidden_size for _ in range(self.vocab_size)]
        dbias, d_output_bias = [[0] for _ in range(self.hidden_size)], [[0] for _ in range(self.vocab_size)]
        hidden_state_update = [0] * self.hidden_size

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

    def calculate_loss(self, predicted_output, targets):
        return sum(-math.log(predicted_output[time_step][targets[time_step]]) for time_step in
                   range(self.sequence_length))

    def generate_sample(self, hidden_state, seed_index, sample_length):
        input_state = [0] * self.vocab_size
        input_state[seed_index] = 1
        generated_indices = []

        for time_step in range(sample_length):
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

    def save_model(self, iteration):
        model_data = {
            'U': self.U,
            'V': self.V,
            'W': self.W,
            'bias': self.bias,
            'output_bias': self.output_bias                     
        }
        with open(f'model_iter_{iteration}.bin', 'wb') as file_pointer:
            pickle.dump(model_data, file_pointer)

    def load_model(self, file_path):
        with open(file_path, 'rb') as file_pointer:
            model_data = pickle.load(file_pointer)

        self.U = model_data['U']
        self.V = model_data['V']
        self.W = model_data['W']
        self.bias = model_data['bias']
        self.output_bias = model_data['output_bias']        

    def infer(self, seed_text, generated_length):
        input_state = [0] * self.vocab_size
        characters = [char for char in seed_text]
        generated_indices = []

        for i in range(len(characters)):
            index = character_to_index[characters[i]]
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

        generated_text = ''.join(index_to_character[index] for index in generated_indices)
        return generated_text

# TRAIN
rnn = RecurrentNeuralNetwork()
rnn.load_config("config.json")

iteration_number = 0
smooth_loss = -math.log(1.0 / vocab_size) * rnn.sequence_length

while smooth_loss > convergence_threshold and iteration_number < max_iterations:
    previous_hidden_state = [0] * rnn.hidden_size
    input_states, hidden_states, predicted_output = rnn.forward_pass(input_indices, previous_hidden_state)
    dU, dW, dV, dbias, d_output_bias = rnn.backward_pass(input_states, hidden_states, predicted_output, target_indices)
    loss = rnn.calculate_loss(predicted_output, target_indices)

    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    previous_hidden_state = hidden_states[rnn.sequence_length - 1]
    if iteration_number % 500 == 0:
        sample_indices = rnn.generate_sample(previous_hidden_state, input_indices[0], 200)
        generated_text = ''.join(index_to_character[index] for index in sample_indices)
        print("Processed Text:")
        print(generated_text)
        print("\nIteration: %d, Loss: %f" % (iteration_number, smooth_loss))
        rnn.save_model(iteration_number)
    iteration_number += 1

print(f"\nmax_iterations: {max_iterations} is completed!\n")
```
After training model use this code to inference the model.
## Inference Code
```python
# INFERENCE
rnn = RecurrentNeuralNetwork()
rnn.load_config("config.json")
rnn.load_model("model_iter_0.bin")

seed_text = 'Once upon a time '
generated_length = 20
generated_output = rnn.infer(seed_text, generated_length)
print(generated_output)
```

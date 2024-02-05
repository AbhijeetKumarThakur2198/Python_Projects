# Recurrent Neural Network Full From Scratch

In this project, I developed a Recurrent Neural Network (RNN) architecture entirely from scratch, abstaining from the use of any third-party modules, including numpy. The RNN is constructed using Python's prebuilt modules such as math, random, and pickle. The pivotal question arises: is there a genuine need for this approach? In my perspective, the answer leans towards no, given the prevalent availability and efficacy of powerful modules like PyTorch and TensorFlow. Nevertheless, my primary objective in undertaking this project was to showcase my proficiency and indulge in the inherent enjoyment it provided. My exploration on the internet revealed a scarcity of resources detailing the construction of an RNN without any external modules. Most tutorials and implementations heavily rely on PyTorch, TensorFlow, and numpy. Consequently, I decided to create an RNN architecture entirely devoid of third-party dependencies, exemplifying the code presented herein.

# Full code:
```python
import random    
import math
import pickle

#####\ SET HYPERPARAMETERS /#####
file_path = "input.txt"
sequence_length = 25
hidden_size = 100
learning_rate = 1e-4
##########################

class DataReader:
    def __init__(self, file_path, sequence_length):
        self.file_pointer = open(file_path, "r")
        self.text_data = self.file_pointer.read()
        unique_characters = list(set(self.text_data))
        self.character_to_index = {char: index for (index, char) in enumerate(unique_characters)}
        self.index_to_character = {index: char for (index, char) in enumerate(unique_characters)}
        self.data_size = len(self.text_data)
        self.vocab_size = len(unique_characters)
        self.pointer = 0
        self.sequence_length = sequence_length

    def next_batch(self):
        input_start = self.pointer
        input_end = self.pointer + self.sequence_length
        input_indices = [self.character_to_index[char] for char in self.text_data[input_start:input_end]]
        target_indices = [self.character_to_index[char] for char in self.text_data[input_start + 1:input_end + 1]]
        self.pointer += self.sequence_length
        if self.pointer + self.sequence_length + 1 >= self.data_size:
            self.pointer = 0
        return input_indices, target_indices

    def just_started(self):
        return self.pointer == 0

    def close(self):
        self.file_pointer.close()

# RECURRENT NEURAL NETWORK ARCHITECTURE
class RecurrentNeuralNetwork:
    def __init__(self, hidden_size, vocab_size, sequence_length, learning_rate):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.U = [[random.uniform(-math.sqrt(1. / vocab_size), math.sqrt(1. / vocab_size)) for _ in range(vocab_size)] for _ in range(hidden_size)]
        self.V = [[random.uniform(-math.sqrt(1. / hidden_size), math.sqrt(1. / hidden_size)) for _ in range(hidden_size)] for _ in range(vocab_size)]
        self.W = [[random.uniform(-math.sqrt(1. / hidden_size), math.sqrt(1. / hidden_size)) for _ in range(hidden_size)] for _ in range(hidden_size)]
        self.bias = [[0] for _ in range(hidden_size)]
        self.output_bias = [[0] for _ in range(vocab_size)]
        self.memory_U = [[0] * vocab_size for _ in range(hidden_size)]
        self.memory_W = [[0] * hidden_size for _ in range(hidden_size)]
        self.memory_V = [[0] * hidden_size for _ in range(vocab_size)]
        self.memory_bias = [[0] for _ in range(hidden_size)]
        self.memory_output_bias = [[0] for _ in range(vocab_size)]

    def softmax(self, x):
        probabilities = [math.exp(xi - max(x)) for xi in x]
        return [pi / sum(probabilities) for pi in probabilities]

    def forward_pass(self, inputs, previous_hidden_state):
        input_states, hidden_states, output_states, predicted_output = {}, {}, {}, {}
        hidden_states[-1] = previous_hidden_state[:]
        for time_step in range(len(inputs)):
            input_states[time_step] = [0] * self.vocab_size
            input_states[time_step][inputs[time_step]] = 1
            hidden_states[time_step] = [math.tanh(sum(self.U[i][j] * input_states[time_step][j] for j in range(self.vocab_size)) + sum(self.W[i][j] * hidden_states[time_step - 1][j] for j in range(self.hidden_size)) + self.bias[i][0]) for i in range(self.hidden_size)]
            output_states[time_step] = [sum(self.V[i][j] * hidden_states[time_step][j] for j in range(self.hidden_size)) + self.output_bias[i][0] for i in range(self.vocab_size)]
            predicted_output[time_step] = self.softmax(output_states[time_step])
        return input_states, hidden_states, predicted_output

    def backward_pass(self, input_states, hidden_states, predicted_output, targets):
        dU, dW, dV = [[0] * self.vocab_size for _ in range(self.hidden_size)], [[0] * self.hidden_size for _ in range(self.hidden_size)], [[0] * self.hidden_size for _ in range(self.vocab_size)]
        dbias, d_output_bias = [[0] for _ in range(self.hidden_size)], [[0] for _ in range(self.vocab_size)]
        hidden_state_update = [0] * self.hidden_size
        for time_step in reversed(range(self.sequence_length)):
            output_error = predicted_output[time_step][:]
            output_error[targets[time_step]] -= 1
            for i in range(self.vocab_size):
                for j in range(self.hidden_size):
                    dV[i][j] += output_error[i] * hidden_states[time_step][j]
                d_output_bias[i][0] += output_error[i]
            hidden_error = [sum(self.V[i][j] * output_error[i] for i in range(self.vocab_size)) + hidden_state_update[j] for j in range(self.hidden_size)]
            hidden_state_recurrent_error = [(1 - hidden_states[time_step][j] * hidden_states[time_step][j]) * hidden_error[j] for j in range(self.hidden_size)]
            for i in range(self.hidden_size):
                dbias[i][0] += hidden_state_recurrent_error[i]
                for j in range(self.vocab_size):
                    dU[i][j] += hidden_state_recurrent_error[i] * input_states[time_step][j]
                for j in range(self.hidden_size):
                    dW[i][j] += hidden_state_recurrent_error[i] * hidden_states[time_step - 1][j]
            hidden_state_update = [sum(self.W[i][j] * hidden_state_recurrent_error[j] for j in range(self.hidden_size)) for i in range(self.hidden_size)]
        for parameter_updates, parameters, memory in zip([dU, dW, dV, dbias, d_output_bias], [self.U, self.W, self.V, self.bias, self.output_bias], [self.memory_U, self.memory_W, self.memory_V, self.memory_bias, self.memory_output_bias]):            
            for i in range(len(parameters)):
                for j in range(len(parameters[0])):
                    parameter_updates[i][j] = max(-5, min(5, parameter_updates[i][j]))
                    memory[i][j] += parameter_updates[i][j] * parameter_updates[i][j]
                    parameters[i][j] += -self.learning_rate * parameter_updates[i][j] / math.sqrt(memory[i][j] + 1e-8)
        return dU, dW, dV, dbias, d_output_bias

    def calculate_loss(self, predicted_output, targets):
        return sum(-math.log(predicted_output[time_step][targets[time_step]]) for time_step in range(self.sequence_length))

    def generate_sample(self, hidden_state, seed_index, sample_length):
        input_state = [0] * self.vocab_size
        input_state[seed_index] = 1
        generated_indices = []
        for time_step in range(sample_length):
            hidden_state = [math.tanh(sum(self.U[i][j] * input_state[j] for j in range(self.vocab_size)) + sum(self.W[i][j] * hidden_state[j] for j in range(self.hidden_size)) + self.bias[i][0]) for i in range(self.hidden_size)]
            output = [sum(self.V[i][j] * hidden_state[j] for j in range(self.hidden_size)) + self.output_bias[i][0] for i in range(self.vocab_size)]
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
            'output_bias': self.output_bias,
            'iteration': iteration
        }
        with open(f'model_iter_{iteration}.pkl', 'wb') as file_pointer:
            pickle.dump(model_data, file_pointer)
                    
    def train_model(self, data_reader):
        iteration_number = 0
        convergence_threshold = 0.01
        smooth_loss = -math.log(1.0 / data_reader.vocab_size) * self.sequence_length
        while smooth_loss > convergence_threshold:
            if data_reader.just_started():
                previous_hidden_state = [0] * self.hidden_size
            inputs, targets = data_reader.next_batch()
            input_states, hidden_states, predicted_output = self.forward_pass(inputs, previous_hidden_state)
            dU, dW, dV, dbias, d_output_bias = self.backward_pass(input_states, hidden_states, predicted_output, targets)
            loss = self.calculate_loss(predicted_output, targets)            
            smooth_loss = smooth_loss * 0.999 + loss * 0.001
            previous_hidden_state = hidden_states[self.sequence_length - 1]
            if not iteration_number % 500:
                sample_indices = self.generate_sample(previous_hidden_state, inputs[0], 200)
                generated_text = ''.join(data_reader.index_to_character[index] for index in sample_indices)
                print(generated_text)
                print("\n\nIteration: %d, Loss: %f" % (iteration_number, smooth_loss))
                self.save_model(iteration_number)  
            iteration_number += 1

    def predict_text(self, data_reader, starting_text, generated_length):
        input_state = [0] * self.vocab_size
        characters = [char for char in starting_text]
        generated_indices = []
        for i in range(len(characters)):
            index = data_reader.character_to_index[characters[i]]
            input_state[index] = 1
            generated_indices.append(index)

        hidden_state = [0] * self.hidden_size
        for time_step in range(generated_length):
            hidden_state = [math.tanh(sum(self.U[i][j] * input_state[j] for j in range(self.vocab_size)) + sum(self.W[i][j] * hidden_state[j] for j in range(self.hidden_size)) + self.bias[i][0]) for i in range(self.hidden_size)]
            output = [sum(self.V[i][j] * hidden_state[j] for j in range(self.hidden_size)) + self.output_bias[i][0] for i in range(self.vocab_size)]
            probabilities = self.softmax(output)
            generated_index = random.choices(range(self.vocab_size), weights=probabilities)[0]
            input_state = [0] * self.vocab_size
            input_state[generated_index] = 1
            generated_indices.append(generated_index)
        generated_text = ''.join(data_reader.index_to_character[index] for index in generated_indices)
        return generated_text

data_reader = DataReader(file_path, sequence_length)
rnn = RecurrentNeuralNetwork(hidden_size=hidden_size, vocab_size=data_reader.vocab_size, sequence_length=sequence_length, learning_rate=learning_rate)
rnn.train_model(data_reader)
print(rnn.predict_text(data_reader, 'get', 50))
```





# INFERENCE
rnn = RecurrentNeuralNetwork()
rnn.load_config("config.json")
rnn.load_model("model_iter_0.bin")

seed_text = 'Once upon a time '
generated_length = 20
generated_output = rnn.infer(seed_text, generated_length)
print(generated_output)

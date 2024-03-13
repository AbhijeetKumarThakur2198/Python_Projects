import json
from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer("vocab.json", "merges.txt")

def load_config(config_file):               
    with open(config_file) as f:
        loaded_config = json.load(f)

    return {key: loaded_config[key] for key in [
        "block_size", "n_embd", "n_head", "n_layer", "dropout"
    ]}

config = load_config("config.json")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = TextGenerationModel().to(device)
model.load_state_dict(torch.load("model99.bin"))
model.eval()

text = "Once upon a time "
context = torch.tensor(tokenizer.encode(text).ids, dtype=torch.long, device=device).unsqueeze(0)
generated_sequence = model.generate(context)[0].tolist()
generated_text = tokenizer.decode(generated_sequence)
print(generated_text)

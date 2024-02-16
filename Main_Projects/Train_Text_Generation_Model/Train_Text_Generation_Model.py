# # PLEASE DON'T CHANGE THE CODE UNLESS YOU'RE FAMILIAR WITH PROGRAMMING. IT MIGHT CAUSE ERRORS OTHERWISE.
# CHECK AND IMPORT SOME NECESSARY MODULES
try:
    import sys # IMPORTANT FOR EXIT IF MODULE NOT FOUND
except Exception as e:
    print(f"Error: {e}")
try:
    import json # IMPORTANT FOR CONFIG FILE
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

try:
    import tqdm # TQDM TO SEE PROGRESS BAR (OPTIONAL)
    from tqdm import auto
except ModuleNotFoundError:
    print("tqdm not found; you will not be able to see the training progress bar!")
    pass
except Exception as e:
    print(f"Error: {e}")
    pass

try:
    import transformers # TRANSFORMERS ONLY FOR MODEL CONVERSATION (OPTIONAL)
    from transformers import GPT2Config, AutoModelForCausalLM, AutoTokenizer # IN THIS CODE I USE GPT2 CONFIG IF YOU WANT TO CONVERT YOUR TRAINED MODEL IN YOUR OWN CONFIG OR FORMAT PLEASE DO MANNUALY
    print("transformers found!")
except ModuleNotFoundError:
    print("transformer not found; you will not able to convert pth to hf model!")
    pass
except Exception as e:
    print(f"Error: {e}")
    pass

try:
    import torch # NECCESARY FOR TRAINING MODEL
    import torch.nn as nn
    from torch.nn import functional as F
    print("Pytorch found!")
    print("Now we are good to go!")
except ModuleNotFoundError:
    print("Pytorch not found!")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

#####\\ MANNUAL SETUP //#####
cli_user_input_mode = True 	# SET False IF YOU DON'T WANT TO USE CLI INPUT MODE 
data_path = "/your/path/data.txt" 	# REPLACE WITH YOUR ACTUAL data.txt FILE PATH
bias = True	 # SET False IF YOU DON'T WANT TO USE bias
num_head = 3	 # SET NUMBER OF HEAD SIZE 
num_layer = 9 	# SET NUMBER OF LAYERS
batch_size = 8	 # ADJUST ACCORDING TO YOUR SYSTEM RAM LIKE IF YOU HAVE 16GB RAM USE 12 batch_size
dimension = -1 	# SET DIMENSION
split_ratio = 0.8	 # SET SPLIT RATIO LIKE 80% = 0.8
block_size = 26 	# SET BLOCK_SIZE ACCORDING TO YOUR TRANING DATA
num_embd = 24	 # SET NUMBER OF EMBEDDING (REMEMBER THAT ALWAYS USE EMBEDDING NUMBER THAT DIVISIBLE BY num_head)
dropout = 0.2 	# SET DROPOUT
learning_rate = 1e-4 	# SET LEARNING RATE
dataset_type = "train" 	# SET "validation" IF YOU WANT TO USE THE VALIDATION DATASET
weight_init_mean = 0.0	 # SET WEIGHT INITIALIZATION MEAN
validation_interval = 50 	# SET VALIDATION INTERVAL
torch.manual_seed(42) 	# SET SEED TO 42 ENSURES CONSISTENT RANDOM NUMBERS FOR REPRODUCIBILITY IN PYTORCH EXPERIMENTS
standard_deviation = 0.2	 # SET STANDARD DEVIATION
total_training_steps = 500 	# SET TOTAL TRAINING STEPS
validation_iterations = 200 	# VALIDATION ITERATIONS
tensor_dtype = torch.int64 	# SET TENSOR DTYPE
gradient_clip_threshold = 0.2 	# SET GRADIENT CLIP THRESHOLD ALSO KNOW AS MAX NORM
optimizer = torch.optim.AdamW 	# OPTIMIZER
device = 'cuda' if torch.cuda.is_available() else 'cpu' 	# CHECK DEVICE 
#######################

# DEFINE FUNCTIONS
def get_user_input(prompt, input_type=int):
    while True:
        try:
            user_input = input(prompt)
            return input_type(user_input)
        except ValueError:
            print("Invalid input. Please enter a valid value.")

def edit_parameter(parameter_name, default_value, info, input_type=int):
    while True:
        print(f"Did you like to edit {parameter_name}?\n1) Yes I want to edit\n2) No keep default\n3) What is that?")
        user_choice = get_user_input("Enter your choice: ")

        if user_choice == 1:
            return get_user_input(f"Enter new {parameter_name}: ", input_type=input_type)
        elif user_choice == 2:
            return default_value
        elif user_choice == 3:
            print(f"{info}")
            print("")
            print(f"Did you want to edit {parameter_name}?\n1) Yes I want to edit\n2) No keep default")
            again_choice = get_user_input("Enter: ")

            if again_choice == 1:
                return get_user_input(f"Enter new {parameter_name}: ", input_type=input_type)
            elif again_choice == 2:
                return default_value
            else:
                print("Wrong input!")
        else:
            print("Please type only 1, 2 or 3!")

def load_and_process_data(data_path, tensor_dtype=torch.int64, get_split=split_ratio):
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
           data_text = f.read()
    except FileNotFoundError:
       print(f"Error: File not found at {data_path}")
       return None
    except Exception as e:
       print(f"Error: Unable to read file at {data_path} Error: {e}")
       return None

    characters = sorted(list(set(data_text)))
    vocab_size = len(characters)
    string_to_index = {letters: index for index, letters in enumerate(characters)}

    def encode(string):
        return [string_to_index[encoded_characters] for encoded_characters in string]

    encoded_data = torch.tensor(encode(data_text), dtype=tensor_dtype)
    split_index = int(get_split * len(encoded_data))
    train_data = encoded_data[:split_index]
    validation_data = encoded_data[split_index:]
    return train_data, validation_data, vocab_size

def get_batch(dataset_type):
    if dataset_type.lower() == "train":
    	data = train_data
    elif dataset_type.lower() == "validation":
    	data = validation_data
    else:
    	print(f"Unknown dataset_type found: {dataset_type}\nPlease use "train" or "validation" only!")
    
    indices = torch.randint(len(data) - block_size, (batch_size,))
    input_sequence = torch.stack([data[index:index+block_size] for index in indices])
    target_sequence = torch.stack([data[index+1:index+block_size+1] for index in indices])
    input_sequence, target_sequence = input_sequence.to(device), target_sequence.to(device)
    return input_sequence, target_sequence

def make_config_file():
    config = {
    "batch_size": batch_size,
    "block_size": block_size,
    "total_training_steps": total_training_steps,
    "validation_interval": validation_interval,
    "learning_rate": learning_rate,
    "device": device,
    "validation_iterations": validation_iterations,
    "num_embd": num_embd,
    "num_head": num_head,
    "num_layer": num_layer,
    "dropout": dropout,
    "bias": bias,
    "dimension": dimension,
    "standard_deviation": standard_deviation,
    "weight_init_mean": weight_init_mean,
    "gradient_clip_threshold": gradient_clip_threshold,
}
    with open('config.json', 'w') as config_file:
        json.dump(config, config_file, indent=2)

@torch.no_grad()
def compute_average_losses():
    losses_dictionary = {}
    model.eval()
    for dataset_type in ['train', 'validation']:
        losses = torch.zeros(validation_iterations)
        for k in range(validation_iterations):
            X, Y = get_batch(dataset_type)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        losses_dictionary[dataset_type] = losses.mean()
    model.train()
    return losses_dictionary

# TRANSFORMER ARCHITECTURE
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(num_embd, head_size, bias=bias)
        self.query = nn.Linear(num_embd, head_size, bias=bias)
        self.value = nn.Linear(num_embd, head_size, bias=bias)
        self.register_buffer('lower_triangular_matrix', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        weights = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        weights = weights.masked_fill(self.lower_triangular_matrix[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=dimension)
        weights = self.dropout(weights)
        value = self.value(x)
        attended_values = weights @ value
        return attended_values

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.linear_projection = nn.Linear(head_size * num_heads, num_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        concatenated_output = torch.cat([h(x) for h in self.heads], dimension)
        processed_output = self.dropout(self.linear_projection(concatenated_output))
        return processed_output

class FeedFoward(nn.Module):
    def __init__(self, num_embd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(num_embd, 4 * num_embd), nn.ReLU(), nn.Linear(4 * num_embd, num_embd), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, num_embd, num_head):
        super().__init__()
        head_size = num_embd // num_head
        self.sa = MultiHeadAttention(num_head, head_size)
        self.ffwd = FeedFoward(num_embd)
        self.ln1 = nn.LayerNorm(num_embd)
        self.ln2 = nn.LayerNorm(num_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class CustomLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, num_embd)
        self.position_embedding_table = nn.Embedding(block_size, num_embd)
        self.blocks = nn.Sequential(*[Block(num_embd, num_head=num_head) for _ in range(num_layer)])
        self.ln_f = nn.LayerNorm(num_embd)
        self.lm_head = nn.Linear(num_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=weight_init_mean, std=standard_deviation)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=weight_init_mean, std=standard_deviation)

    def forward(self, idx, targets=None):
        idx = idx.int()
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

if __name__ == "__main__":
    if cli_user_input_mode == True:
    	        
        print("Welcome, please select:\n1) Make pth model\n2) Convert pth to hf\n3) Exit")
        choice = int(input("Enter: "))        

        while True:
            if choice == 1:
                try:
                    print("Fill the parameters:")
                    data_path = edit_parameter("data_path", "/home", "Hint: `data_path` refers to the location of your data file, typically named 'data.txt'. It holds the text data used for training the model. Ensure the correct path is provided, and the file contains the necessary text for effective model learning.", input_type=str)
                    batch_size = edit_parameter("batch size", 8, "Hint: `batch_size` defines the number of training examples processed in each iteration. Typically set to 8 for efficiency, but can be adjusted based on available memory. Larger values speed up training but consume more memory. Experiment to find the right balance for your setup.", input_type=int)
                    block_size = edit_parameter("block size", 46, "Hint:`block_size` sets the size of input sequence chunks processed by the model. For optimal results, choose a value around 512. It influences the context the model considers during training and generation. Adjust it based on your specific requirements for sequence processing.", input_type=int)
                    total_training_steps = edit_parameter("total_training_steps", 5000, "Hint: `total_training_steps` sets the maximum number of training iterations. It determines how many times the model processes the entire training dataset. A common value is 5000, but adjust based on the convergence behavior and training time constraints of your specific task.", input_type=int)
                    validation_interval = edit_parameter("validation_interval", 500, "Hint: `validation_interval` sets the frequency of model evaluation during training. It determines after how many iterations the model's performance is assessed on the validation set. A recommended value is 500, but adjust based on your dataset and training dynamics for efficient monitoring.", input_type=int)
                    learning_rate = edit_parameter("learning_rate", 1e-4, "Hint:`learning_rate` determines the step size during model parameter updates. A common starting value is 1e-4. Adjustments may be necessary based on the specific task and model behavior. Experiment to find an optimal learning rate for faster convergence and better performance.", input_type=float)
                    validation_iterations = edit_parameter("validation_iterations", 200, "Hint: `validation_iterations` determines the number of iterations used for model evaluation during the validation phase. It defines how many batches are processed to assess the model's performance. A suggested value is 200, but adjust based on the size of your validation set and computational resources for weight_init_meaningful evaluations.", input_type=int)
                    num_embd = edit_parameter("num_embd", 6, "Hint:`num_embd` represents the dimensionensionality of the embedding space in your model. It determines the size of the vector used to represent each token. A recommended value is 384, which is divisible by common numbers of attention heads, like 8, 12, or 16. Experiment to find the right balance between model capacity and resource usage.", input_type=int)
                    num_head = edit_parameter("num_head", 6, "Hint: `num_head` refers to the number of attention heads in the multi-head attention mechanism. It helps the model focus on different aspects of the input simultaneously. Consider choosing a value that is a divisor of common embedding dimensionensions, like 8, 12, or 16, for efficient parallel computation. Commonly set to 6, but experiment to find the right balance between model complexity and task performance.", input_type=int)
                    num_layer = edit_parameter("num_layer", 6, "Hint: `num_layer` specifies the number of transformer layers in the model. It determines the depth and complexity of the network. A typical value is 6, but adjust based on the task complexity. More layers capture intricate patterns, but be mindful of computational resources.", input_type=int)
                    dropout = edit_parameter("dropout", 0.2, "Hint: `dropout` is a regularization technique. It randomly sets a fraction of input units to zero during training, preventing overfitting. A common starting value is 0.2. Experiment to strike a balance between regularization strength and model performance in your specific scenario.", input_type=float)
                    print("Would you like to change additional code?\n1) Yes I wanted to edit\n2) No keep default\n3) What is that?")
                    ask_additional = int(input("Enter your option: "))
                    if ask_additional == 3:
                        print("info")
                        ask_additional = int(input("Type 1, 2, or 3: "))

                    elif ask_additional == 1:
                        dimension = edit_parameter("dimension", -1, "Hint: `dimension` is an experimental code parameter that represents an additional dimensionension in the model. It can be filled based on specific requirements, contributing to model customization. If not needed, you can set it to -1 or leave it as a default value.", input_type=int)
                        weight_init_mean = edit_parameter("weight_init_mean", 0.0, "Hint: `weight_init_mean` is a parameter representing the weight_init_mean value used during weight initialization in the model. It influences the starting values of weights. The default value is 0.0, but you can experiment with different weight_init_mean values based on your model's requirements and behavior.", input_type=float)
                        standard_deviation = edit_parameter("standard_deviation", 0.2, "Hint: `standard_deviation` is the standard deviation parameter used during weight initialization in the model. It determines the spread of initial weight values around the weight_init_mean. The default value is 0.2, but adjusting this parameter allows you to control the initialization range for better convergence and model performance.", input_type=float)
                        gradient_clip_threshold = edit_parameter("gradient_clip_threshold", 1.0, "Hint: `gradient_clip_threshold` is a parameter used during gradient clipping in training. It sets the maximum allowed norm of the gradients, preventing exploding gradients. A common value is 1.0. Adjust based on your model's sensitivity to gradient scaling and the need for stable training.", input_type=float)
                        get_split = edit_parameter("Split Ratio", 0.8, "Hint: Choose", input_type=float)
                        
                        print("Select bias to:\n1) True\n2) False\n3) What is that?")
                        ask_bias = int(input("Enter your choice: "))
                        if ask_bias == 3:
                            print("Hint: The `bias` parameter is a boolean that indicates whether the model layers should include bias terms during computation. Setting it to `True` allows the layers to learn an additive bias, providing additional flexibility in capturing patterns. Adjust based on your model's complexity and requirements.")
                            print("Select bias to:\n1) True\n2) False")
                            ask_bias = int(input("Enter your choice: "))
                        elif ask_bias == 1:
                            bias = True
                        elif ask_bias == 2:
                            bias = False
                        else:
                            print("Please enter only 1, 2 or 3")                                                                      
              
                        print("Would you like to edit dataset_type?\n1) Yes, I want\n2) No, I donot\n3) What is that?")
                        ask_dataset_type = int(input("Enter your choice: "))
                        if ask_dataset_type == 3:
                            print("Info....")
                            print("Did you want to edit?\n1) Yes\n2) No")
                            again_dataset_type = int(input("Enter your choice: "))

                        elif ask_dataset_type == 1:
                            print("Chosse:\n1) Train\n2) Validation")
                            get_dataset_type = int(input("Enter: "))
                            if get_dataset_type == 1:
                                dataset_type = "train"
                            elif get_dataset_type == 2:
                                dataset_type = "validation"
                            else:
                                print("Invalid input!")
                        elif ask_dataset_type == 2:
                            dataset_type = "train"
                        else:
                            print("Invalid input!")

                        print("Did you want to edit tensor type?\n1) Yes, I want to edit\n2) No, keep the default\n3) What is that?")
                        ask_dtype = int(input("Type your choice: "))
                        if ask_dtype == 3:
                            print("Dtype info......")
                            print("Did you want to edit dtype?\n1) Yes, I wanted\n2) No, keep the default")
                            ask_dtype = int(input("Type your choice: "))

                        elif ask_dtype == 1:
                            print("Select dtype:\nInteger dtypes:\n1) int_64_signed\n2) int_8_unsigned")
                            ask_dtypeensor = int(input("Type your choice: "))
                            dtype_mapping = {
                                   1: torch.int64,
                                   2: torch.uint8,
                              }
                            tensor_dtype = dtype_mapping.get(ask_dtypeensor, None)
                            if tensor_dtype is None:
                                print("Invalid input!")
                            else:
                                print(f"Selected dtype: {tensor_dtype}")

                        elif ask_dtype == 2:
                            tensor_dtype = torch.int64
                            print(f"Selected dtype: {tensor_dtype}")
                        else:
                            print("Invalid input!")

                        train_data, validation_data, vocab_size = load_and_process_data(data_path, tensor_dtype, get_split)
                        model = CustomLanguageModel().to(device)

                        print("Did you want to edit optimizer?\n1) Yes, I want to edit\n2) No, keep default that\n3) What is that?")
                        ask_optimizer = int(input("Enter your choice: "))
                        if ask_optimizer == 3:
                            print("Info.......")
                            print("Did you want to edit optimizer?\n1) Yes, I wanted\n2) No, keep the default")
                            ask_optimizer = int(input("Enter your choice: "))

                        elif ask_optimizer == 1:
                            print("Here is the list of 5 best optimizers:\n1) AdamW\n2) SGD\n3) Adagrad\n4) RMSprop\n5) Adamax")
                            get_optimizer = int(input("Enter 1, 2, 3, 4, 5 to select an optimizer or [Press enter to use default]: "))

                            optimizer_dict = {
                                  1: torch.optim.AdamW(model.parameters(), lr=learning_rate),
                                  2: torch.optim.SGD(model.parameters(), lr=learning_rate),
                                  3: torch.optim.Adagrad(model.parameters(), lr=learning_rate),
                                  4: torch.optim.RMSprop(model.parameters(), lr=learning_rate),
                                  5: torch.optim.Adamax(model.parameters(), lr=learning_rate),
                               }

                            if get_optimizer in optimizer_dict:
                               optimizer = optimizer_dict[get_optimizer]
                            else:
                               print("Using default optimizer (AdamW).")
                               optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
                        elif ask_optimizer == 2:
                            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
                        else:
                            print("Wrong input!")

                    elif ask_additional == 2:
                        dimension, weight_init_mean, standard_deviation, bias, gradient_clip_threshold, get_split = -1, 0.0, 0.2, True, 1.0, 0.8
                        train_data, validation_data, vocab_size = load_and_process_data(data_path)
                        model = CustomLanguageModel().to(device)
                        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

                    else:
                        print("Invalid choice. Default values will be used.")
                        
                    print("Your model parameters will be", sum(p.numel() for p in model.parameters()) / 1e6, "Million")
                    print("Would you like to proceed or change parameters again?\n1) Yes, proceed it\n2) No, I will edit again")
                    ask_confirm = int(input("Enter your option: "))
                    if ask_confirm == 1:
                        pass
                    elif ask_confirm == 2:
                        continue
                    else:
                        print("Please type 1 or 2 only")

                    print("\nThis training process take time it depend what hypermeters you filled. So please wait until it show Training is completed!")
                    make_config_file()

                    try:
                        bar = auto.tqdm(range(1, total_training_steps + 1), position=0)
                    except ImportError:
                        print("tqdm not found; progress bar will not be displayed.")                        
                        bar = range(1, total_training_steps + 1)

                    for iter in bar:
                        if iter % validation_interval == 0 or iter == total_training_steps:
                            losses = compute_average_losses()
                            torch.save(model.state_dict(), f'model{iter}.pth')
                            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['validation']:.4f}, Checkpoint model saved!")
                            xb, yb = get_batch(dataset_type)
                            logits, loss = model(xb, yb)
                            optimizer.zero_grad(set_to_none=True)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_threshold)
                            loss.backward()
                            optimizer.step()
                        bar.update(1)
                    bar.close()
                    print("Training is completed!")
                    break

                except ValueError as ve:
                    print(f"ValueError: {ve}. Please enter a valid number.")
                except Exception as e:
                    print(f"Something went wrong: {e}")
           
            elif choice == 2:
                print("You exit the program!")
                break

            else:
                print("Invalid choice!")

    elif cli_user_input_mode == False:
        try:
            train_data, validation_data, vocab_size = load_and_process_data(data_path, tensor_dtype)
            model = CustomLanguageModel().to(device)
            make_config_file()
            optimizer = optimizer(model.parameters(), lr=learning_rate)
            
            print("Your model parameters will be", sum(p.numel() for p in model.parameters()) / 1e6, "Million")
            print("\nThis training process take time it depend what hypermeters you filled. So please wait until it show Training is completed!")
            try:
                bar = auto.tqdm(range(1, total_training_steps + 1), position=0)
            except ImportError:
                print("tqdm not found; progress bar will not be displayed.")
                bar = range(1, total_training_steps + 1)

            for iter in bar:
                if iter % validation_interval == 0 or iter == total_training_steps:
                    losses = compute_average_losses()
                    torch.save(model.state_dict(), f'model{iter}.pth')
                    print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['validation']:.4f}, Checkpoint model saved!")
                    xb, yb = get_batch(dataset_type)
                    logits, loss = model(xb, yb)
                    optimizer.zero_grad(set_to_none=True)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_threshold)
                    loss.backward()
                    optimizer.step()
                bar.update(1)
            bar.close()
            print("Training is completed!")

        except Exception as e:
                  print(f"Error: {e}")

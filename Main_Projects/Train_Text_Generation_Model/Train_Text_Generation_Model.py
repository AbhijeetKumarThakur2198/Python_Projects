try:    
    import sys
except Exception as e:
    print(f"Error: {e}")
try:    
    import json
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

try:
    import tqdm
    from tqdm import auto
except ModuleNotFoundError:
    print("tqdm not found; you will not be able to see the training progress bar!")
    pass
except Exception as e:
    print(f"Error: {e}")
    pass

try:
    import transformers
    from transformers import GPT2Config, AutoModelForCausalLM, AutoTokenizer
    print("transformers found!")
except ModuleNotFoundError:
    print("transformer not found; you will not able to convert pth to hf model!")
    pass
except Exception as e:
    print(f"Error: {e}")
    pass

try:
    import torch
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
user_input_mode = True       
file_path = "/content/input.txt" 
batch_size = 8
block_size = 26 
max_iters = 500
eval_interval = 50
learning_rate = 1e-4
eval_iters = 200
n_embd = 20
n_head = 6
n_layer = 6
dropout = 0.2
dim = -1
bias = True
mean = 0.0
std = 0.2
max_norm = 0.2
tensor_dtype = torch.int64
#set_optimizer = AdamW
device = 'cuda' if torch.cuda.is_available() else 'cpu'
get_sp = 0.9   
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
            return default_value
    else:
        print("Please type only 1, 2 or 3!")   
    
def load_and_process_data(file_path, tensor_dtype=torch.int64, get_sp=0.8):
    try:        
        with open(file_path, 'r', encoding='utf-8') as f:           
           data_text = f.read()
    except FileNotFoundError:       
       print(f"Error: File not found at {file_path}")
       return None
    except Exception as e:
       print(f"Error: Unable to read file at {file_path} Error: {e}")
       return None

    characters = sorted(list(set(data_text)))
    vocab_size = len(characters)
    string_to_index = {letters: index for index, letters in enumerate(characters)}

    def encode(string):
        return [string_to_index[encoded_characters] for encoded_characters in string]

    encoded_data = torch.tensor(encode(data_text), dtype=tensor_dtype)    
    split_index = int(get_sp * len(encoded_data))
    train_data = encoded_data[:split_index]
    validation_data = encoded_data[split_index:]
    return train_data, validation_data, vocab_size

def get_batch(dataset_type):
    data = train_data if dataset_type == 'train' else validation_data
    indices = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[index:index+block_size] for index in indices])
    y = torch.stack([data[index+1:index+block_size+1] for index in indices])
    x, y = x.to(device), y.to(device)
    return x, y

def make_config_file():
    config = {
    "batch_size": batch_size,
    "block_size": block_size,
    "max_iters": max_iters,
    "eval_interval": eval_interval,
    "learning_rate": learning_rate,
    "device": device,
    "eval_iters": eval_iters,
    "n_embd": n_embd,
    "n_head": n_head,
    "n_layer": n_layer,
    "dropout": dropout,
    "bias": bias,
    "dim": dim,
    "std": std,
    "mean": mean, 
    "max_norm": max_norm,
}
    with open('config.json', 'w') as config_file:
        json.dump(config, config_file, indent=2)                                            

@torch.no_grad()
def estimate_loss():
    losses_dictionary = {}
    model.eval()
    for dataset_type in ['train', 'validation']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
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
        self.key = nn.Linear(n_embd, head_size, bias=bias)
        self.query = nn.Linear(n_embd, head_size, bias=bias)
        self.value = nn.Linear(n_embd, head_size, bias=bias)
        self.register_buffer('lower_triangular_matrix', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):        
        B,T,C = x.shape
        k = self.key(x)  
        q = self.query(x)        
        weights = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 
        weights = weights.masked_fill(self.lower_triangular_matrix[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=dim) 
        weights = self.dropout(weights)       
        value = self.value(x) 
        attended_values = weights @ value
        return attended_values

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.linear_projection = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        concatenated_output = torch.cat([h(x) for h in self.heads], dim)
        processed_output = self.dropout(self.linear_projection(concatenated_output))
        return processed_output

class FeedFoward(nn.Module):  
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):  
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class CustomLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) 
        self.lm_head = nn.Linear(n_embd, vocab_size)        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=mean, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=mean, std=std)

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
    if user_input_mode == True:     
        try:        
            print("Welcome, please select:\n1) Make pth model\n2) Convert pth to hf\n3) Exit")
            choice = int(input("Enter: "))
        except ValueError:      
            print("Please type 1, 2, or 3 for exit!")
            
        while True:        
            if choice == 1:                     
                try:
                    torch.manual_seed(42)
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
                    print("Fill the parameters:")
                    file_path = edit_parameter("file_path", "/home", "Hint: `file_path` refers to the location of your data file, typically named 'data.txt'. It holds the text data used for training the model. Ensure the correct path is provided, and the file contains the necessary text for effective model learning.", input_type=str)
                    batch_size = edit_parameter("batch size", 8, "Hint: `batch_size` defines the number of training examples processed in each iteration. Typically set to 8 for efficiency, but can be adjusted based on available memory. Larger values speed up training but consume more memory. Experiment to find the right balance for your setup.", input_type=int)
                    block_size = edit_parameter("block size", 46, "Hint:`block_size` sets the size of input sequence chunks processed by the model. For optimal results, choose a value around 512. It influences the context the model considers during training and generation. Adjust it based on your specific requirements for sequence processing.", input_type=int)
                    max_iters = edit_parameter("max_iters", 5000, "Hint: `max_iters` sets the maximum number of training iterations. It determines how many times the model processes the entire training dataset. A common value is 5000, but adjust based on the convergence behavior and training time constraints of your specific task.", input_type=int)
                    eval_interval = edit_parameter("eval_interval", 500, "Hint: `eval_interval` sets the frequency of model evaluation during training. It determines after how many iterations the model's performance is assessed on the validation set. A recommended value is 500, but adjust based on your dataset and training dynamics for efficient monitoring.", input_type=int)
                    learning_rate = edit_parameter("learning_rate", 1e-4, "Hint:`learning_rate` determines the step size during model parameter updates. A common starting value is 1e-4. Adjustments may be necessary based on the specific task and model behavior. Experiment to find an optimal learning rate for faster convergence and better performance.", input_type=float)
                    eval_iters = edit_parameter("eval_iters", 200, "Hint: `eval_iters` determines the number of iterations used for model evaluation during the validation phase. It defines how many batches are processed to assess the model's performance. A suggested value is 200, but adjust based on the size of your validation set and computational resources for meaningful evaluations.", input_type=int)
                    n_embd = edit_parameter("n_embd", 6, "Hint:`n_embd` represents the dimensionality of the embedding space in your model. It determines the size of the vector used to represent each token. A recommended value is 384, which is divisible by common numbers of attention heads, like 8, 12, or 16. Experiment to find the right balance between model capacity and resource usage.", input_type=int)
                    n_head = edit_parameter("n_head", 6, "Hint: `n_head` refers to the number of attention heads in the multi-head attention mechanism. It helps the model focus on different aspects of the input simultaneously. Consider choosing a value that is a divisor of common embedding dimensions, like 8, 12, or 16, for efficient parallel computation. Commonly set to 6, but experiment to find the right balance between model complexity and task performance.", input_type=int)
                    n_layer = edit_parameter("n_layer", 6, "Hint: `n_layer` specifies the number of transformer layers in the model. It determines the depth and complexity of the network. A typical value is 6, but adjust based on the task complexity. More layers capture intricate patterns, but be mindful of computational resources.", input_type=int)
                    dropout = edit_parameter("dropout", 0.2, "Hint: `dropout` is a regularization technique. It randomly sets a fraction of input units to zero during training, preventing overfitting. A common starting value is 0.2. Experiment to strike a balance between regularization strength and model performance in your specific scenario.", input_type=float)
                    print("Would you like to change additional code?\n1) Yes I wanted to edit\n2) No keep default\n3) What is that?")
                    assk = int(input("Type 1, 2, or 3: "))                                                           
                    if assk == 3:               
                        print("info")
                        assk = int(input("Type 1, 2, or 3: "))              
                                                                                                                            
                    elif assk == 1:                               
                        dim = edit_parameter("dim", -1, "Hint: `dim` is an experimental code parameter that represents an additional dimension in the model. It can be filled based on specific requirements, contributing to model customization. If not needed, you can set it to -1 or leave it as a default value.", input_type=int)
                        mean = edit_parameter("mean", 0.0, "Hint: `mean` is a parameter representing the mean value used during weight initialization in the model. It influences the starting values of weights. The default value is 0.0, but you can experiment with different mean values based on your model's requirements and behavior.", input_type=float)
                        std = edit_parameter("std", 0.2, "Hint: `std` is the standard deviation parameter used during weight initialization in the model. It determines the spread of initial weight values around the mean. The default value is 0.2, but adjusting this parameter allows you to control the initialization range for better convergence and model performance.", input_type=float)              
                        max_norm = edit_parameter("max_norm", 1.0, "Hint: `max_norm` is a parameter used during gradient clipping in training. It sets the maximum allowed norm of the gradients, preventing exploding gradients. A common value is 1.0. Adjust based on your model's sensitivity to gradient scaling and the need for stable training.", input_type=float)
                        print("Select bias to:\n1) True\n2) False\n3) What is that?")
                        asssk = int(input("Enter your choice: "))
                        if asssk == 3:
                            print("Hint: The `bias` parameter is a boolean that indicates whether the model layers should include bias terms during computation. Setting it to `True` allows the layers to learn an additive bias, providing additional flexibility in capturing patterns. Adjust based on your model's complexity and requirements.")
                            print("Select bias to:\n1) True\n2) False")
                            asssk = int(input("Enter your choice: "))
                        elif asssk == 1:
                            bias = True
                        elif asssk == 2:
                            bias = False                                            
                        else:
                            print("Please enter only 1, 2 or 3")
    
                        print("Did you want to edit split data?\n1) Yes, I wanted to edit\n2) No, keep default\n3) What is that?")
                        ask_sp = int(input("Enter your choice: "))
                        if ask_sp == 3:
                            print("info")
                            print("Did you want to edit split data?\n1) Yes, I wanted to edit\n2) No, keep default\n3) What is that?")
                            ask_sp = int(input("Enter your choice"))
                            
                            if ask_sp == 1:
                                print("Choose:\n1) Auto Split\n2) Set Custom Num\n3) What is that?")
                                ask_sp1 = int(input("Enter your choice: "))
                                if ask_sp1 == 3:
                                    print("info")
                                    ask_sp1 = int(input("Enter your choice: "))
                                    
                                elif ask_sp1 == 1:
                                    vocab_size = load_and_process_data(file_path)
                                    min_split_ratio = 0.1
                                    max_split_ratio = 0.9
                                    auto_split_ratio = min(max_split_ratio, max(min_split_ratio, vocab_size / 1000))
                                    get_sp = auto_split_ratio
                                    
                                elif ask_sp1 == 2:
                                    get_sp = float(input("Enter number in %: "))
                                    
                                else:
                                    print("Invalid input!")                                            
            
                        print("Did you want to edit tensor type?\n1) Yes, I want to edit\n2) No, keep the default\n3) What is that?")
                        get_t = int(input("Type your choice: "))
                        if get_t == 3:                                                              
                            print("Dtype info......")
                            print("Did you want to edit dtype?\n1) Yes, I wanted\n2) No, keep the default")
                            get_t = int(input("Type your choice: "))    
                                         
                        elif get_t == 1:                            
                            print("Here is a list of dtypes:\nComplex dtypes:\nInteger dtypes:\n1) int_64_signed\n2) int_32_signed\n3) int_16_signed\n4) int_8_unsigned")
                            get_tensor = int(input("Type your choice: "))
                            dtype_mapping = {                   
                                   1: torch.int64,
                                   2: torch.int32,
                                   3: torch.int16,
                                   4: torch.uint8,                              
                              }                                                                        
                            tensor_dtype = dtype_mapping.get(get_tensor, None)
                            if tensor_dtype is None:                            
                                print("Invalid input!")
                            else:
                                print(f"Selected dtype: {tensor_dtype}")
                                                                         
                        elif get_t == 2:                                               
                            tensor_dtype = torch.int64
                            print(f"Selected dtype: {tensor_dtype}")
                        else:               
                            print("Invalid input!")                 
                                              
                        train_data, validation_data, vocab_size = load_and_process_data(file_path, tensor_dtype, get_sp)                                                   
                        model = CustomLanguageModel().to(device)
                        
                        print("Did you want to edit optimizer?\n1) Yes, I want to edit\n2) No, keep default that\n3) What is that?")
                        ask_op = int(input("Enter your choice: "))
                        if ask_op == 3:
                            print("Info.......")
                            print("Did you want to edit optimizer?\n1) Yes, I wanted\n2) No, keep the default")
                            ask_op = int(input("Enter your choice: "))                        
                                                                    
                        elif ask_op == 1:                   
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
                        elif ask_op == 2:
                            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
                        else:
                            print("Wrong input!")
                                                                                                             
                    elif assk == 2:                 
                        dim, mean, std, bias, max_norm = -1, 0.0, 0.2, True, 1.0
                        train_data, validation_data, vocab_size = load_and_process_data(file_path)                               
                        model = CustomLanguageModel().to(device)                                   
                        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
                                                                             
                    else:
                        print("Invalid choice. Default values will be used.")                                                                                                                                                                                                                                                                                                                                
                    print("Your model parameters will be", sum(p.numel() for p in model.parameters()) / 1e6, "Million")
                    print("Would you like to proceed or change parameters again?\n1) Yes, proceed it\n2) No, I will edit again")
                    ask = int(input("Type 1 or 2: "))
                    if ask == 1:
                        pass
                    elif ask == 2:
                        continue
                    else:
                        print("Please type 1 or 2 only")
    
                    print("It takes a while depending on the details filled.")
                    make_config_file()            
                                  
                    try:
                        bar = auto.tqdm(range(1, max_iters + 1), position=0)
                    except ImportError:
                        print("tqdm not found; progress bar will not be displayed.")
                        bar = range(1, max_iters + 1)
                                     
                    for iter in bar:                                    
                        if iter % eval_interval == 0 or iter == max_iters:
                            losses = estimate_loss()
                            torch.save(model.state_dict(), f'model{iter}.pth')
                            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['validation']:.4f}, Checkpoint model saved!")
                            xb, yb = get_batch('train')
                            logits, loss = model(xb, yb)
                            optimizer.zero_grad(set_to_none=True)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                            loss.backward()
                            optimizer.step()
                        bar.update(1)        
                    bar.close()                                          
                    print("Your model training is completed!")
                    break                    
                         
                except ValueError as ve:
                    print(f"ValueError: {ve}. Please enter a valid number.")
                except Exception as e:
                    print(f"Something went wrong: {e}")
    
            elif choice == 2:                                                                           
                get_pth = input("Enter your model path: ")
                get_config = input("Enter your config path: ")
                get_path = input("Enter your path were you wanted to save hf model: ")
                          
                with open(get_config) as config_file:                              
                    config = json.load(config_file)
                   
                n_embd = config["n_embd"]
                n_head = config["n_head"]
                n_layer = config["n_layer"]
                block_size = config["block_size"]
                bias = config["bias"]
                dim = config["dim"]
                mean = config["mean"]
                std = config["std"]
                batch_size = config["batch_size"]
                eval_interval = config["eval_interval"]
                dropout = config["dropout"]                                
                pytorch_model = CustomLanguageModel()
                pytorch_model.load_state_dict(torch.load(get_pth))
                pytorch_model.eval()
    
                config = GPT2Config.from_pretrained(get_config) 
                tokenizer = AutoTokenizer.from_pretrained('gpt2') 
                
                hf_model = AutoModelForCausalLM.from_config(config)
                state_dict = pytorch_model.state_dict()
                new_state_dict = hf_model.state_dict()
                            
                for key in state_dict:              
                    if key in new_state_dict:               
                        if state_dict[key].shape == new_state_dict[key].shape:                      
                            new_state_dict[key] = state_dict[key]
                
                hf_model.load_state_dict(new_state_dict)
                tokenizer.save_pretrained(get_path)
                hf_model.save_pretrained(get_path)
                print(f"HF model saved in path: {get_path}")
                break
    
            elif choice == 3:
                print("You exit the program!")
                break
    
            else:
                print("Invalid choice!")
      
    elif user_input_mode == False:
        try:    
            train_data, validation_data, vocab_size = load_and_process_data(file_path, tensor_dtype)
            model = CustomLanguageModel().to(device)
            make_config_file()
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  
            try:                                                     
                bar = auto.tqdm(range(1, max_iters + 1), position=0)
            except ImportError:         
                print("tqdm not found; progress bar will not be displayed.")
                bar = range(1, max_iters + 1)
            
            for iter in bar:
                if iter % eval_interval == 0 or iter == max_iters:
                    losses = estimate_loss()
                    torch.save(model.state_dict(), f'model{iter}.pth')
                    print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['validation']:.4f}, Checkpoint model saved!")
                    xb, yb = get_batch('train')
                    logits, loss = model(xb, yb)
                    optimizer.zero_grad(set_to_none=True)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                    loss.backward()
                    optimizer.step()
                bar.update(1)
            bar.close()
            print("Your model training is completed!")
                    
        except Exception as e:                
                  print(f"Error: {e}")

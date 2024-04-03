#######|| CONFIGURATION ||#######
max_length = 50  # SET MAX LENGTH FOR GENERATED TEXT
temperature = 0.7  # SET TEMPERATURE FOR MODEL
return_tensors = "pt"  # SET RETURN TENSORS 
num_return_sequences = 1  # SET NUMBER OF RETURN SEQUENCES
skip_special_tokens = True  # SET TRUE OR FALSE FOR SKIP SPECIAL TOKENS
seed_text = "Once upon a time "  # SET TEXT FOR GENERATION
Fine_Tuned_Model_Path = "/content/"  # SET PATH OF YOUR FINE-TUNED MODEL
############################

# IMPORT NECCESARY MODULE
try:
    import torch
except ModuleNotFoundError:
    print("Pytorch module not found in your environment please download it!")
except Exception as e:
    print(f"Error: {e}")

try:
    from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
except ModuleNotFoundError:
    print("Transformers module not found in your environment please download it!")
except Exception as e:
    print(f"Error: {e}")

# LOAD MODEL
tokenizer = GPT2Tokenizer.from_pretrained(Fine_Tuned_Model_Path)
model = GPT2LMHeadModel.from_pretrained(Fine_Tuned_Model_Path)

# ENCODE SEED TEXT INTO INTEGERS
encoded_text = tokenizer.encode(seed_text, return_tensors=return_tensors)

# Create attention mask
attention_mask = torch.ones(encoded_text.shape, dtype=torch.long)

# GENERATE OUTPUT
output = model.generate(
    encoded_text,
    max_length=max_length,
    num_return_sequences=num_return_sequences,
    temperature=temperature,
    do_sample=True,
    attention_mask=attention_mask,
    pad_token_id=tokenizer.eos_token_id
)

# DECODE INTEGERS TO STRINGS AND PRINT
decoded_text = tokenizer.decode(output[0], skip_special_tokens=skip_special_tokens)
print(decoded_text)

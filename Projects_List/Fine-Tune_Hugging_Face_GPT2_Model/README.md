# Fine-Tune GPT2 Hugging Face Model

## Overview
In this project, we aim to refine the Hugging Face GPT-2 model through fine-tuning. This strategy will facilitate the training of the GPT-2 model to recognize and generate text based on specific patterns.

The process involves selecting a pre-trained GPT-2 model from the Hugging Face library and adjusting its parameters to adapt it to the desired pattern recognition task. By fine-tuning the model on a specific dataset or corpus, we can enhance its ability to generate coherent and contextually relevant text within the defined pattern.

Furthermore, fine-tuning the GPT-2 model enables us to customize its responses to better suit the requirements of our project, whether it's generating creative fiction, assisting in natural language understanding tasks, or aiding in text completion applications.

Through this approach, we can harness the power of transfer learning to leverage the knowledge encoded in the pre-trained GPT-2 model and further refine it to excel in our specific domain or task at hand. This not only saves computational resources and time but also ensures that the model becomes adept at generating high-quality text outputs tailored to our needs.

Let's begin by diving into it! Firstly, we import essential modules such as **Transformers**, **PyTorch**, and **Accelerate** to empower the fine-tuning process. Then, we utilize the **Trainer()** function from the transformers library to fine-tune the model. For further elaboration, let's delve into the code.

## How To Use This Project
If you wish to use this project, you can do so via **Google Colab** [Here](https://colab.research.google.com/github/AbhijeetKumarThakur2198/Python_Projects/blob/main/Projects_List/Fine-Tune_Hugging_Face_GPT2_Model/FineTuneHuggingFaceGPT2Model.ipynb), or manually on your environment by copying and running the code provided below.

## Fine-Tune Code
Here is the Fine-Tune code:
```python
#####|| HYPERPARAMETERS ||#####
model_name = "gpt2"  # ENTER MODEL FOR FINE TUNE LIKE "gpt2", "gpt2-medium", "gpt2-large" and "gpt2-xl"
file_path = "data.txt"  # ENTER PATH OF YOUR DATA FILE
output_path = "Fine_Tuned_Model"  # ENTER PATH OF WHERE YOU WANT TO SAVE FINE-TUNE MODEL
batch_size = 16  # ENTER BATCH SIZE
num_epochs = 500  # ENTER NUMBER OF EPOCHS
block_size = 124  # ENTER NUMBER OF BLOCK SIZE
learning_rate = 5e-5  # ENTER NUMBER OF LEARNING RATE
save_steps = 10000  # ENTER NUMBER OF SAVE STEPS
overwrite_output_path = True  # ENTER TRUE OR FALSE FOR OVERWRITING OUTPUT PATH
###########################

# IMPORT NECESSARY MODULES
try:
    import re
    import os
except Exception as e:
    print(f"Error: {e}")

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

# DEFINE FUNCTIONS
def read_txt(file_path):
    with open(file_path, "r") as file:
        text = file.read()
    return text

def load_dataset(file_path, tokenizer, block_size):
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size,
    )
    return dataset

def fine_tune(train_file_path, model_name, output_path, overwrite_output_path, per_device_train_batch_size, num_train_epochs, save_steps, learning_rate, block_size):
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    text_data = read_txt(train_file_path)
    text_data = re.sub(r'\n+', '\n', text_data).strip()
    train_dataset = load_dataset(train_file_path, tokenizer, block_size)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    training_args = TrainingArguments(
        output_dir=output_path,
        overwrite_output_dir=overwrite_output_path,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        save_steps=save_steps,
        learning_rate=learning_rate,
        do_train=True
    )

    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
            train_dataset=train_dataset
        )
        trainer.train()
        trainer.save_model()

        print("Your fine-tuning process is completed. Now you can use your fine-tuned model.")
    except Exception as e:
        print(f"Oh! It seems like something went wrong: {str(e)}. Please check all information again or open GitHub issue!")

# FINE TUNE MODEL
fine_tune(
    train_file_path=file_path,
    model_name=model_name,
    output_path=output_path,
    overwrite_output_path=overwrite_output_path,
    per_device_train_batch_size=batch_size,
    num_train_epochs=num_epochs,
    save_steps=save_steps,
    learning_rate=learning_rate,
    block_size=block_size
)
```

## Inference Code
Here is the inference code to test our fine-tuned model:
```python
#######|| CONFIGURATION ||#######
max_length = 50  # SET MAX LENGTH FOR GENERATED TEXT
temperature = 0.7  # SET TEMPERATURE FOR MODEL
return_tensors = "pt"  # SET RETURN TENSORS 
num_return_sequences = 1  # SET NUMBER OF RETURN SEQUENCES
skip_special_tokens = True  # SET TRUE OR FALSE FOR SKIP SPECIAL TOKENS
seed_text = "Once upon a time "  # SET TEXT FOR GENERATION
Fine_Tuned_Model_Path = "/content/"  # SET PATH OF YOUR FINE-TUNED MODEL
############################

# IMPORT NECCESARY MODULES
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
```

## How To Report Problem
If you encounter any errors or bugs in the code, please create an issue Here.

## License
This project is under the [MIT License](https://github.com/AbhijeetKumarThakur2198/Python_Projects/tree/main/Projects_List/LICENSE.md) - see the [LICENSE.md](https://github.com/AbhijeetKumarThakur2198/Python_Projects/tree/main/Projects_List/LICENSE.md) file for details.

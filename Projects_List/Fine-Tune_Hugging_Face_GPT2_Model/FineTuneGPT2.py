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

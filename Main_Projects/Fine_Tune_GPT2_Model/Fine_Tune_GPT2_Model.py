# PLEASE DON'T CHANGE THE CODE UNLESS YOU'RE FAMILIAR WITH PROGRAMMING. IT MIGHT CAUSE ERRORS OTHERWISE.
# IMPORT NECESSARY MODULES 
import re
import os
import torch
import transformers
from transformers import Trainer, TrainingArguments
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import TextDataset, DataCollatorForLanguageModeling

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

def train(train_file_path, model_name, output_dir, overwrite_output_dir, per_device_train_batch_size, num_train_epochs, save_steps, learning_rate, block_size):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir)    
    model.save_pretrained(output_dir)

# DATA CLEANING AND PREPROCESSING
    text_data = read_txt(train_file_path)
    text_data = re.sub(r'\n+', '\n', text_data).strip()
    train_dataset = load_dataset(train_file_path, tokenizer, block_size)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
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

# MAIN FUNCTION FOR USER INPUT
if __name__ == "__main__":    
    while True:
        print("REMEMBER BEFORE FINE-TUNE YOU MUST HAVE NECESSARY MODULES LIKE PYTORCH, TRANSFORMERS, ACCELERATE IF NOT PLEASE INSTALL THEM OTHERWISE YOU GET ERROR")
    
        print("Type 1 to start fine-tune process or type 2 for EXIT")
        print("\nMenu:")
        print("1. Fine-tune the model")
        print("2. Exit")

        choice = input("Enter your choice (1 or 2): ")
        if choice == '1':         
            model_name = edit_parameter("model_name_or_path", "gpt2", "Hint: Enter name or path of model. If you have model in your local device give the path of model folder but if you don't have model simply write model name.", input_type=str)
            data_txt = edit_parameter("data_txt", "/home", "Hint: Enter the path of your data.txt file.", input_type=str)        
            output_dir = edit_parameter("output_dir", "/home", "Hint: Enter path where you want to save finetuned model.", input_type=str)
            batch_size = edit_parameter("batch_size", 8, "Hint: `batch_size` defines the number of training examples processed in each iteration. Typically set to 8 for efficiency, but can be adjusted based on available memory. Larger values speed up training but consume more memory. Experiment to find the right balance for your setup.", input_type=int)
            num_epochs = edit_parameter("num_epochs", 100, "Hint: Enter epoches", input_type=int)
            block_size = edit_parameter("block size", 46, "Hint:`block_size` sets the size of input sequence chunks processed by the model. For optimal results, choose a value around 512. It influences the context the model considers during training and generation. Adjust it based on your specific requirements for sequence processing.", input_type=int)
            learning_rate = edit_parameter("learning_rate", 1e-4, "Hint:`learning_rate` determines the step size during model parameter updates. A common starting value is 1e-4. Adjustments may be necessary based on the specific task and model behavior. Experiment to find an optimal learning rate for faster convergence and better performance.", input_type=float)
            save_steps = edit_parameter("save_steps", 5000, "Hint: Enter save steps", input_type=int)
            print("Did you want to train model?\n1) Yes, I want\n2) No, I don't want")
            confirm = int(input("Enter your choice: "))
            if confirm == 1:
                pass
            elif confirm == 2:
                continue
            else:
                print("Please type only 1 or 2")
                
            train(
                train_file_path=data_txt,
                model_name=model_name,
                output_dir=output_dir,
                overwrite_output_dir=False,
                per_device_train_batch_size=batch_size,
                num_train_epochs=num_epochs,
                save_steps=save_steps,
                learning_rate=learning_rate,
                block_size=block_size
            )
            break
            
        elif choice == '2':
            print("You exit the program!")
            break
            
        else:
            print("Please type 1 or 2")

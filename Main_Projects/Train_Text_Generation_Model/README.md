# Main Project 1

## Language Model Training and Conversion

This repository provides a versatile language model training script and an efficient PyTorch to Hugging Face model conversion tool. It is designed to be user-friendly and customizable, allowing easy adaptation to various datasets and model configurations.

### Prerequisites

Before using the scripts, ensure you have the following Python libraries installed:

- `torch`: PyTorch library
- `transformers`: Hugging Face Transformers library
- `tqdm`: Progress bar library

### Manual Setup

Before running the script, make sure to configure the script parameters manually. Open the script and find the `MANNUAL SETUP` section. Adjust the parameters according to your dataset and model requirements.

### Script Execution

To run the script interactively, set `user_input_mode = True`. The script will prompt you to choose between making a PyTorch model or converting a PyTorch model to Hugging Face format.

If `user_input_mode` is set to `False`, the script will execute with predefined parameters.

### Customization

- You can modify script parameters interactively using the prompt.
- The script supports the adjustment of model dimensions, optimizer selection, and data splitting.
- It provides options to edit parameters, view information, and make informed choices during the configuration process.

### Usage Examples

1. **Model Training:**
   - Choose option 1 to make a PyTorch model.
   - Adjust parameters as prompted or keep default values.
   - The script will save model checkpoints during training.

2. **Model Conversion:**
   - Choose option 2 to convert a PyTorch model to Hugging Face.
   - Enter the path to the PyTorch model and configuration files.
   - Specify the path where you want to save the Hugging Face model.

### Note

- The script supports both GPU (`cuda`) and CPU (`cpu`) devices.
- Ensure proper installation of required libraries before running the script.

### Acknowledgments

This script is based on the GPT-2 architecture, utilizing the PyTorch and Hugging Face Transformers libraries.

Feel free to explore, experiment, and adapt this script for your language modeling projects. Happy coding!

# Llama-3.2-1B-Instruct Fine-Tuning with Unsloth
This repository contains a Jupyter Notebook for fine-tuning the Llama-3.2-1B-Instruct model using the Unsloth library. The workflow demonstrates how to prepare a custom dataset, apply the correct chat template, and train the model using HuggingFace’s Trainer API.
------
## Workflow Overview
### 1. Environment Setup
Installs Unsloth, Xformers, and other required packages for efficient training.

### 2. Model Loading
Loads the Llama-3.2-1B-Instruct model and tokenizer using Unsloth.

### 3. Dataset Preparation
Loads a Bhojpuri language dataset from HuggingFace and reformats it into a conversational format suitable for instruction tuning.

### 4. Chat Template Application
Applies the Llama-3 chat template to the tokenizer for proper formatting.

### 5. Dataset Conversion
Converts the conversation data into a HuggingFace Dataset, ensuring compatibility with the SFT Trainer.

### 6. PEFT Model Setup
Configures the model for parameter-efficient fine-tuning (PEFT) using LoRA.

### 7. Training Arguments
Defines training parameters such as batch size, learning rate, and optimizer.

### 8. Trainer Initialization & Training
Initializes the SFTTrainer and starts the training process.

### 9. Model Testing
Tests the finetuned model with a sample Bhojpuri prompt.

## Requirements
- Python 3.8+
- Jupyter Notebook
- Packages: unsloth, xformers, trl, peft, accelerate, bitsandbytes, datasets, pandas, transformers
## Usage
- Clone this repository.
- Open Finetune_002.ipynb in Jupyter Notebook or VS Code.
- Run each cell sequentially to install dependencies, prepare data, train, and test the model.
## Dataset
- Uses the alpaca_data_cleaned_bhojpuri dataset for instruction tuning.
## References
- Unsloth GitHub
- HuggingFace Transformers
- PEFT
- TRL

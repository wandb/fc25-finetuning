import wandb
import json
import pandas as pd
from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    EarlyStoppingCallback
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)

from huggingface_hub import login

#helper
# Load and validate the JSONL dataset
def load_jsonl_dataset(file_path):
    """
    Load a JSONL dataset into a pandas DataFrame.

    Each line is parsed as a JSON object.
    Handles and reports JSON decoding errors without stopping execution.

    Args:
        file_path (str): Path to the JSONL file.

    Returns:
        pd.DataFrame: DataFrame containing the loaded dataset.
    """
    data = []
    error_count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[Error] Line {i}: Could not decode JSON.")
                    print(f"Content: {line}")
                    print(f"Error: {str(e)}\n")
                    error_count += 1
                    continue
    if error_count > 0:
        print(f"\n⚠️ Finished loading with {error_count} decoding errors. Please check your dataset formatting!")
    else:
        print("✅ Successfully loaded dataset with no errors.")
    return pd.DataFrame(data)

# Create a prompt format from each question-answer pair
def create_prompt(row):
    return f"Question: {row['question']}\nAnswer: {row['answer']}"

# Utility function to reload and prepare the dataset when needed
def reload_astros_dataset(artifact_dir, filename="astro_dataset_train.jsonl"):
    """
    Reload the Astros dataset from a downloaded artifact directory.

    Args:
        artifact_dir (str or Path): Path to the artifact directory.
        filename (str): Name of the JSONL dataset file. Defaults to "astro_dataset_train.jsonl".

    Returns:
        tuple: A pandas DataFrame and a Hugging Face Dataset ready for training.
    """
    dataset_path = Path(artifact_dir) / filename
    df = load_jsonl_dataset(str(dataset_path))
    df['text'] = df.apply(create_prompt, axis=1)
    return df, Dataset.from_pandas(df)

def load_and_prepare_dataset(artifact_dir, filename, dataset_type="dataset"):
    """
    Load a JSONL dataset, create prompts, and convert to Hugging Face Dataset format.

    Args:
        artifact_dir (str or Path): Path to the artifact directory
        filename (str): Name of the JSONL file to load
        dataset_type (str): Type of dataset (e.g., "training" or "evaluation")

    Returns:
        tuple: (pandas DataFrame, Hugging Face Dataset)
    """
    print(f"\nLoading {dataset_type} dataset...")
    df = load_jsonl_dataset(str(Path(artifact_dir) / filename))
    df['text'] = df.apply(create_prompt, axis=1)
    hf_dataset = Dataset.from_pandas(df)
    print(f"✅ {dataset_type.capitalize()} dataset loaded with {len(df)} examples")
    return df, hf_dataset

#helper
import shutil
from pathlib import Path

def download_all_models(entity, project):
    models = {
        "falcon-rw-1b": "v0",
        "TinyLlama": "v1",
    }

    for model_name, version in models.items():
        print(f"\n⬇️ Downloading {model_name} (version {version}) from Weights & Biases...")

        run = wandb.init(
            entity=entity,
            project=project,
            job_type="model_retrieval",
            name=f"fetch_{model_name}_model"
        )

        artifact = run.use_artifact(
            f'wandb-registry-model/FC_FT_Workshop_Models:{version}', type='model'
        )
        downloaded_dir = artifact.download()
        run.finish()

        local_path = Path(f"./models/{model_name}_{version}")

        # Clear previous directory if exists
        if local_path.exists():
            shutil.rmtree(local_path)

        shutil.move(downloaded_dir, local_path)

        print(f"✅ Model saved to: {local_path}")
        print(f"✅ Model downloaded successfully!")

def get_models_from_wandb(entity, project):
    download_all_models(entity, project)
    model, tokenizer, model_name = load_model_and_tokenizer()
    return model, tokenizer, model_name

#helper
def select_model():
    """
    Display a menu of available models and let the user select one.

    Returns:
        tuple: (model_name, model_version)
    """
    models = {
        "1": ("falcon-rw-1b", "v0"),
        "2": ("TinyLlama", "v1"),  # Changed to v2 to match actual download
    }
    print("\nAvailable Models:")
    print("1. Falcon RW 1B")
    print("2. TinyLlama 1B")
    
    while True:
        choice = input("\nSelect a model (1-2): ").strip()
        if choice in models:
            model_name, version = models[choice]
            print(f"\n✅ Selected: {model_name}")
            return model_name, version
        print("Invalid choice. Please select 1 or 2.")

def load_model_and_tokenizer():
    """
    Load the selected model and tokenizer from local cache with QLoRA configuration.

    Returns:
        tuple: (model, tokenizer, model_name)
    """
    model_name, version = select_model()
    model_dir = Path(f"./models/{model_name}_{version}")

    print(f"\n📦 Loading model from: {model_dir}")

    # Step 1: Load tokenizer
    print("\nStep 1: Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    print("✅ Tokenizer loaded successfully!")

    # Step 2: Configure 4-bit quantization
    print("\nStep 2: Configuring QLoRA...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Step 3: Load model with quantization
    print("Loading model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    if model_name == "falcon-rw-1b":
        targets = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]

    if model_name == "TinyLlama":
        targets = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

    # Step 4: Configure LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=targets,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print("✅ Model loaded with QLoRA successfully!")
    #run {animate_dir}/celebrate_model.py
    celebrate_model(model.base_model.model.__class__.__name__)
    return model, tokenizer, model_name

#helper
# Set pad token and pad token ID if missing (important for consistent model behavior)
#if tokenizer.pad_token is None:
#    tokenizer.pad_token = tokenizer.eos_token
#    model.config.pad_token_id = model.config.eos_token_id

# Function to tokenize input prompts for training
def tokenize_function(examples):
    """
    Tokenize the text prompts with padding and truncation.

    Args:
        examples (dict): A batch of examples with 'text' field.

    Returns:
        dict: Tokenized outputs (input_ids, attention_mask, etc.)
    """
    tokenized = tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=512,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# Will be adding a few metric functions users can use to test out out the model - TODO: Cleanup , either remove or add addititions functions - not currently used
def compute_perpexity(eval_preds):
    """
    Compute perplexity metric from model evaluation logits.

    Args:
        eval_preds (tuple): Tuple of (logits, labels) from evaluation step.

    Returns:
        dict: Dictionary containing 'perplexity' score.
    """
    logits, labels = eval_preds
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)

    # Shift logits and labels for causal language modeling
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # Compute cross-entropy loss
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    perplexity = math.exp(loss.item()) if loss.item() < 100 else float("inf")
    return {"perplexity": perplexity}

def tokenized_train_test(training_dataset, split):
  # Apply the tokenizer to the datasets
    split_dataset = training_dataset.train_test_split(test_size=split)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=eval_dataset.column_names,
    )

    print("✅ Tokenization applied to Training & Evaluation Datasets successfully!")

    return train_dataset, eval_dataset


#helper
def load_finetuned_model(adapter_dir, base_model_dir):

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)

    # Load base model

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model with quantization
    print("Loading model with 4-bit quantization...")
    base_model = AutoModelForCausalLM.from_pretrained(
        Path(base_model_dir),
        quantization_config=bnb_config,
        device_map="auto",
        )
    # Load fine-tuned adapter
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    return tokenizer, model
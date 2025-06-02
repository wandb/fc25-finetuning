import wandb
import json
import pandas as pd
from datasets import Dataset
import torch

import boto3
import os

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
import os

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


def download_model(entity, project, model_name, version, run_id):
    """
    Downloads a specific model from W&B given its name and version.
    """
    print(f"\n⬇️ Downloading {model_name} (version {version}) from Weights & Biases...")

    run = wandb.init(
        entity=entity,
        project=project,
        job_type="model_retrieval",
        id=run_id
    )

    artifact = run.use_artifact(
        f'wandb-registry-model/FC_FT_Workshop_Models:{version}', type='model'
    )
    downloaded_dir = artifact.download()
    run.finish()

    local_path = Path(f"./models/{model_name}_{version}")
    
    if local_path.exists():
        shutil.rmtree(local_path)

    shutil.move(downloaded_dir, local_path)

    print(f"✅ Model saved to: {local_path}")
    print(f"✅ Model downloaded successfully!")

def get_model_from_wandb(entity, project, run_id):
    """
    Select and download one model from W&B, then load it.
    """
    model_name, version = select_model()
    download_model(entity, project, model_name, version, run_id)
    return model_name, version


# Updated load_model_and_tokenizer to accept model_name and version
def load_model_and_tokenizer(model_name, version):
    """
    Load model and tokenizer for selected model/version.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
        trust_remote_code=True, 
    )

    model = prepare_model_for_kbit_training(model)

    # Target modules
    if model_name == "falcon-rw-1b":
        targets = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    elif model_name == "TinyLlama":
        targets = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    else:
        raise ValueError(f"Unknown model name: {model_name}")

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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id    

    model.print_trainable_parameters()

    print("✅ Model loaded with QLoRA successfully!")

    return model, tokenizer, model_name

# Function to tokenize input prompts for training
def get_tokenize_function(tokenizer):
    def tokenize_function(examples):

        texts_with_eos = [text + tokenizer.eos_token for text in examples["text"]]

        tokenized = tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=512
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    return tokenize_function

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

def tokenized_train_test(training_dataset, split, tokenizer):

    split_dataset = training_dataset.train_test_split(test_size=split)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    # Create tokenizer function with tokenizer bound
    tokenize = get_tokenize_function(tokenizer)

    train_dataset = train_dataset.map(
        tokenize,
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    eval_dataset = eval_dataset.map(
        tokenize,
        batched=True,
        remove_columns=eval_dataset.column_names,
    )

    print("✅ Tokenization applied to Training & Evaluation Datasets successfully!")
    return train_dataset, eval_dataset


def load_finetuned_model(adapter_dir, base_model_dir):

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)

    # Load model with quantization
    print("Loading model with 4-bit quantization...")
    base_model = AutoModelForCausalLM.from_pretrained(
        Path(base_model_dir),
        device_map="auto",
        )
    # Load fine-tuned adapter
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    print("✅ Loaded model with finetuned adapter successfully!")
    return tokenizer, model


def bedrock_access():
    session = boto3.Session()
    credentials = session.get_credentials().get_frozen_credentials()
    
    os.environ["AWS_ACCESS_KEY_ID"] = credentials.access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = credentials.secret_key
    os.environ["AWS_SESSION_TOKEN"] = credentials.token  # Important for temp credentials!
    os.environ["AWS_REGION"] = session.region_name or "us-west-2"
    
    print("✅ AWS credentials and region set in environment variables.")    

def create_sagemaker_endpoint():
    import sagemaker
    import boto3
    from sagemaker import get_execution_role
    from pathlib import Path
    import os
    from sagemaker.s3 import S3Uploader
    from sagemaker.huggingface import get_huggingface_llm_image_uri
    import json
    from sagemaker.huggingface import HuggingFaceModel
    import subprocess
    
    boto_session = bedrock_access()
    
    sess = sagemaker.Session(boto_session=boto_session)
    # sagemaker session bucket -> used for uploading data, models and logs
    # sagemaker will automatically create this bucket if it not exists
    sagemaker_session_bucket=None
    if sagemaker_session_bucket is None and sess is not None:
        # set to default bucket if a bucket name is not given
        sagemaker_session_bucket = sess.default_bucket()
    
    sess = sagemaker.Session(default_bucket=sagemaker_session_bucket, boto_session=boto_session)
    print(f"sagemaker session region: {sess.boto_region_name}")

    cmd = [
        "tar",
        "-cf", "model.tar.gz",
        "--use-compress-program=pigz",
        "-C", "merged_model_llama",
        "."
    ]
    subprocess.run(cmd, check=True)
    
    # upload model.tar.gz to s3
    s3_model_uri = S3Uploader.upload(local_path=str(Path("model.tar.gz")), desired_s3_uri=f"s3://{sess.default_bucket()}/ft-model", sagemaker_session=sess)
     
    print(f"model uploaded to: {s3_model_uri}")
     
    # retrieve the llm image uri
    llm_image = get_huggingface_llm_image_uri(
        backend="huggingface", 
        region=sess.boto_region_name
    )
     
    # print ecr image uri
    print(f"llm image uri: {llm_image}")
     
    # sagemaker config
    instance_type = "ml.g5.xlarge"
    number_of_gpu = 1
    health_check_timeout = 300
    
    role = get_execution_role()
    
    boto3.client("iam").attach_role_policy(
        RoleName=role.split("role/")[1],
        PolicyArn="arn:aws:iam::aws:policy/AdministratorAccess"
    )
     
    # create HuggingFaceModel
    llm_model = HuggingFaceModel(
        role = get_execution_role(),
        image_uri=llm_image,
        model_data=s3_model_uri,
        transformers_version="4.52.4",
        pytorch_version="2.6.0",
        py_version="py310",
        env={
          'HF_MODEL_ID': "/opt/ml/model", # path to where sagemaker stores the model
          'SM_NUM_GPUS': json.dumps(number_of_gpu), # Number of GPU used per replica
          'MAX_INPUT_LENGTH': json.dumps(512), # Max length of input text
          'MAX_TOTAL_TOKENS': json.dumps(2048), # Max length of the generation (including input text)
          'MAX_BATCH_TOTAL_TOKENS': json.dumps(8192),
        },
    )
    
    # Deploy model to an endpoint
    # https://sagemaker.readthedocs.io/en/stable/api/inference/model.html#sagemaker.model.Model.deploy
    llm = llm_model.deploy(
      initial_instance_count=1,
      instance_type=instance_type,
      # volume_size=400, # If using an instance with local SSD storage, volume_size must be None, e.g. p4 but not p3
      container_startup_health_check_timeout=health_check_timeout, # 10 minutes to be able to load the model
    )
    
    print(f"Deployment completed. \n This is your endpoint name! Submit this on the quest page to get points\n {llm.endpoint_name}")
    return llm
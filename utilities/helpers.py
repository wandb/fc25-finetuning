import wandb
import json
import pandas as pd
from datasets import Dataset
import torch

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

from IPython.display import display, HTML

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


def download_model(entity, project, model_name, version):
    """
    Downloads a specific model from W&B given its name and version.
    """
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
    
    if local_path.exists():
        shutil.rmtree(local_path)

    shutil.move(downloaded_dir, local_path)

    print(f"✅ Model saved to: {local_path}")
    print(f"✅ Model downloaded successfully!")

def get_model_from_wandb(entity, project):
    """
    Select and download one model from W&B, then load it.
    """
    model_name, version = select_model()
    download_model(entity, project, model_name, version)
    model, tokenizer, model_name = load_model_and_tokenizer(model_name, version)
    celebrate_model(model.base_model.model.__class__.__name__)
    return model, tokenizer, model_name

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
        trust_remote_code=True
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
    model.print_trainable_parameters()

    print("✅ Model loaded with QLoRA successfully!")

    return model, tokenizer, model_name

# Function to tokenize input prompts for training
def get_tokenize_function(tokenizer):
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=512,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    return tokenize_function

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

#animation
def celebrate_model(model_class):
    html = f"""
    <div id="celebration" style="display: none; text-align: center; padding: 50px; font-family: 'Segoe UI', sans-serif;">
        <div style="
            font-size: 20px;
            color: #999;
            margin-bottom: 10px;
            letter-spacing: 1px;
        ">
            Mission Update
        </div>
        <div style="
            font-size: 40px;
            font-weight: bold;
            color: #fac13c;
            margin-bottom: 20px;
        ">
            Your model is ready for staging 🚀
        </div>
        <div style="
            font-size: 28px;
            color: #ffffff;
            background: #1a1a1a;
            display: inline-block;
            padding: 10px 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px #fac13c;
        ">
            {model_class}
        </div>
    </div>

    <script>
    setTimeout(function() {{
        document.getElementById('celebration').style.display = 'block';
    }}, 1000);
    </script>
    """
    display(HTML(html))

#animation
def launch_sequence(training_args=None, num_columns=2):
    """
    Displays an epic launch animation.
    Optionally shows training arguments inside the mission control panel.
    """
    # Base HTML template with a placeholder for dynamic Training Args
    HTML_TEMPLATE = r"""
    <div id="launch-container" style="position: relative; width: 100%; height: 100vh; background: radial-gradient(ellipse at bottom, #000 0%, #020111 100%); overflow: auto; font-family: 'Comic Sans MS', cursive, sans-serif; display: flex; align-items: center; justify-content: center;">

    <style>
      /* --- Same CSS as before --- */
      #launch-container .stars {
        position: absolute;
        width: 100%;
        height: 100%;
        overflow: hidden;
        z-index: 1;
      }
      #launch-container .star {
        position: absolute;
        width: 2px;
        height: 2px;
        background: white;
        border-radius: 50%;
        opacity: 0.7;
        animation: moveStar 5s linear infinite;
      }
      @keyframes moveStar {
        0% { transform: translateY(0); }
        100% { transform: translateY(600px); }
      }
      #launch-container .countdown {
        position: absolute;
        width: 100%;
        top: 30%;
        font-size: 4em;
        text-align: center;
        z-index: 2;
        color: #fff;
        text-shadow: 0 0 10px #0ff;
      }
      #launch-container .rocket {
          position: absolute; /* <-- must stay absolute */
          bottom: 20px;
          left: 50%;
          transform: translateX(-50%);
          font-size: 80px;
          z-index: 3;
          transition: transform 3s ease-out, bottom 3s ease-out;
      }
      #launch-container .flame {
          position: absolute; /* <-- stays absolute INSIDE rocket */
          top: 100%; /* start exactly under the rocket */
          left: 50%;
          transform: translateX(-50%) translateY(0);
          width: 20px;
          height: 40px;
          background: radial-gradient(circle, orange 0%, red 100%);
          border-radius: 50%;
          animation: flameFlicker 0.2s infinite alternate;
          opacity: 0;
          z-index: 1;
      }
      @keyframes flameFlicker {
        0% { transform: translateX(-50%) scaleY(2); }
        100% { transform: translateX(-50%) scaleY(1.5); }
      }
      #launch-container #control-panel {
        display: none;
        position: relative;
        background: rgba(0,0,0,0.8);
        padding: 30px;
        border: 2px solid #0ff;
        border-radius: 10px;
        width: 90%;
        max-width: 800px;
        z-index: 5;
        text-align: center;
        box-sizing: border-box;
        overflow: auto;
        margin: 0 auto;
        color: white;
      }
      #launch-container #control-panel table {
        margin: 10px auto 20px auto;
      }
      #launch-container #control-panel h3 {
        margin-top: 0;
        margin-bottom: 20px;
        font-size: 1.8em;
        text-shadow: 0 0 5px #0ff;
      }
      #launch-container .execute-text {
        margin: 20px 0;
        font-size: 1.2em;
        color: #0ff;
      }
      #launch-container .confetti {
        position: absolute;
        width: 8px;
        height: 8px;
        background: hsl(var(--hue), 70%, 60%);
        animation: fall 3s ease-out forwards;
        z-index: 4;
      }
      @keyframes fall {
        to {
          transform: translateY(600px) rotate(720deg);
          opacity: 0;
        }
      }
    </style>

    <div class="stars" id="stars"></div>
    <div class="countdown" id="countdown">Preparing...</div>
    <div class="rocket" id="rocket">
        🛸
        <div class="flame" id="flame"></div>
    </div>


    <div id="control-panel">
      <h3>🚀 Fine-Tuning Mission Control</h3>
      <!--ARGS_PLACEHOLDER-->
      <div class="execute-text">Fly Me To The Moon</div>
    </div>

    <script>
    // --- Main Launch Animation Setup ---
    (function() {
      const container = document.getElementById('launch-container');
      const stars = document.getElementById('stars');
      const countdownEl = document.getElementById('countdown');
      const rocket = document.getElementById('rocket');
      const flame = document.getElementById('flame');
      const panel = document.getElementById('control-panel');

      // Create stars
      for (let i = 0; i < 150; i++) {
        let star = document.createElement('div');
        star.className = 'star';
        star.style.top = Math.random() * 100 + '%';
        star.style.left = Math.random() * 100 + '%';
        star.style.animationDuration = (2 + Math.random() * 3) + 's';
        stars.appendChild(star);
      }

      // Countdown and launch
      let countdown = 3;
      countdownEl.innerText = countdown;
      const interval = setInterval(() => {
        countdown--;
        if (countdown > 0) {
          countdownEl.innerText = countdown;
        } else if (countdown === 0) {
          countdownEl.innerText = "Liftoff!";
          rocket.style.bottom = '600px';
          rocket.style.transform = 'translateX(-50%) translateY(-200px)';
          flame.style.opacity = 1;
          setTimeout(() => {
            rocket.style.display = 'none';
            flame.style.display = 'none';
            countdownEl.style.display = 'none';
            panel.style.display = 'block';
            releaseConfetti();
          }, 3000);
          clearInterval(interval);
        }
      }, 1000);

      // Confetti
      function releaseConfetti() {
        for (let i = 0; i < 50; i++) {
          let c = document.createElement('div');
          c.className = 'confetti';
          c.style.left = Math.random() * 100 + '%';
          c.style.top = '-10px';
          c.style.setProperty('--hue', Math.random() * 360);
          container.appendChild(c);
          setTimeout(() => c.remove(), 3000);
        }
      }
    })();
    </script>


    </div>
    """

    args_html = ""
    if training_args:
        # Create a default instance with just required params
        default_args = TrainingArguments(output_dir="./results")
        
        # Find only the arguments that differ from defaults
        changed_args = {}
        for k, v in vars(training_args).items():
            # Only include if it's different from the default
            if k in vars(default_args) and v != getattr(default_args, k):
                changed_args[k] = v
        
        # Now build HTML from the changed arguments
        items = []
        for key, value in changed_args.items():
            if isinstance(value, bool):
                value = "✅" if value else "❌"
            items.append(f"<strong>{key}:</strong> {value}")

        # Pad the list so it divides evenly into columns
        while len(items) % num_columns != 0:
            items.append("")  # empty cell

        # Split into rows
        rows = [items[i:i+num_columns] for i in range(0, len(items), num_columns)]

        # Build the table
        args_html = "<table style='width:100%; text-align:left; border-collapse:separate; border-spacing: 10px;'>"
        for row in rows:
            args_html += "<tr>"
            for cell in row:
                args_html += f"<td style='vertical-align:top; padding: 4px 12px;'>{cell}</td>"
            args_html += "</tr>"
        args_html += "</table>"
    else:
        args_html = "<div>No mission parameters provided.</div>"

    final_html = HTML_TEMPLATE.replace("<!--ARGS_PLACEHOLDER-->", args_html)
    display(HTML(final_html))

def split_args_into_columns(args_list, num_columns=2):
    # Split the list into num_columns nearly equal parts
    avg = len(args_list) // num_columns
    columns = []
    for i in range(num_columns):
        start = i * avg
        # Last column takes the remainder
        end = (i + 1) * avg if i < num_columns - 1 else len(args_list)
        columns.append(args_list[start:end])
    return columns

"""
Fine-tune Gemma 3-4B for Hindi Clause Boundary Detection
=========================================================

This script fine-tunes Gemma 3-4B model on annotated Hindi clause boundary data.
Approach: Seq2Seq - Input: raw sentence → Output: annotated sentence

Usage:
    python finetune_gemma_clause_boundary.py
"""
import sys
import os
import re
import torch
from datetime import datetime
from collections import Counter
from sklearn.model_selection import train_test_split


# ============================================================================
# LOGGING SETUP
# ============================================================================

class TeeLogger:
    """Write output to both console and log file simultaneously."""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Ensure immediate write

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def setup_logging(output_dir):
    """Setup logging to save console output to file."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"training_log_{timestamp}.txt")

    # Redirect stdout to both console and file
    tee = TeeLogger(log_file)
    sys.stdout = tee

    print("="*80)
    print(f"LOG FILE: {log_file}")
    print("="*80)
    print()

    return tee, log_file
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    # EarlyStoppingCallback
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Paths - UPDATE THESE
    DATA_PATH = sys.argv[1]  # Path to annotated file (e.g., "S_T_Nor_0013_300_ha_sentencewise.txt")
    OUTPUT_DIR = sys.argv[2]  # Directory to save fine-tuned model (e.g., "./gemma-clause-boundary-finetuned")
    HF_TOKEN = "HF_TOKEN"  # Hugging Face token for accessing Gemma

    # Model
    MODEL_NAME = "google/gemma-3-4b-it"

    # Training splits
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    # LoRA config (efficient fine-tuning)
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

    # Training hyperparameters
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 16
    NUM_EPOCHS = 3
    LEARNING_RATE = 2e-4
    MAX_SEQ_LENGTH = 512
    WARMUP_STEPS = 100
    SAVE_STEPS = 10
    EVAL_STEPS = 10
    LOGGING_STEPS = 10

    # Early stopping
    # EARLY_STOPPING_PATIENCE = 3  # Stop if no improvement for 3 evaluations

    # Random seed
    SEED = 42


# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def load_annotated_data(file_path):
    """Load and parse annotated clause boundary data."""
    print(f"Loading data from: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    annotated_sentences = []
    raw_sentences = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Remove line numbers (format: "  123|sentence")
        if '|' in line:
            line = line.split('|', 1)[1]

        annotated_sentences.append(line)

        # Extract raw sentence (remove all clause tags)
        raw = re.sub(r'(MCL|RCL|RP|NF|INF|COM|COND|ADVCL)\[', '', line)
        raw = re.sub(r'\](MCL|RCL|RP|NF|INF|COM|COND|ADVCL)', '', raw)
        raw = re.sub(r'\s+', ' ', raw)
        raw_sentences.append(raw.strip())

    print(f"✓ Loaded {len(raw_sentences)} sentence pairs")
    return raw_sentences, annotated_sentences


def get_stratification_labels(annotated_sentences):
    """
    Create stratification labels based on clause types present.
    Returns label for each sentence (e.g., "MCL", "MCL+RCL", "MCL+COND")
    """
    labels = []

    for sentence in annotated_sentences:
        # Extract all clause types present
        clause_types = set()
        for tag in ['MCL', 'RCL', 'RP', 'NF', 'INF', 'COM', 'COND', 'ADVCL']:
            if f'{tag}[' in sentence:
                clause_types.add(tag)

        # Create label (sorted for consistency)
        label = '+'.join(sorted(clause_types)) if clause_types else 'NONE'
        labels.append(label)

    return labels

import numpy as np
from collections import Counter

def safe_stratify(labels):
    counts = Counter(labels)
    return np.array([
        lbl if counts[lbl] >= 2 else "RARE_CLASS"
        for lbl in labels
    ])

def create_train_val_test_split(raw_sentences, annotated_sentences, config):
    """Split data into train, validation, and test sets with stratification."""

    # Get stratification labels
    strat_labels = get_stratification_labels(annotated_sentences)

    # Count label distribution
    label_counts = Counter(strat_labels)
    print(f"\n📊 Label distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {label}: {count}")

    # Handle rare labels (< 3 occurrences) - can't stratify with train_test_split
    min_samples = 3
    rare_labels = {label for label, count in label_counts.items() if count < min_samples}

    if rare_labels:
        print(f"\n⚠️  Warning: {len(rare_labels)} label combinations have < {min_samples} samples")
        print("  These will be handled specially to avoid stratification errors")

        # Separate rare and common samples
        rare_indices = [i for i, label in enumerate(strat_labels) if label in rare_labels]
        common_indices = [i for i, label in enumerate(strat_labels) if label not in rare_labels]

        # Stratified split on common samples
        common_raw = [raw_sentences[i] for i in common_indices]
        common_ann = [annotated_sentences[i] for i in common_indices]
        common_labels = [strat_labels[i] for i in common_indices]

        # Split common samples
        train_raw, temp_raw, train_ann, temp_ann = train_test_split(
            common_raw, common_ann,
            test_size=(config.VAL_RATIO + config.TEST_RATIO),
            stratify=common_labels,
            random_state=config.SEED
        )
        strat_labels = safe_stratify(temp_ann)
        temp_labels = get_stratification_labels(strat_labels)
        val_raw, test_raw, val_ann, test_ann = train_test_split(
            temp_raw, temp_ann,
            test_size=config.TEST_RATIO / (config.VAL_RATIO + config.TEST_RATIO),
            stratify=temp_labels,
            random_state=config.SEED
        )

        # Add rare samples to train (to ensure model sees them)
        rare_raw = [raw_sentences[i] for i in rare_indices]
        rare_ann = [annotated_sentences[i] for i in rare_indices]
        train_raw.extend(rare_raw)
        train_ann.extend(rare_ann)

    else:
        # All labels have enough samples - normal stratified split
        train_raw, temp_raw, train_ann, temp_ann = train_test_split(
            raw_sentences, annotated_sentences,
            test_size=(config.VAL_RATIO + config.TEST_RATIO),
            stratify=strat_labels,
            random_state=config.SEED
        )

        temp_labels = get_stratification_labels(temp_ann)
        val_raw, test_raw, val_ann, test_ann = train_test_split(
            temp_raw, temp_ann,
            test_size=config.TEST_RATIO / (config.VAL_RATIO + config.TEST_RATIO),
            stratify=temp_labels,
            random_state=config.SEED
        )

    # Create data pairs
    train_data = list(zip(train_raw, train_ann))
    val_data = list(zip(val_raw, val_ann))
    test_data = list(zip(test_raw, test_ann))

    print(f"\n✓ Stratified split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

    # Verify distribution in each split
    print(f"\n📊 Tag distribution per split:")
    for split_name, data in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
        split_labels = get_stratification_labels([ann for _, ann in data])
        split_counts = Counter(split_labels)
        print(f"  {split_name}: {len(split_counts)} unique label combinations")

    return train_data, val_data, test_data


def save_splits_to_files(train_data, val_data, test_data, output_dir):
    """Save train/val/test splits to text files for reproducibility."""
    import os

    # Create train and test directories
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Save train and val to train/ directory
    for split_name, data in [('train', train_data), ('val', val_data)]:
        ann_file = os.path.join(train_dir, f"{split_name}_annotated.txt")
        with open(ann_file, 'w', encoding='utf-8') as f:
            for i, (raw, ann) in enumerate(data, 1):
                f.write(f"{i:6d}|{ann}\n")
        print(f"  ✓ {split_name.capitalize()}: {ann_file}")

    # Save test to test/ directory
    ann_file = os.path.join(test_dir, "test_annotated.txt")
    with open(ann_file, 'w', encoding='utf-8') as f:
        for i, (raw, ann) in enumerate(test_data, 1):
            f.write(f"{i:6d}|{ann}\n")
    print(f"  ✓ Test: {ann_file}")

    print(f"\n✓ All splits saved to: {output_dir}")


def format_instruction(raw_sentence):
    """Create instruction prompt for the model."""
    instruction = """Annotate the following Hindi sentence with clause boundaries.

CLAUSE TYPES:
- MCL (Main Clause)
- RCL (Relative Clause)
- RP (Relative Participle)
- COND (Conditional)
- NF (Non-Finite)
- INF (Infinitive)
- COM (Complementizer)
- ADVCL (Adverbial)

FORMAT: TYPE[ text ]TYPE (nesting allowed)

Input: {input}
Output:"""

    return instruction.format(input=raw_sentence)


def create_dataset(data_pairs, tokenizer, config):
    """Convert sentence pairs to tokenized dataset."""
    inputs = []
    outputs = []

    for raw, annotated in data_pairs:
        # Format as instruction following
        prompt = format_instruction(raw)

        # Create full text with chat template
        messages = [{"role": "user", "content": prompt}]
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_text = full_prompt + annotated + tokenizer.eos_token

        inputs.append(full_text)

    # Tokenize
    tokenized = tokenizer(
        inputs,
        truncation=True,
        max_length=config.MAX_SEQ_LENGTH,
        padding=False
    )

    # Create dataset
    dataset = Dataset.from_dict({
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"]
    })

    return dataset


# ============================================================================
# MODEL SETUP
# ============================================================================

def setup_model_and_tokenizer(config):
    """Load and prepare model with LoRA for efficient fine-tuning."""
    print(f"Loading model: {config.MODEL_NAME}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_NAME,
        token=config.HF_TOKEN
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    from transformers import BitsAndBytesConfig

    # 4-bit config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model with 4-bit
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        quantization_config=bnb_config,  # Add this
        device_map="auto",
        token=config.HF_TOKEN
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        target_modules=config.LORA_TARGET_MODULES,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("✓ Model loaded with LoRA adapters")
    return model, tokenizer


# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, tokenizer, train_dataset, val_dataset, config):
    """Fine-tune the model."""
    print("\n" + "="*80)
    print("TRAINING CONFIGURATION")
    print("="*80)
    print(f"Epochs: {config.NUM_EPOCHS}")
    print(f"Batch size: {config.BATCH_SIZE} (effective: {config.BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS})")
    print(f"Learning rate: {config.LEARNING_RATE}")
    # print(f"Early stopping: Enabled (patience={config.EARLY_STOPPING_PATIENCE})")
    print(f"Save best model: Enabled (metric=eval_loss)")
    print("="*80)
    print("\nStarting training...")

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.LEARNING_RATE,
        warmup_steps=config.WARMUP_STEPS,
        logging_steps=config.LOGGING_STEPS,
        save_steps=config.SAVE_STEPS,
        eval_steps=config.EVAL_STEPS,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,  # Lower loss is better
        fp16=False,
        bf16=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        save_total_limit=3,
        report_to="none",
        seed=config.SEED
    )

    # Early stopping callback
    # early_stopping = EarlyStoppingCallback(
    #     early_stopping_patience=config.EARLY_STOPPING_PATIENCE
    # )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        # callbacks=[early_stopping]
    )

    # Train
    result = trainer.train()

    # Check if early stopping was triggered
    total_steps = result.global_step
    max_steps = (len(train_dataset) // (config.BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS)) * config.NUM_EPOCHS

    print("\n" + "="*80)
    if total_steps < max_steps:
        print("🛑 Early stopping triggered! Best model loaded.")
    else:
        print("✅ Training completed all epochs. Best model loaded.")
    print(f"Training steps: {total_steps}")
    print(f"Final train loss: {result.training_loss:.4f}")
    print("="*80)

    # Save best model (already loaded due to load_best_model_at_end=True)
    best_model_path = os.path.join(config.OUTPUT_DIR, "best_model")
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)

    print(f"✓ Best model saved to: {best_model_path}")
    return trainer


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, tokenizer, test_data, config):
    """Evaluate model on test set."""
    print("\nEvaluating on test set...")

    model.eval()
    predictions = []

    for raw, gold in test_data:
        # Generate prediction
        prompt = format_instruction(raw)
        messages = [{"role": "user", "content": prompt}]
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.1,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        prediction = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        predictions.append(prediction.strip())

    # Calculate metrics
    exact_matches = sum(1 for p, (_, g) in zip(predictions, test_data) if p == g)
    accuracy = (exact_matches / len(test_data)) * 100

    print(f"\n{'='*80}")
    print(f"TEST SET RESULTS")
    print(f"{'='*80}")
    print(f"Exact Match Accuracy: {accuracy:.2f}%")
    print(f"Correct: {exact_matches}/{len(test_data)}")

    # Save predictions
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(config.OUTPUT_DIR, f"test_predictions_{timestamp}.txt")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("FINE-TUNED GEMMA TEST PREDICTIONS\n")
        f.write("=" * 80 + "\n\n")

        for i, (pred, (raw, gold)) in enumerate(zip(predictions, test_data), 1):
            f.write(f"[{i}]\n")
            f.write(f"RAW:  {raw}\n")
            f.write(f"PRED: {pred}\n")
            f.write(f"GOLD: {gold}\n")
            f.write(f"MATCH: {'YES' if pred == gold else 'NO'}\n\n")

    print(f"✓ Predictions saved to: {output_file}")
    return accuracy


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main training pipeline."""
    config = Config()

    # Validate paths
    if not config.DATA_PATH:
        raise ValueError("Please set DATA_PATH in Config")
    if not config.OUTPUT_DIR:
        raise ValueError("Please set OUTPUT_DIR in Config")
    if not config.HF_TOKEN:
        raise ValueError("Please set HF_TOKEN in Config")

    # Setup logging to save console output
    tee_logger, log_file = setup_logging(config.OUTPUT_DIR)

    try:
        print("="*80)
        print("GEMMA 3-4B CLAUSE BOUNDARY FINE-TUNING")
        print("="*80)

        # Load data
        raw_sentences, annotated_sentences = load_annotated_data(config.DATA_PATH)

        # Split data
        train_data, val_data, test_data = create_train_val_test_split(
            raw_sentences, annotated_sentences, config
        )

        # Save splits to files
        print("\nSaving data splits...")
        save_splits_to_files(train_data, val_data, test_data, config.OUTPUT_DIR)

        # Setup model
        model, tokenizer = setup_model_and_tokenizer(config)

        # Create datasets
        print("\nPreparing datasets...")
        train_dataset = create_dataset(train_data, tokenizer, config)
        val_dataset = create_dataset(val_data, tokenizer, config)

        # Train
        trainer = train_model(model, tokenizer, train_dataset, val_dataset, config)

        # Evaluate
        accuracy = evaluate_model(model, tokenizer, test_data, config)

        print("\n" + "="*80)
        print("TRAINING COMPLETE!")
        print("="*80)
        print(f"Best model saved to: {config.OUTPUT_DIR}/best_model")
        print(f"Test accuracy: {accuracy:.2f}%")
        print(f"\nNote: The 'best_model' is based on lowest validation loss during training.")
        print(f"Log file saved to: {log_file}")
        print("="*80)

    except Exception as e:
        print("\n" + "="*80)
        print("ERROR OCCURRED!")
        print("="*80)
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        print("="*80)
        raise

    finally:
        # Always close log file and restore stdout
        tee_logger.close()
        sys.stdout = tee_logger.terminal


if __name__ == "__main__":
    main()

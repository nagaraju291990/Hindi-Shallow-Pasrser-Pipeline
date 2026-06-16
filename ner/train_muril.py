# -*- coding: utf-8 -*-
"""
Hindi NER Training Script - MuRIL Model
Model: google/muril-base-cased

Usage:
  From raw data:  python train_hindi_muril.py --data_path ./data --output_dir ./hindi-ner-muril
  From JSON splits (if pre-generated):
                  python train_hindi_muril.py --splits_path . --output_dir ./hindi-ner-muril

Data expected at: ./data/Hindi_NER_Set1/ ... ./data/Hindi_NER_Set8/
Each .conllu / .txt file format:
  <sentence text line>          <- skipped
  token1<TAB>TAG1
  token2<TAB>TAG2
  ...
  <blank line>                  <- sentence separator
"""

import os
import json
import argparse
import numpy as np
from collections import Counter
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import torch

# ============== Configuration ==============
MODEL_NAME = "google/muril-base-cased"
DEFAULT_OUTPUT_DIR = "./hindi-ner-muril"

# Hindi NER Set folder names
HINDI_NER_SETS = [
    'Hindi_NER_Set1', 'Hindi_NER_Set2', 'Hindi_NER_Set3', 'Hindi_NER_Set4',
    'Hindi_NER_Set5', 'Hindi_NER_Set6', 'Hindi_NER_Set7', 'Hindi_NER_Set8'
]

# ============== Load from JSON splits ==============
def load_json_splits(base_path="."):
    """Load pre-split data from JSON files (if prepare_splits.py was run)."""
    train_path = os.path.join(base_path, "train.json")
    val_path   = os.path.join(base_path, "validation.json")
    test_path  = os.path.join(base_path, "test.json")

    if not all(os.path.exists(p) for p in [train_path, val_path, test_path]):
        return None, None, None

    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(val_path, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    with open(test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    print(f"Loaded from JSON splits:")
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    return train_data, val_data, test_data


# ============== Data Loading ==============
def parse_conllu_file(file_path):
    """
    Parse a single Hindi .conllu NER file.

    Format per sentence block:
      Line 1: full sentence text  (skipped)
      Lines 2+: token<TAB>tag
      Blank line: sentence separator
    """
    sentences = []
    current_tokens = []
    current_tags = []
    skip_next = False  # flag to skip the sentence-text header line

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.rstrip('\n').rstrip('\r')

        # Blank line = sentence boundary
        if not line.strip():
            if current_tokens and current_tags:
                sentences.append({
                    'tokens': current_tokens,
                    'ner_tags': current_tags
                })
            current_tokens = []
            current_tags = []
            skip_next = True   # next non-blank line is a sentence header
            continue

        # Skip sentence-text header lines (no tab → plain sentence)
        if skip_next:
            skip_next = False
            if '\t' not in line:
                continue  # it's a header line, skip it
            # If it does have a tab, fall through and parse it normally

        # Parse token<TAB>tag lines
        if '\t' in line:
            parts = line.split('\t')
            if len(parts) >= 2:
                token = parts[0].strip()
                tag   = parts[1].strip()
                if token and tag:
                    current_tokens.append(token)
                    current_tags.append(tag)

    # Catch last sentence if file doesn't end with blank line
    if current_tokens and current_tags:
        sentences.append({
            'tokens': current_tokens,
            'ner_tags': current_tags
        })

    return sentences


def load_all_ner_data(data_dir):
    """Load all .conllu NER files from a directory (recursively)."""
    all_sentences = []

    for root, dirs, files in os.walk(data_dir):
        for file in sorted(files):
            if (file.endswith('.conllu') or file.endswith('.txt')) and not file.startswith('.'):
                file_path = os.path.join(root, file)
                try:
                    sents = parse_conllu_file(file_path)
                    all_sentences.extend(sents)
                except Exception as e:
                    print(f"  [WARN] Error loading {file}: {e}")

    return all_sentences


# ============== Tag Normalization ==============
def get_unique_tags(data):
    """Extract all unique NER tags from the dataset."""
    tags = set()
    for item in data:
        tags.update(item['ner_tags'])
    return sorted(list(tags))


def normalize_tags(data):
    """
    Normalize common tag inconsistencies found in Hindi NER data.
    Extend this mapping as new inconsistencies are discovered.
    """
    tag_mapping = {
        # Generic noise → O
        'o': 'O', '0': 'O', 'OO': 'O', 'O**': 'O',
        "'": 'O', '5': 'O', 'CEE': 'O',
        'Artifacts': 'O',            # bare label without B-/I-
        '-Organization': 'O',        # malformed
        '-time': 'O',                # malformed
        'B=LOCOMOTIVE': 'O',

        # Capitalisation fixes
        'B-location':     'B-Location',  'I-location':     'I-Location',
        'B-person':       'B-Person',    'I-person':       'I-Person',
        'B-organization': 'B-Organization', 'I-organization': 'I-Organization',
        'B-Organisation': 'B-Organization', 'I-Organisation': 'I-Organization',
        'B-date':         'B-Date',      'I-date':         'I-Date',
        'B-DATE':         'B-Date',      'I-DATE':         'I-Date',
        'B-time':         'B-Time',      'I-time':         'I-Time',
        'B-count':        'B-Count',     'I-count':        'I-Count',
        'B-unit':         'B-Unit',      'I-unit':         'I-Unit',
        'B-quantity':     'B-Quantity',  'I-quantity':     'I-Quantity',
        'B-artifact':     'B-Artifacts', 'I-artifact':     'I-Artifacts',
        'B-Artefact':     'B-Artifacts', 'I-Artefact':     'I-Artifacts',
        'B-Artefacts':    'B-Artifacts', 'I-Artefacts':    'I-Artifacts',
        'B-ArtifactS':    'B-Artifacts', 'I-ArtifactS':    'I-Artifacts',
        'BArtifacts':     'B-Artifacts',
        'B-facility':     'B-Facilities','I-facility':     'I-Facilities',
        'B-Facility':     'B-Facilities','I-Facility':     'I-Facilities',
        'B-disease':      'B-Disease',   'I-disease':      'I-Disease',
        'B-currency':     'B-Currency',  'I-currency':     'I-Currency',
        'B-day':          'B-Day',       'I-day':          'I-Day',

        # Spacing / typo fixes
        'I -Quantity':    'I-Quantity',
        'Ia-Unit':        'I-Unit',
        'i-Time':         'I-Time',
        'I_Time':         'I-Time',
        'D-Time':         'B-Time',

        # Rare / unsupported types → O (keep label space clean)
        'B-PLANT': 'O',     'I-PLANT': 'O',  'I=PLANT': 'O',
        'B-ORGANISM': 'O',  'I-ORGANISM': 'O',
        'B-ENTERTAINMENT': 'O', 'I-ENTERTAINMENT': 'O',
        # Additional noise found in full dataset scan
        '-Date': 'O', 'I_LOCOMOTIVE': 'O', 'Os': 'O', 'Time': 'O',
        'b-Artifacts': 'B-Artifacts', 'observe': 'O', 'of': 'O',
        'VII': 'O', 'O"': 'O', 'O2': 'O', 'OBC': 'O',
        'O-Organization': 'O', 'O I-Artifacts': 'O',
    }

    VALID_TAGS = {
        'O',
        'B-Location', 'I-Location',
        'B-Organization', 'I-Organization',
        'B-Person', 'I-Person',
        'B-Date', 'I-Date',
        'B-Count', 'I-Count',
        'B-Quantity', 'I-Quantity',
        'B-Artifacts', 'I-Artifacts',
        'B-Facilities', 'I-Facilities',
        'B-Currency', 'I-Currency',
        'B-Time', 'I-Time',
        'B-Unit', 'I-Unit',
        'B-Day', 'I-Day',
        'B-Disease', 'I-Disease'
    }

    normalized_data = []
    changes_counter = Counter()

    for item in data:
        new_tags = []
        for tag in item['ner_tags']:
            mapped = tag_mapping.get(tag, tag)
            if mapped not in VALID_TAGS:
                mapped = 'O'
            if mapped != tag:
                changes_counter[f"{tag} -> {mapped}"] += 1
            new_tags.append(mapped)
        normalized_data.append({
            'tokens': item['tokens'],
            'ner_tags': new_tags
        })

    total_changes = sum(changes_counter.values())
    print(f"  Tag normalization: {total_changes} changes made")
    if changes_counter:
        for change, count in changes_counter.most_common(10):
            print(f"    {change}: {count}")

    return normalized_data


def filter_invalid_sentences(data, label2id):
    """Remove sentences with tags not present in label2id."""
    valid = []
    skipped = 0
    for item in data:
        if all(tag in label2id for tag in item['ner_tags']):
            valid.append(item)
        else:
            skipped += 1
    if skipped:
        print(f"  Filtered out {skipped} sentences with unknown tags")
    return valid


# ============== Dataset Preparation ==============
def convert_to_hf_dataset(data, label2id):
    """Convert parsed data list to HuggingFace Dataset format."""
    return Dataset.from_dict({
        'tokens':   [item['tokens'] for item in data],
        'ner_tags': [[label2id[tag] for tag in item['ner_tags']] for item in data]
    })


# ============== Training ==============
def main(args):
    print(f"\n{'='*55}")
    print(f"  Hindi NER Training - MuRIL")
    print(f"  Model: {MODEL_NAME}")
    print(f"{'='*55}\n")

    # ---- Load data ----
    print("Loading data...")
    train_data, val_data, test_data = load_json_splits(args.splits_path)

    if train_data is None:
        # Load from raw .conllu files
        if not args.data_path:
            print("ERROR: No JSON splits found and --data_path not provided.")
            print("       Run prepare_splits.py first, or pass --data_path ./data")
            return

        print(f"JSON splits not found. Loading raw data from: {args.data_path}")
        all_data = []

        for set_name in HINDI_NER_SETS:
            set_path = os.path.join(args.data_path, set_name)
            if os.path.exists(set_path):
                sents = load_all_ner_data(set_path)
                all_data.extend(sents)
                print(f"  {set_name}: {len(sents)} sentences")
            else:
                print(f"  [WARN] {set_name} not found at {set_path}")

        print(f"\nTotal sentences loaded: {len(all_data)}")

        # Normalize tags
        print("\nNormalizing tags...")
        all_data = normalize_tags(all_data)

        # Split 80 / 10 / 10
        print("\nSplitting data (80/10/10)...")
        train_data, temp_data = train_test_split(all_data, test_size=0.20, random_state=42)
        val_data, test_data   = train_test_split(temp_data, test_size=0.50, random_state=42)
        print(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # ---- Build label mappings from all data ----
    all_data_combined = train_data + val_data + test_data
    unique_tags = get_unique_tags(all_data_combined)
    label2id = {tag: idx for idx, tag in enumerate(unique_tags)}
    id2label  = {idx: tag for tag, idx in label2id.items()}

    print(f"\nUnique NER tags ({len(unique_tags)}):")
    for tag in unique_tags:
        print(f"  {tag}")

    # Save label mappings
    mappings_path = os.path.join(args.output_dir, "label_mappings.json") \
        if os.path.isdir(args.output_dir) else "label_mappings_hindi.json"
    os.makedirs(args.output_dir, exist_ok=True)
    mappings_path = os.path.join(args.output_dir, "label_mappings.json")
    with open(mappings_path, 'w', encoding='utf-8') as f:
        json.dump({'label2id': label2id, 'id2label': {str(k): v for k, v in id2label.items()}}, f,
                  ensure_ascii=False, indent=2)
    print(f"\nLabel mappings saved to: {mappings_path}")

    # Filter any remaining unknown tags
    train_data = filter_invalid_sentences(train_data, label2id)
    val_data   = filter_invalid_sentences(val_data,   label2id)
    test_data  = filter_invalid_sentences(test_data,  label2id)

    # ---- Build HuggingFace DatasetDict ----
    dataset = DatasetDict({
        'train':      convert_to_hf_dataset(train_data, label2id),
        'validation': convert_to_hf_dataset(val_data,   label2id),
        'test':       convert_to_hf_dataset(test_data,  label2id)
    })

    # ---- Load MuRIL model & tokenizer ----
    print(f"\nLoading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(unique_tags),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    # ---- Tokenize & align labels ----
    def tokenize_and_align_labels(examples):
        tokenized = tokenizer(
            examples['tokens'],
            truncation=True,
            is_split_into_words=True,
            max_length=128
        )

        labels = []
        for i, label in enumerate(examples['ner_tags']):
            word_ids   = tokenized.word_ids(batch_index=i)
            label_ids  = []
            prev_word  = None

            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != prev_word:
                    label_ids.append(label[word_idx])
                else:
                    # Continuation subword: B- → I-
                    orig_label = id2label[label[word_idx]]
                    if orig_label.startswith('B-'):
                        i_tag = 'I-' + orig_label[2:]
                        label_ids.append(label2id.get(i_tag, label[word_idx]))
                    else:
                        label_ids.append(label[word_idx])
                prev_word = word_idx
            labels.append(label_ids)

        tokenized['labels'] = labels
        return tokenized

    print("Tokenizing datasets...")
    tokenized_dataset = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset['train'].column_names
    )

    # ---- Metrics ----
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)

        true_labels = [
            [id2label[l] for l in label if l != -100]
            for label in labels
        ]
        true_preds = [
            [id2label[p] for p, l in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, labels)
        ]

        return {
            'precision': precision_score(true_labels, true_preds),
            'recall':    recall_score(true_labels, true_preds),
            'f1':        f1_score(true_labels, true_preds),
        }

    # ---- Training Arguments ----
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy='epoch',
        save_strategy='epoch',
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to='none',
        save_total_limit=3,
    )

    # ---- Trainer ----
    import inspect
    trainer_init_params = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs = {
        'model': model,
        'args': training_args,
        'train_dataset': tokenized_dataset['train'],
        'eval_dataset': tokenized_dataset['validation'],
        'data_collator': DataCollatorForTokenClassification(tokenizer),
        'compute_metrics': compute_metrics,
        'callbacks': [EarlyStoppingCallback(early_stopping_patience=3)]
    }
    if 'processing_class' in trainer_init_params:
        trainer_kwargs['processing_class'] = tokenizer
    else:
        trainer_kwargs['tokenizer'] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    # ---- Train ----
    print(f"\nStarting training for {args.epochs} epochs...")
    trainer.train()

    # ---- Evaluate on test set ----
    print(f"\nEvaluating on test set...")
    test_results = trainer.evaluate(tokenized_dataset['test'])
    print(f"\nTest Results:")
    print(f"  F1:        {test_results['eval_f1']:.4f}")
    print(f"  Precision: {test_results['eval_precision']:.4f}")
    print(f"  Recall:    {test_results['eval_recall']:.4f}")

    # Detailed per-entity report
    print(f"\nDetailed classification report on test set:")
    raw_preds = trainer.predict(tokenized_dataset['test'])
    preds_argmax = np.argmax(raw_preds.predictions, axis=2)
    test_labels_raw = raw_preds.label_ids

    true_labels_all = [
        [id2label[l] for l in label if l != -100]
        for label in test_labels_raw
    ]
    true_preds_all = [
        [id2label[p] for p, l in zip(pred, label) if l != -100]
        for pred, label in zip(preds_argmax, test_labels_raw)
    ]
    print(classification_report(true_labels_all, true_preds_all))

    # ---- Save final model ----
    save_path = os.path.join(args.output_dir, 'final')
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\nModel saved to: {save_path}")
    print("Done! ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Hindi NER with MuRIL')
    parser.add_argument('--splits_path',   type=str, default='.',
                        help='Path containing pre-generated train/validation/test.json splits')
    parser.add_argument('--data_path',     type=str, default=None,
                        help='Path to raw Hindi NER data directory (Hindi_NER_Set1..8)')
    parser.add_argument('--output_dir',    type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Directory to save checkpoints and final model')
    parser.add_argument('--epochs',        type=int,   default=10,   help='Number of training epochs')
    parser.add_argument('--batch_size',    type=int,   default=16,   help='Batch size per device')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Learning rate')

    args = parser.parse_args()
    main(args)

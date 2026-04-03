# Hindi Clause Boundary Detection Model

Fine-tuned Gemma model for detecting clause boundaries in Hindi text.

## Model Details

- **Base Model:** google/gemma-3-4b-it
- **Task:** Clause Boundary Detection
- **Language:** Hindi
- **Fine-tuning Method:** LoRA (Low-Rank Adaptation)
- **Framework:** PEFT 0.18.0

## Repository Structure

```
hin-cb-model-repo/
├── model/               # Fine-tuned LoRA adapter weights and configuration
├── train/               # Training and validation datasets
│   ├── train_annotated.txt
│   └── val_annotated.txt
├── test/                # Test dataset
│   └── test_annotated.txt
├── inference.py         # Inference script
├── training.py          # Training script
├── requirements.txt     # Python dependencies
├── .gitignore          # Git ignore rules
└── README.md           # This file

```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd hin-cb-model-repo
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. You need a Hugging Face token if the base model is gated:
```bash
# Linux/Mac:
export HF_TOKEN="your_huggingface_token_here"

# Windows (PowerShell):
$env:HF_TOKEN="your_huggingface_token_here"

# Windows (CMD):
set HF_TOKEN=your_huggingface_token_here
```

**Note:** The base model (`google/gemma-3-4b-it`) will be downloaded automatically on first run (~7-8 GB) and cached locally for future use.

## Usage

### Inference

Run inference on a text file containing Hindi sentences:

```bash
python inference.py <model_path> <input_file> [output_file]
```

**Arguments:**
- `model_path`: Path to model directory (e.g., `./model`)
- `input_file`: Path to file containing Hindi sentences (one per line)
- `output_file`: (Optional) Path to save predictions. Default: `predictions_YYYYMMDD_HHMMSS.txt`

**Example:**
```bash
python inference.py ./model test/test_annotated.txt results.txt
```

### Input Format

Input file should contain Hindi sentences, one per line. Line numbers (format: `123|sentence`) are automatically removed.

### Output Format

The model outputs clause boundaries with the following types:
- **MCL** (Main Clause)
- **RCL** (Relative Clause)
- **RP** (Relative Participle)
- **COND** (Conditional)
- **NF** (Non-Finite)
- **INF** (Infinitive)
- **COM** (Complementizer)
- **ADVCL** (Adverbial)

Format: `TYPE[ text ]TYPE` (nesting allowed)

## Data

Training and validation datasets are in `train/`, and test dataset is in `test/`.

## Requirements

See `requirements.txt` for full list of dependencies. Key packages:
- `torch` - PyTorch for model operations
- `transformers` - Hugging Face transformers library
- `peft` - Parameter-Efficient Fine-Tuning (required for LoRA adapters)
- `accelerate` - Device management and optimization

## Important Notes

- **LoRA Adapter**: This repository contains a LoRA adapter, not a full model. The inference script will automatically download the base model (`google/gemma-3-4b-it`) from Hugging Face on first run.
- **First Run**: Initial inference will take longer as the base model (~7-8 GB) is downloaded and cached.
- **Model Caching**: The base model is cached locally and reused in subsequent runs (no re-download needed).


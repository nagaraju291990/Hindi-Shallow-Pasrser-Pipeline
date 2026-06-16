# Hindi NER - MuRIL Model

Named Entity Recognition for Hindi language using **google/muril-base-cased** as the base model.

**Trained Model:** [NeXuSuss/hindi-ner-muril](https://huggingface.co/NeXuSuss/hindi-ner-muril) on HuggingFace

## Model Information

| Property | Value |
|----------|-------|
| **Base Model** | [google/muril-base-cased](https://huggingface.co/google/muril-base-cased) |
| **Architecture** | BERT (MuRIL - Multilingual Representations for Indian Languages) |
| **Languages** | 17 Indian languages + transliterated text |
| **Developer** | Google Research |

## Supported Entity Types (26 classes + O)

- **Person** - Names of people (e.g., नरेंद्र मोदी)
- **Organization** - Company, institution names (e.g., टाटा)
- **Location** - Places, cities, countries (e.g., गुजरात, दिल्ली)
- **Facilities** - Buildings, hospitals, airports (e.g., एम्स)
- **Artifacts** - Man-made objects
- **Date** - Dates and years
- **Day** - Special days and weekdays
- **Time** - Time expressions
- **Count** - Numerical counts
- **Quantity** - Measurements
- **Currency** - Money amounts (e.g., रुपये)
- **Unit** - Units of measurement (e.g., किमी, किलोग्राम)
- **Disease** - Diseases and medical terms

## Installation

```bash
pip install -r requirements.txt
```

## Downloading the Model

### Option 1 — Google Drive (Recommended)

1. Download the model folder from [Google Drive](https://drive.google.com/drive/folders/10d0VU1EaAaHdNN59Myy3vVsnl1eV_UJV?usp=drive_link)
2. Place the downloaded `hindi-ner-muril` folder at `./Hindi-NER/model`

Your directory should look like:
```
Telugu-NER/
└── Hindi-NER/
    └── model/
        ├── config.json
        ├── model.safetensors
        ├── tokenizer.json
        └── ...
```

**Or download programmatically using `gdown`:**
```bash
pip install gdown
gdown --folder "https://drive.google.com/drive/folders/10d0VU1EaAaHdNN59Myy3vVsnl1eV_UJV" -O ./ner/model
```
After downloading the model unrar the model.  
### Note that the model should be placed in Hindi-Shallow-Parser-Pipeline/ner/model

## Usage

### Single Sentence

```bash
# With local model
python inference_muril.py --model_path ./model "प्रधानमंत्री नरेंद्र मोदी ने नई दिल्ली में भाषण दिया"

# Human-readable entity output
python inference_muril.py --model_path ./model --pretty "नरेंद्र मोदी भारत के प्रधानमंत्री हैं"
```

### Process File

```bash
python inference_muril.py --model_path ./model --input_file input.txt --output_file output.conll
```

### From HuggingFace Model (private, token required)

```bash
python inference_muril.py --model_path NeXuSuss/hindi-ner-muril --hf_token hf_xxxx "Hindi sentence"
```

> **Note:** Default model path is `NeXuSuss/hindi-ner-muril`. Use `--model_path ./model` if downloaded locally.

## Output Format (CoNLL)

```
नरेंद्र	B-Person
मोदी	I-Person
भारत	B-Location
के	O
```

## Training

**From JSON splits (recommended):**
```bash
python train_muril.py --output_dir ./output --epochs 10
```

**From raw data:**
```bash
python train_muril.py --data_path ./data --output_dir ./output
```

## Files

| File | Description |
|------|-------------|
| `model/` | Trained model files ([Google Drive](https://drive.google.com/drive/folders/10d0VU1EaAaHdNN59Myy3vVsnl1eV_UJV?usp=drive_link) / [HuggingFace](https://huggingface.co/NeXuSuss/hindi-ner-muril)) |
| `train.json` | Training data |
| `validation.json` | Validation data |
| `test.json` | Test data |
| `label_mappings.json` | Label to ID mappings (27 tags) |
| `train_muril.py` | Training script |
| `inference_muril.py` | Inference script |
| `requirements.txt` | Dependencies |

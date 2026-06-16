"""
Hindi NER Inference Script - MuRIL Model

Usage:
  Single sentence:  python inference_muril.py "हिंदी वाक्य यहाँ लिखें"
  Process file:     python inference_muril.py --input_file input.txt --output_file output.conll
  HuggingFace:      python inference_muril.py --model_path NeXuSuss/hindi-ner-muril "वाक्य"
  Local model:      python inference_muril.py --model_path ./model "वाक्य"
  With HF token:    python inference_muril.py --hf_token <token> "वाक्य"
  Or set env var:   HF_TOKEN=<token> python inference_muril.py "वाक्य"

Output: CoNLL format (token<TAB>tag)
"""

import os
import sys
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

DEFAULT_MODEL_PATH = "NeXuSuss/hindi-ner-muril"


def load_model(model_path, hf_token=None):
    """Load model and tokenizer from local path or HuggingFace Hub."""
    token = hf_token or os.environ.get("HF_TOKEN")
    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
    model     = AutoModelForTokenClassification.from_pretrained(model_path, token=token)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, tokenizer, device


def predict_conll(text, model, tokenizer, device):
    """
    Predict NER tags for a Hindi sentence.
    Returns list of (token, tag) tuples in CoNLL format.
    """
    id2label = model.config.id2label

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()
    tokens      = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    results      = []
    current_word = ""
    current_tag  = None

    for token, pred in zip(tokens, predictions):
        if token in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>']:
            continue

        if token.startswith('##'):
            current_word += token[2:]
        else:
            if current_word:
                results.append((current_word, id2label[current_tag]))
            current_word = token
            current_tag  = pred

    if current_word:
        results.append((current_word, id2label[current_tag]))

    return results


def predict_pretty(text, model, tokenizer, device):
    """Predict and return grouped entity spans."""
    results  = predict_conll(text, model, tokenizer, device)
    entities = []
    current_entity_tokens = []
    current_entity_type   = None

    for token, tag in results:
        if tag.startswith('B-'):
            if current_entity_tokens:
                entities.append((' '.join(current_entity_tokens), current_entity_type))
            current_entity_tokens = [token]
            current_entity_type   = tag[2:]
        elif tag.startswith('I-') and current_entity_tokens:
            current_entity_tokens.append(token)
        else:
            if current_entity_tokens:
                entities.append((' '.join(current_entity_tokens), current_entity_type))
                current_entity_tokens = []
                current_entity_type   = None

    if current_entity_tokens:
        entities.append((' '.join(current_entity_tokens), current_entity_type))

    return entities


def process_file(input_file, output_file, model, tokenizer, device):
    """Process input file (one sentence per line) and write CoNLL output."""
    with open(input_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]

    with open(output_file, 'w', encoding='utf-8') as f:
        for i, sentence in enumerate(sentences):
            results = predict_conll(sentence, model, tokenizer, device)
            for token, tag in results:
                f.write(f"{token}\t{tag}\n")
            f.write("\n")

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(sentences)} sentences...")

    print(f"Output saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Hindi NER Inference - MuRIL')
    parser.add_argument('sentence',      type=str, nargs='?', default=None,
                        help='Hindi sentence to analyse')
    parser.add_argument('--model_path',  type=str, default=DEFAULT_MODEL_PATH,
                        help='Path to trained model (local dir or HuggingFace repo id)')
    parser.add_argument('--hf_token',    type=str, default=None,
                        help='HuggingFace token for private models (or set HF_TOKEN env var)')
    parser.add_argument('--input_file',  type=str, default=None,
                        help='Input file with one sentence per line')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output CoNLL file')
    parser.add_argument('--pretty',      action='store_true',
                        help='Print grouped entities instead of CoNLL format')

    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model_path, args.hf_token)
    print(f"Model loaded on: {device}\n")

    if args.input_file and args.output_file:
        print(f"Processing file: {args.input_file}")
        process_file(args.input_file, args.output_file, model, tokenizer, device)

    elif args.sentence:
        if args.pretty:
            entities = predict_pretty(args.sentence, model, tokenizer, device)
            if entities:
                print("Entities found:")
                for span, etype in entities:
                    print(f"  [{etype}]  {span}")
            else:
                print("No named entities found.")
        else:
            results = predict_conll(args.sentence, model, tokenizer, device)
            for token, tag in results:
                print(f"{token}\t{tag}")

    else:
        print("Error: Provide either a sentence or --input_file + --output_file")
        print("\nUsage examples:")
        print('  python inference_muril.py "प्रधानमंत्री नरेंद्र मोदी ने नई दिल्ली में भाषण दिया"')
        print('  python inference_muril.py --pretty "नरेंद्र मोदी भारत के प्रधानमंत्री हैं"')
        print('  python inference_muril.py --input_file sentences.txt --output_file output.conll')
        sys.exit(1)


if __name__ == "__main__":
    main()

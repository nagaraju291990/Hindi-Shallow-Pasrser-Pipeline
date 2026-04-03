import sys
import torch
torch._dynamo.reset()
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from datetime import datetime


def load_model(model_path, hf_token=None):
    """Load fine-tuned model and tokenizer"""
    print(f"Loading model from: {model_path}")
    
    # Load adapter config to get base model name
    config = PeftConfig.from_pretrained(model_path)
    base_model_name = config.base_model_name_or_path
    
    print(f"Loading base model: {base_model_name}")
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=hf_token
    )
    
    # Load LoRA adapter
    print(f"Loading LoRA adapter from: {model_path}")
    model = PeftModel.from_pretrained(base_model, model_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        token=hf_token
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✓ Model loaded successfully")
    return model, tokenizer


def format_instruction(raw_sentence):
    """Create instruction prompt for the model"""
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


def annotate_sentence(model, tokenizer, raw_sentence, temperature=0.1, max_new_tokens=300):
    """Annotate a single sentence with clause boundaries"""
    
    # Format prompt
    prompt = format_instruction(raw_sentence)
    messages = [{"role": "user", "content": prompt}]
    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Tokenize
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    prediction = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return prediction.strip()


def load_sentences_from_file(file_path):
    """Load raw sentences from a file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    sentences = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Remove line numbers if present (format: "  123|sentence")
        if '|' in line:
            line = line.split('|', 1)[1].strip()
        
        sentences.append(line)
    
    return sentences


def save_predictions(predictions, output_file):
    """Save predictions to file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("GEMMA CLAUSE BOUNDARY PREDICTIONS\n")
        f.write("="*80 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Total sentences: {len(predictions)}\n")
        f.write("="*80 + "\n\n")
        
        for i, (raw, pred) in enumerate(predictions, 1):
            f.write(f"[{i}]\n")
            f.write(f"RAW:  {raw}\n")
            f.write(f"PRED: {pred}\n\n")
    
    print(f"✓ Predictions saved to: {output_file}")


def main():
    """Main inference function"""
    
    # Parse arguments
    if len(sys.argv) < 3:
        print("Usage: python inference.py <model_path> <input_file> [output_file]")
        print("\nExample:")
        print("  python inference.py ./best_model sentences.txt predictions.txt")
        sys.exit(1)
    
    model_path = sys.argv[1]
    input_file = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    print("="*80)
    print("GEMMA CLAUSE BOUNDARY INFERENCE")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print("="*80 + "\n")
    
    # Load model
    # If your model is gated, uncomment and add token:
    # hf_token = "your_token_here"
    model, tokenizer = load_model(model_path, hf_token=None)
    
    # Load sentences
    print(f"\nLoading sentences from: {input_file}")
    sentences = load_sentences_from_file(input_file)
    print(f"✓ Loaded {len(sentences)} sentences\n")
    
    # Annotate sentences
    print("Annotating sentences...")
    predictions = []
    
    for i, sentence in enumerate(sentences, 1):
        print(f"[{i}/{len(sentences)}] Processing...", end='\r')
        prediction = annotate_sentence(model, tokenizer, sentence)
        predictions.append((sentence, prediction))
    
    print(f"\n✓ Completed {len(predictions)} annotations\n")
    
    # Save results
    save_predictions(predictions, output_file)
    
    print("\n" + "="*80)
    print("INFERENCE COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()

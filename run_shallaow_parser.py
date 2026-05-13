import sys
import subprocess
import re
import os
import torch
import shutil
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

# -------- CB MODEL --------
def load_cb_model(model_path, hf_token):
    config = PeftConfig.from_pretrained(model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        device_map="cpu",
        torch_dtype=torch.bfloat16,
        token=hf_token
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def get_cb_annotation(model, tokenizer, raw_sentence):
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
    prompt = instruction.format(input=raw_sentence)
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
    return prediction.strip()

def parse_cb_to_boundaries(cb_output, tokens):
    boundary_tags = [[] for _ in tokens]

    pattern = r'(\w+)\[\s*([^\[\]]+)\s*\]\1'
    temp_output = cb_output
    all_clauses = []

    while True:
        match = re.search(pattern, temp_output)
        if not match:
            break

        clause_type = match.group(1)
        clause_text = match.group(2).strip()
        clause_tokens = clause_text.split()

        all_clauses.append((clause_type, clause_tokens))

        temp_output = temp_output[:match.start()] + clause_text + temp_output[match.end():]

    all_clauses.reverse()

    for clause_type, clause_tokens in all_clauses:
        for i in range(len(tokens)):
            if tokens[i:i+len(clause_tokens)] == clause_tokens:
                boundary_tags[i].append(f"{{{clause_type}}}")
                boundary_tags[i+len(clause_tokens)-1].append(f"{{/{clause_type}}}")

    return boundary_tags
def ssf_to_final_format(data, cb_tags, fileno=1):
    output = []
    sentno = 0
    token_id = 0
    idx = 0

    for entry in data:
        if entry is None:
            sentno += 1
            output.append("\n")
            token_id = 0
            continue

        cols = entry["cols"]
        cols = ['' if x == 'unk' else x for x in cols]
        token = cols[0]
        pos = cols[1]
        chunk = cols[2]
        lcat = cols[3]
        gend = cols[4]
        numb = cols[5]
        pers = cols[6]
        case = cols[7]
        root = cols[-2]
        vib = cols[-1]
        # To fix N_NNP issue decision taken by Dipti mam to convert all N_NNP to N_NN 
        if(re.search(r'N_NNP', pos)):
            pos = "N_NN"

        fs = f"{root},{lcat},{gend},{numb},{pers},{case},{vib},{vib}"

        cb = "".join(cb_tags[idx]) if idx < len(cb_tags) else "O"

        line = f"{fileno}\t{sentno}\t{token_id}\t{token}\t{pos}\t{chunk}\t{fs}\tO\t{cb}"

        output.append(line)

        token_id += 1
        idx += 1

    return "\n".join(output)

# -------- Step1 --------
def run_step1(input_folder):
    cmd = f"""bash run_pos_chunk_morph_models_for_hindi.sh \
{input_folder} tmp_feat tmp_out \
model-lcat.m model-gender.m model-number.m model-person.m \
model-case.m model-vib.m model-pos.m model-chunk.m"""

    # ❗ do NOT stop on vib error
    subprocess.run(cmd, shell=True)


# -------- Read Step1 Output --------
def read_step1(file_path):
    data = []
    words = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() == "":
                data.append(None)
                continue

            cols = line.strip().split("\t")

            word = cols[0]
            cat = cols[3] if len(cols) > 3 else ""
            sixth = cols[5] if len(cols) > 5 else ""

            data.append({
                "cols": cols,
                "word": word,
                "cat": cat,
                "sixth": sixth
            })

            words.append(word)

    return data, words


# -------- Morph --------
def run_morph(words):
    import os

    setu_path = os.environ.get("setu")

    if not setu_path:
        raise ValueError("❌ 'setu' env variable not set")

    morph_script = f"{setu_path}/bin/sl/morph/hin/morph_run.sh"
    converter_dir = f"{setu_path}/bin/sl/morph/hin/convertor"

    utf2wx = f"{converter_dir}/utf2wx_run_ssf.sh"
    wx2utf = f"{converter_dir}/wx2utf_run_ssf.sh"

    os.makedirs("tmp_out", exist_ok=True)

    temp_input = "tmp_out/morph_input.txt"
    temp_wx = "tmp_out/morph_input.wx"
    temp_morph_out = "tmp_out/morph_output.wx"
    temp_utf_out = "tmp_out/morph_output.txt"

    with open(temp_input, "w", encoding="utf-8") as f:
        for i, word in enumerate(words, start=1):
            f.write(f"{i}\t{word}\n")

    subprocess.run(
        f"sh {utf2wx} {temp_input} > {temp_wx}",
        shell=True,
        check=True,
        capture_output=True
    )

    subprocess.run(
        f"sh {morph_script} {temp_wx} > {temp_morph_out}",
        shell=True,
        text=True,
        capture_output=True
    )

    subprocess.run(
        f"sh {wx2utf} {temp_morph_out} > {temp_utf_out}",
        shell=True,
        check=True,
        capture_output=True
    )

    with open(temp_utf_out, "r", encoding="utf-8") as f:
        output = f.read()

    print("\n====== Morph executed ======")
    print("================================\n")

    lines = output.strip().split("\n")
    lines = [l for l in lines if l.strip()]

    return lines


def build_morph_dict(morph_lines):
    morph_dict = {}

    for line in morph_lines:
        parts = line.split("\t")

        if len(parts) < 3:
            continue

        word = parts[1]

        analyses = parts[3].split("|")

        for ana in analyses:
            m = re.search(r"af='([^']+)'", ana)
            if not m:
                continue

            af = m.group(1).split(",")

            if len(af) < 6:
                continue

            root = af[0]
            cat = af[1]
            sixth = af[6]
            if(not sixth):
                sixth = ""

            key = (word, cat)

            if key not in morph_dict:
                morph_dict[key] = []

            morph_dict[key].append((root, sixth))

    return morph_dict


# -------- Merge --------
def merge(data, morph_dict):
    for entry in data:
        if entry is None:
            continue

        word = entry["word"]
        cat = entry["cat"]

        key = (word, cat)

        if key in morph_dict:
            root, sixth = morph_dict[key][0]
        else:
            root = word
            sixth = ""

        entry["cols"].append(root)
        entry["cols"].append(sixth)

    return data


# -------- Write Output --------
def write_output(data, outfile):
    with open(outfile, "w", encoding="utf-8") as f:
        for entry in data:
            if entry is None:
                f.write("\n")
            else:
                f.write("\t".join(entry["cols"]) + "\n")


# -------- Process One File --------
def process_file(input_file, output_file, cb_model, cb_tokenizer):
    print(f"▶ Processing: {input_file}")

    data, words = read_step1(input_file)

    if not words:
        print("⚠️ Empty file")
        return

    # Morph
    morph_lines = run_morph(words)
    morph_dict = build_morph_dict(morph_lines)
    merged = merge(data, morph_dict)

    # -------- Sentence reconstruction --------
    sentences = []
    current = []

    for entry in merged:
        if entry is None:
            if current:
                sentences.append(current)
                current = []
            continue
        current.append(entry["word"])

    if current:
        sentences.append(current)

    # -------- CB --------
    all_cb_tags = []
    for sent in sentences:
        sentence_text = " ".join(sent)
        clean_text = sentence_text.strip()
        clean_text = re.sub(r'^\\s*', '', clean_text).strip()
        # text_for_cb = re.sub(r' +?\.( )+?$', '', clean_text)
        text_for_cb = re.sub(r'\s*\.\s*$', '', clean_text)
        text_for_cb = clean_text.replace(" ।", "").replace("।", "").strip()
        cb_output = get_cb_annotation(cb_model, cb_tokenizer, text_for_cb)
        # print(cb_output)
        tags = parse_cb_to_boundaries(cb_output, sent)
        all_cb_tags.extend(tags)

    # -------- Final format --------
    # print(all_cb_tags)
    final_output = ssf_to_final_format(merged, all_cb_tags)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(final_output)


# -------- MAIN --------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <input_folder> <output_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

        # -------- CLEAN BEFORE RUN --------
    if os.path.exists("tmp_out") and os.path.isdir("tmp_out"):
        shutil.rmtree("tmp_out")

    if os.path.exists(output_folder) and os.path.isdir(output_folder):
        shutil.rmtree(output_folder)

    # recreate output folder
    os.makedirs(output_folder, exist_ok=True)

    HF_TOKEN = "HF_TOKEN"
    CB_MODEL_PATH = "./cb_hindi_model/model"

    print("Loading CB model...")
    cb_model, cb_tokenizer = load_cb_model(CB_MODEL_PATH, HF_TOKEN)

    print("▶ Running Shallow Parser on Folder...")
    run_step1(input_folder)

    step1_out_dir = "tmp_out"

    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(step1_out_dir):
        if file.startswith("morph_"):
            continue

        if "output" not in file:
            continue

        input_path = os.path.join(step1_out_dir, file)

        output_file = file.replace(".txt", "_final.txt")
        output_path = os.path.join(output_folder, output_file)

        # process_file(input_path, output_path)
        process_file(input_path, output_path, cb_model, cb_tokenizer)

    print("✅ Done (output saved in given output directory)")

    import shutil

if os.path.exists("tmp_out"):
    shutil.rmtree("tmp_out")

# print("✅ Done (tmp cleaned, output saved)")

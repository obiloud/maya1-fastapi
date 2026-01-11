import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import time


CODE_START_TOKEN_ID = 128257
CODE_END_TOKEN_ID = 128258
CODE_TOKEN_OFFSET = 128266
SNAC_MIN_ID = 128266
SNAC_MAX_ID = 156937
SNAC_TOKENS_PER_FRAME = 7

SOH_ID = 128259
EOH_ID = 128260
SOA_ID = 128261
BOS_ID = 128000
TEXT_EOT_ID = 128009

tokenizer = AutoTokenizer.from_pretrained(
    "maya-research/maya1",
    trust_remote_code=True
)

with open('dataset.json', 'r') as file:
    data = json.load(file)

benchmark_dataset = []

for i, example in enumerate(data):
    start_inference = time.perf_counter()
    description = example["description"]
    text = example["text"]

    # ENCODE DIRECTLY (No string decoding/encoding for special tokens)
    desc_ids = tokenizer.encode(f'<description="{description}"> {text}', add_special_tokens=False)
    
    # Reconstruct the exact sequence expected by Maya1
    # BOS + SOH + TEXT + EOT + EOH + SOA + SOS
    input_ids_list = [
        SOH_ID, 
        BOS_ID, 
        *desc_ids, 
        TEXT_EOT_ID, 
        EOH_ID, 
        SOA_ID, 
        CODE_START_TOKEN_ID
    ]
    
    prompt = tokenizer.decode(input_ids_list)
    
    benchmark_dataset.append({"description":description, "text": text, "prompt": prompt})

    end_inference = time.perf_counter()
    print(f"⏱️ Inference for item {i} took {end_inference - start_inference:.2f}s.")


with open("benchmark_dataset.json", "w") as file:
    file.write(json.dumps(benchmark_dataset, indent=2))

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

model = AutoModelForCausalLM.from_pretrained(
    "maya-research/maya1", 
    dtype=torch.bfloat16, 
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "maya-research/maya1",
    trust_remote_code=True
)

with open('dataset.json', 'r') as file:
    data = json.load(file)

calibration_dataset = []

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
    
    inputs = torch.tensor([input_ids_list]).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            inputs, 
            max_new_tokens=2048,  # Increase to let model finish naturally
            min_new_tokens=28,  # At least 4 SNAC frames
            temperature=0.4, 
            top_p=0.9, 
            repetition_penalty=1.1,  # Prevent loops
            do_sample=True,
            eos_token_id=CODE_END_TOKEN_ID,  # Stop at end of speech token
            pad_token_id=tokenizer.pad_token_id,
        )
    
    generated_ids = outputs[0, inputs.shape[1]:].tolist()
    
    calibration_dataset.append({"description":description, "text": text, "generated_ids": generated_ids})

    end_inference = time.perf_counter()
    print(f"⏱️ Inference for item {i} took {end_inference - start_inference:.2f}s.")


with open("calibration_dataset.json", "w") as file:
    file.write(json.dumps(calibration_dataset, indent=2))

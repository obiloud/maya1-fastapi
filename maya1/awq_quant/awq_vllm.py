from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, SparseAutoModelForCausalLM

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.utils import dispatch_for_generation

import json

# Select model and load it.
MODEL_ID = "./local_model"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Select calibration dataset.
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

# Select number of samples. 256 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 2048

with open('dataset.json', 'r') as file:
    data = json.load(file)

print(data)

# Load dataset and preprocess.
ds = Dataset.from_list(data)

# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    recipe="recipe.yaml",
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    output_dir="maya1-awq-asym"
)


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


def build_prompt(tokenizer, description: str, text: str) -> str:
    """Build formatted prompt for Maya1."""
    soh_token = tokenizer.decode([SOH_ID])
    eoh_token = tokenizer.decode([EOH_ID])
    soa_token = tokenizer.decode([SOA_ID])
    sos_token = tokenizer.decode([CODE_START_TOKEN_ID])
    eot_token = tokenizer.decode([TEXT_EOT_ID])
    bos_token = tokenizer.bos_token
    
    formatted_text = f'<description="{description}"> {text}'
    
    prompt = (
        soh_token + bos_token + formatted_text + eot_token +
        eoh_token + soa_token + sos_token
    )
    
    return prompt

# Confirm generations of the quantized model look sane.
print("\n\n")
print("========== SAMPLE GENERATION ==============")

model_awq = SparseAutoModelForCausalLM.from_pretrained(
    "./maya1-awq-asym",
    device_map="auto",
    dtype="auto"
)
tokenizer_awq = AutoTokenizer.from_pretrained("./maya1-awq-asym")

prompt = build_prompt(tokenizer_awq, "Realistic male voice in the 30s age with american accent. Normal pitch, warm timbre, conversational pacing.", "Hello! This is Maya1 <laugh_harder> the best open source voice AI model with emotions.")

inputs = tokenizer_awq(prompt, return_tensors="pt")
outputs = model_awq.generate(**inputs, 
            max_new_tokens=2048,  # Increase to let model finish naturally
            min_new_tokens=28,  # At least 4 SNAC frames
            temperature=0.4, 
            top_p=0.9, 
            repetition_penalty=1.1,  # Prevent loops
            eos_token_id=CODE_END_TOKEN_ID,  # Stop at end of speech token
            pad_token_id=tokenizer.pad_token_id,
        )

generated_ids = outputs[0, inputs['input_ids'].shape[1]:].tolist()
    
print(f"Generated {len(generated_ids)} tokens")

# Debug: Check what tokens we got
print(f"   First 20 tokens: {generated_ids[:20]}")
print(f"   Last 20 tokens: {generated_ids[-20:]}")

# Check if EOS was generated
if CODE_END_TOKEN_ID in generated_ids:
    eos_position = generated_ids.index(CODE_END_TOKEN_ID)
    print(f" EOS token found at position {eos_position}/{len(generated_ids)}")

print("==========================================\n\n")

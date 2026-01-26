from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.transformers.compression.compressed_tensors_utils import modify_save_pretrained
import torch
import json



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

# Select number of samples. 256 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 2048

with open('calibration_dataset.json', 'r') as file:
    data = json.load(file)

print(data)

# Load dataset and preprocess.
ds = Dataset.from_list(data)

def preprocess_fn(example):
    description = example["description"]
    text = example["text"]
    snac_tokens = example["generated_ids"]

    # Use the specific prompt structure Maya1 expects
    desc_ids = tokenizer.encode(f'<description="{description}"> {text}', add_special_tokens=False)
    
    # Construct the full sequence including the ground-truth audio
    input_ids_list = [
        SOH_ID, 
        BOS_ID, 
        *desc_ids, 
        TEXT_EOT_ID, 
        EOH_ID, 
        SOA_ID, 
        CODE_START_TOKEN_ID,
        *snac_tokens,
        CODE_END_TOKEN_ID
    ]
    
    # Return as standard Python lists. 
    # llmcompressor's data collator will convert these to 2D tensors automatically.
    return {
        "input_ids": input_ids_list,
        "attention_mask": [1] * len(input_ids_list)
    }

# Remove tensors from the map function
tokenized_ds = ds.map(preprocess_fn, batched=False, remove_columns=ds.column_names)

# Apply algorithms.
oneshot(
    model=model,
    dataset=tokenized_ds, # Use the tokenized version
    recipe="./recipe.yaml",
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=len(data), # Use all samples since the list is small
    output_dir="maya1-awq"
)

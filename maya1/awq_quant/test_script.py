from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

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


def build_prompt(tokenizer, description: str, text: str):
    # Build manually using IDs to avoid tokenizer regex issues and hidden spaces
    prompt_ids = [
        SOH_ID,
        BOS_ID,
        *tokenizer.encode(f'<description="{description}"> {text}', add_special_tokens=False),
        TEXT_EOT_ID,
        EOH_ID,
        SOA_ID,
        CODE_START_TOKEN_ID
    ]
    # Return as a list of IDs directly for vLLM
    return prompt_ids


def main():
    # Confirm generations of the quantized model look sane.
    print("========== SAMPLE GENERATION ==============")
    # Initialize the engine
    llm = LLM(
        model="./maya1-awq",
        quantization="compressed-tensors",
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=2048,
        trust_remote_code=True
    )

    prompt_ids = build_prompt(llm.get_tokenizer(), 
                          "Realistic male voice in the 30s age with american accent. Normal pitch, warm timbre, conversational pacing.", 
                          "Hello! This is Maya1 <laugh_harder> the best open source voice AI model with emotions.")

    
    sampling_params = SamplingParams(
        temperature=0.4, 
        top_p=0.9, 
        max_tokens=2048,
        repetition_penalty=1.1,
        stop_token_ids=[CODE_END_TOKEN_ID]
    )
    outputs = llm.generate({"prompt_token_ids": prompt_ids}, sampling_params=sampling_params)

    generated_ids = outputs[0].outputs[0].token_ids
        
    print(f"Generated {len(generated_ids)} tokens")

    # Debug: Check what tokens we got
    print(f"   First 20 tokens: {generated_ids[:20]}")
    print(f"   Last 20 tokens: {generated_ids[-20:]}")

    # Check if EOS was generated
    if CODE_END_TOKEN_ID in generated_ids:
        eos_position = generated_ids.index(CODE_END_TOKEN_ID)
        print(f" EOS token found at position {eos_position}/{len(generated_ids)}")

    print("==========================================\n\n")

if __name__ == "__main__":
    main()
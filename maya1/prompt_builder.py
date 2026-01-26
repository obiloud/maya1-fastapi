"""
Maya1 Prompt Builder
Builds formatted prompts for description-conditioned TTS.
Format: <SOH><BOS><description="..."> text<EOT><EOH><SOA><SOS>
"""
from .constants import (
    CODE_START_TOKEN_ID,
    SOH_ID,
    EOH_ID,
    SOA_ID,
    BOS_ID,
    TEXT_EOT_ID
)


class Maya1PromptBuilder:
    """Builds prompts in the format expected by Maya1 model."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def build_prefix(self, description: str, text: str) -> dict[str, int]:    
        # Build manually using IDs to avoid tokenizer regex issues and hidden spaces
        prompt_ids = [
            SOH_ID,
            BOS_ID,
            *self.tokenizer.encode(f'<description="{description}"> {text}', add_special_tokens=False),
            TEXT_EOT_ID,
            EOH_ID,
            SOA_ID,
            CODE_START_TOKEN_ID
        ]
        # Return as a list of IDs directly for vLLM
        return {"prompt_token_ids": prompt_ids}


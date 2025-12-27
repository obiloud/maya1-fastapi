"""
Maya1 Model Loader
Loads Maya1 model with vLLM engine and validates emotion tags.
"""

import os
from transformers import AutoTokenizer
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from .constants import (
    ALL_EMOTION_TAGS,
    DEFAULT_MAX_MODEL_LEN,
    SOH_ID, EOH_ID, SOA_ID, BOS_ID, TEXT_EOT_ID, CODE_START_TOKEN_ID,
)


class Maya1Model:
    """Maya1 TTS Model with vLLM inference engine."""
    
    def __init__(
        self,
        model_path: str = None,
        dtype: str = "bfloat16",
        max_model_len: int = DEFAULT_MAX_MODEL_LEN,
        gpu_memory_utilization: float = 0.8,
        tensor_parallel_size: int = 1,
        **engine_kwargs
    ):
        """
        Initialize Maya1 model with vLLM.
        
        Args:
            model_path: Path to checkpoint (local or HF repo)
            dtype: Model precision (bfloat16 recommended)
            max_model_len: Maximum sequence length
            gpu_memory_utilization: GPU memory fraction
            tensor_parallel_size: Number of GPUs
        """
        # Use provided path or environment variable or default
        if model_path is None:
            model_path = os.environ.get(
                'MAYA1_MODEL_PATH',
                os.path.expanduser('~/models/maya1-voice')
            )
        
        self.model_path = model_path
        self.dtype = dtype
        
        print(f"Initializing Maya1 Model")
        print(f"Model: {model_path}")
        
        trust_remote_code=True
        if './' in model_path:
            # local path
            trust_remote_code=False

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )
        
        print(f"Tokenizer loaded: {len(self.tokenizer)} tokens")
        
        # Validate emotion tags
        self._validate_emotion_tags()
        
        # Precompute special token strings
        self._init_special_tokens()
        
        # Initialize vLLM engine
        print(f"Initializing vLLM engine...")
        engine_args = AsyncEngineArgs(
            model=model_path,
            tokenizer=model_path,
            quantization="fp8",
            kv_cache_dtype="fp8",
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code,
            disable_log_stats=False,
            max_num_seqs=1,
            enable_prefix_caching=True,
            **engine_kwargs
        )
        
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        print(f"Maya1 Model ready\n")
    
    def _validate_emotion_tags(self):
        """Validate that all 20 emotion tags are single tokens."""
        failed_tags = []
        for tag in ALL_EMOTION_TAGS:
            token_ids = self.tokenizer.encode(tag, add_special_tokens=False)
            if len(token_ids) != 1:
                failed_tags.append((tag, len(token_ids)))
        
        if failed_tags:
            print(f"ERROR: {len(failed_tags)} emotion tags are NOT single tokens!")
            raise AssertionError(f"Emotion tags validation failed")
        
        print(f"All {len(ALL_EMOTION_TAGS)} emotion tags validated")
    
    def _init_special_tokens(self):
        """Precompute special token strings for fast prefix building."""
        self.soh_token = self.tokenizer.decode([SOH_ID])
        self.bos_token = self.tokenizer.bos_token
        self.eot_token = self.tokenizer.decode([TEXT_EOT_ID])
        self.eoh_token = self.tokenizer.decode([EOH_ID])
        self.soa_token = self.tokenizer.decode([SOA_ID])
        self.sos_token = self.tokenizer.decode([CODE_START_TOKEN_ID])
    
    async def generate(self, prompt: str, sampling_params: SamplingParams):
        """
        Generate tokens from prompt (non-streaming).
        Args:
            prompt: Input prompt
            sampling_params: vLLM sampling parameters
        Returns:
            Generated output from vLLM
        """
        request_id = f"req_{id(prompt)}"
        
        # Collect results from async generator
        final_output = None
        async for output in self.engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id
        ):
            final_output = output
        
        return [final_output] if final_output else []
    
    async def generate_stream(self, prompt: str, sampling_params: SamplingParams):
        """
        Generate tokens from prompt (streaming).
        Args:
            prompt: Input prompt
            sampling_params: vLLM sampling parameters
        Yields:
            Generated outputs from vLLM
        """
        request_id = f"req_{id(prompt)}"
        
        # Stream from engine
        async for output in self.engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id
        ):
            yield output

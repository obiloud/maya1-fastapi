"""
Maya1 Model Loader
Loads Maya1 model with vLLM engine and validates emotion tags.
"""

import os
from transformers import AutoTokenizer
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
import logging
import time 
import asyncio
from .constants import DEFAULT_MAX_MODEL_LEN

logger = logging.getLogger(__name__)

class Maya1Model:
    """Maya1 TTS Model with vLLM inference engine."""
    
    def __init__(
        self,
        model_path: str = None,
        dtype: str = "bfloat16",
        max_model_len: int = DEFAULT_MAX_MODEL_LEN,
        gpu_memory_utilization: float = 0.85,
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
        
        logger.info(f"Initializing Maya1 Model")
        logger.debug(f"Model: {model_path}")
        
        # Initialize vLLM engine
        logger.info(f"Initializing vLLM engine...")
        engine_args = AsyncEngineArgs(
            model=model_path,
            dtype=dtype,
            kv_cache_dtype="fp8",
            enforce_eager=False,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=1,
            tensor_parallel_size=tensor_parallel_size,
            enable_chunked_prefill=True,
            **engine_kwargs
        )
        
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        logger.info(f"Maya1 Model ready\n")
    
    async def get_tokenizer(self):
        return await self.engine.get_tokenizer()
    
    async def get_engine_health_status(self):
        """Directly inspect the vLLM scheduler state."""
        if not self.engine:
            return "Engine not initialized"
        
        # AsyncLLMEngine stores stats in the engine_step outputs
        # but we can also check the stats produced by the background loop
        stats = self.engine.engine.get_stats() # Internal vLLM method
        
        return {
            "num_running": stats.num_running,
            "num_swapped": stats.num_swapped,
            "num_waiting": stats.num_waiting,
            "gpu_cache_usage": stats.gpu_cache_usage,
            "cpu_cache_usage": stats.cpu_cache_usage,
        }

    async def generate(self, prompt, sampling_params: SamplingParams):
        """
        Generate tokens from prompt (non-streaming).
        Args:
            prompt: Input prompt
            sampling_params: vLLM sampling parameters
        Returns:
            Generated output from vLLM
        """
        request_id = f"req_{id(prompt)}"

        logger.info(f"🔮 [vLLM {request_id}] Starting stream iteration...")
        start_time = time.perf_counter()
        
        results_generator = self.engine.generate(prompt, sampling_params, request_id)
        
        try:
            # We wrap the iteration to see exactly when the FIRST token arrives
            first_token_received = False
            final_output = None
            
            async for request_output in results_generator:
                if not first_token_received:
                    ttft = time.perf_counter() - start_time
                    logger.info(f"⚡ [vLLM {request_id}] TTFT: {ttft:.2f}s")
                    first_token_received = True
                
                final_output = request_output
                
                # Yield control back to the event loop to let the watchdog bark
                await asyncio.sleep(0) 

            return [final_output]
        except Exception as e:
            logger.error(f"❌ [vLLM {request_id}] Error during generation: {e}")
            raise

    async def generate_stream(self, prompt, sampling_params: SamplingParams):
        """
        Generate tokens from prompt (streaming).
        Args:
            prompt: Input prompt
            sampling_params: vLLM sampling parameters
        Yields:
            Generated outputs from vLLM
        """
        request_id = f"req_{id(prompt)}"
        
        logger.info(f"🔮 [vLLM {request_id}] Starting stream iteration...")
        start_time = time.perf_counter()

        results_generator = self.engine.generate(prompt, sampling_params, request_id)
        
        # Stream from engine
        try:
            first_token_received = False

            async for output in results_generator:
                if not first_token_received:
                    ttft = time.perf_counter() - start_time
                    logger.info(f"⚡ [vLLM {request_id}] TTFT: {ttft:.2f}s")
                    first_token_received = True
                yield output
                
                # Yield control back to the event loop to let the watchdog bark
                await asyncio.sleep(0) 
        except Exception as e:
            logger.error(f"❌ [vLLM {request_id}] Error during generation: {e}")
            raise

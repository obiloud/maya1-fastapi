import re
import logging
from typing import AsyncGenerator, Optional, List
from vllm import SamplingParams
import numpy as np
from .utils import recursive_word_chunker, parse_pause_tags, generate_silent_bytes
import asyncio
from dataclasses import dataclass

from .constants import (
    CODE_START_TOKEN_ID,
    CODE_END_TOKEN_ID,
    SNAC_MIN_ID,
    SNAC_MAX_ID,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MIN_TOKENS,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_SEED,
)

RATE = 2400
CHANNELS = 1
QUEUE_MAX_SIZE = 3000
AUDIO_CHUNK_SIZE = 8192

# # Buffering Config
# BUFFER_DURATION_SEC = 3.0
# BYTES_PER_SEC = RATE * 2 * CHANNELS
# MIN_START_BYTES = BYTES_PER_SEC * BUFFER_DURATION_SEC 
# REBUFFER_TARGET_SEC = 2.0
# REBUFFER_TARGET_BYTES = BYTES_PER_SEC * REBUFFER_TARGET_SEC 
CROSSFADE_SAMPLES = 1200
MAX_WORDS_PER_CHUNK = 60
DESCRIPTION_DEFAULT = "Realistic male voice in the 40s with British accent. Low pitch, mellow timbre, slow pacing."


logger = logging.getLogger('streamin_pipeline')

@dataclass
class PipelineItem:
    type: str
    content: Optional[str]
    duration: Optional[float]

class Maya1LongPipeline:
    
    def __init__(self, model, prompt_builder, snac_decoder):
        """
        Initialize sliding window streaming pipeline.
        
        Args:
            model: Maya1Model instance
            prompt_builder: Maya1PromptBuilder instance
            snac_decoder: SNACDecoder instance
        """
        self.model = model
        self.prompt_builder = prompt_builder
        self.snac_decoder = snac_decoder
        self.audio_queue = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)
        self.previous_chunk_tail = np.zeros(CROSSFADE_SAMPLES, dtype=np.float32) 
        self.is_generating = False
        
        logger.info("Pipeline initialized")

    def prepare_pipeline(self, text) -> List[PipelineItem]:
        """
        1. Parses PAUSE tags.
        2. Chunks text segments.
        3. Returns a flat list of items: [{'type': 'text', 'content': '...'}, {'type': 'pause', 'duration': 2.0}]
        """
        raw_sequence = parse_pause_tags(text)
        pipeline_items = []
        
        r = r'[\n\s]+'

        for item in raw_sequence:
            if isinstance(item, float):
                # It's a pause
                pipeline_items.append(PipelineItem(type="pause", content=None, duration=item))
            elif isinstance(item, str):
                # It's text, chunk it further
                chunks = recursive_word_chunker(item, MAX_WORDS_PER_CHUNK)
                for c in chunks:
                    pipeline_items.append(PipelineItem(type='text', content= re.sub(pattern=r, repl=" ", string=c), duration=None))
                    
        return pipeline_items

    def _extract_snac_codes(self, token_ids: List[int]) -> List[int]:
        # Find SOS and EOS positions
        try:
            sos_idx = token_ids.index(CODE_START_TOKEN_ID)
        except ValueError:
            sos_idx = -1
        
        try:
            eos_idx = token_ids.index(CODE_END_TOKEN_ID)
        except ValueError:
            eos_idx = len(token_ids)
        
        # Extract tokens between SOS and EOS
        if sos_idx >= 0:
            snac_tokens = token_ids[sos_idx + 1:eos_idx]
        else:
            # If no SOS found, take everything before EOS
            snac_tokens = token_ids[:eos_idx]
        
        # Filter to only valid SNAC token IDs
        snac_codes = [
            token_id for token_id in snac_tokens
            if SNAC_MIN_ID <= token_id <= SNAC_MAX_ID
        ]
        
        return snac_codes
    
    def _crossfade_chunks(self, new_chunk_bytes: bytes) -> bytes:
        # Convert incoming bytes to float32 for clean DSP math
        new_audio = np.frombuffer(new_chunk_bytes, dtype=np.int16).astype(np.float32)
        
        if len(new_audio) < (2 * CROSSFADE_SAMPLES):
            return new_chunk_bytes

        # If no previous tail, just store tail and return body
        if self.previous_chunk_tail is None:
            self.previous_chunk_tail = new_audio[-CROSSFADE_SAMPLES:]
            return new_audio[:-CROSSFADE_SAMPLES].astype(np.int16).tobytes()

        # DSP Correct Fade (Float space)
        fade_out = np.linspace(1.0, 0.0, CROSSFADE_SAMPLES)
        fade_in = np.linspace(0.0, 1.0, CROSSFADE_SAMPLES)
        
        mixed = (self.previous_chunk_tail * fade_out) + (new_audio[:CROSSFADE_SAMPLES] * fade_in)
        
        # Prepare for next chunk
        self.previous_chunk_tail = new_audio[-CROSSFADE_SAMPLES:]
        body = new_audio[CROSSFADE_SAMPLES:-CROSSFADE_SAMPLES]
        
        return np.concatenate((mixed, body)).astype(np.int16).tobytes()
    
    async def fetch_audio_manager(self, sampling_params, description:str = DESCRIPTION_DEFAULT, pipeline_items:List[PipelineItem] = None):
        self.is_generating = True
    
        # This semaphore ensures we don't overload the GPU/Memory 
        # by generating too many chunks ahead of time.
        look_ahead_limit = asyncio.Semaphore(2) 

        async def process_chunk(item:PipelineItem):
            async with look_ahead_limit:
                if item.type == 'pause':
                    if self.previous_chunk_tail is not None:
                        await self.audio_queue.put(self.previous_chunk_tail.astype(np.int16).tobytes())
                        self.previous_chunk_tail = np.zeros(CROSSFADE_SAMPLES, dtype=np.float32)    # Reset crossfader

                    return generate_silent_bytes(item.duration)
                
                # 1. Inference (LLM Stage)
                prompt = self.prompt_builder.build_prefix(description, item.content)
                outputs = await self.model.generate(prompt, sampling_params)
                snac_codes = self._extract_snac_codes(outputs[0].outputs[0].token_ids)
                
                # 2. Decoding (DSP Stage)
                # While this is decoding, the semaphore allows the NEXT 
                # call to model.generate to start!
                return await self.snac_decoder.decode_single_async(snac_codes)

        try:
            # We process items in order, but the underlying tasks can overlap
            for idx, item in enumerate(pipeline_items):
                if item.type == 'text':
                    logger.info(f"  {idx}: [TTS] {item.content[:50]}...")
                else:
                    logger.info(f"  {idx}: [SILENCE] {item.duration}s")                
                
                audio_bytes = await process_chunk(item)
                
                if audio_bytes:
                    processed = self._crossfade_chunks(audio_bytes)
                    for i in range(0, len(processed), AUDIO_CHUNK_SIZE):
                        await self.audio_queue.put(processed[i:i+AUDIO_CHUNK_SIZE])
        except asyncio.CancelledError:
            logger.info("Streaming cancelled by user. Stopping inference.")
            raise            
        finally:
            if self.previous_chunk_tail is not None:
                await self.audio_queue.put(self.previous_chunk_tail.astype(np.int16).tobytes())
            self.is_generating = False
            await self.audio_queue.put(None)
        
    async def generate_speech_stream(
        self,
        description: str,
        text: str,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        seed: Optional[int] = None,
    ) -> AsyncGenerator[bytes, None]:
        """
        Generate speech audio with sliding window streaming.
        
        Args:
            description: Voice description
            text: Text to synthesize (may include <emotion> tags)
            temperature: Sampling temperature
            top_p: Nucleus sampling
            max_tokens: Max SNAC tokens to generate
            repetition_penalty: Prevent loops
            seed: Random seed
        
        Yields:
            Audio bytes (int16 PCM, 24kHz mono)
        """
        pipeline_items = self.prepare_pipeline(text)

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            min_tokens=DEFAULT_MIN_TOKENS,
            repetition_penalty=repetition_penalty,
            stop_token_ids=[CODE_END_TOKEN_ID],
            seed=seed if seed is not None else DEFAULT_SEED,
        )

        producer_task = asyncio.create_task(self.fetch_audio_manager(sampling_params, description, pipeline_items))
        
        try:
            while True:
                # Re-buffering logic based on queue size, not global byte counter
                if self.audio_queue.qsize() < 3 and self.is_generating:
                    await asyncio.sleep(0.05)
                    continue
                    
                data = await self.audio_queue.get()
                if data is None: 
                    self.audio_queue.task_done()
                    break

                yield data
        finally:
            await producer_task

            self.previous_chunk_tail = np.zeros(CROSSFADE_SAMPLES, dtype=np.float32)
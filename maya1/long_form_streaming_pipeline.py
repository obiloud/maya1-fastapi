import logging
import numpy as np
import multiprocessing as mp
import re
import os
import logging
import asyncio

from typing import AsyncGenerator, List
from vllm import SamplingParams
from .async_snac_decoder import AsyncSNACProcess

from .constants import (
    CODE_END_TOKEN_ID, SNAC_MIN_ID, SNAC_MAX_ID, 
    DEFAULT_TEMPERATURE, DEFAULT_TOP_P, AUDIO_SAMPLE_RATE,
)

logger = logging.getLogger("maya1_sliding_window_pipeline")

class Maya1SlidingWindowPipeline:
    def __init__(self, model, prompt_builder, log_queue):
        self.model = model
        self.prompt_builder = prompt_builder
        self.log_queue = log_queue
        self.prev_tail = None
        
        # Audio Window Config
        self.WINDOW_SIZE = 28  # 4 frames
        self.YIELD_STRIDE = 7  # 1 frame
        self.FADE_MS = 50
        self.FADE_SAMPLES = int((AUDIO_SAMPLE_RATE * self.FADE_MS) / 1000)

        ctx = mp.get_context('spawn')

        self.ready_event = ctx.Event()

        # Initialize Async Decoding Process
        self.token_q = ctx.Queue(maxsize=20)
        self.audio_q = ctx.Queue(maxsize=20)

        self.decoder_proc = AsyncSNACProcess(
            self.token_q, 
            self.audio_q, 
            self.log_queue,
            ready_event=self.ready_event
        )
        self.decoder_proc.start()

        main_pid = os.getpid()
        child_pid = self.decoder_proc.pid

        logger.info(f"Main PID is {main_pid}")
        logger.info(f"Child PID is {child_pid}")

        if main_pid == child_pid:
            raise RuntimeError("CRITICAL: Multiprocessing failed to fork a new PID!")

    def _split_text_contextually(self, text: str) -> List[str]:
        """Splits text at natural phrase boundaries for stable prosody."""
        # Split by . , ! ? or ; but keep the punctuation
        return re.split(r"(?<=[.!?])\s+", text)

    def _apply_micro_fade(self, new_audio):
        # 5ms is the 'goldilocks' zone for SNAC stitching
        fade_len = int(24000 * 0.005) 
        
        if self.prev_tail is None:
            self.prev_tail = new_audio[-fade_len:]
            return (new_audio[:-fade_len] * 32767).astype(np.int16).tobytes()

        # Square root fade for constant power
        fade_in = np.sqrt(np.linspace(0, 1, fade_len))
        fade_out = np.sqrt(np.linspace(1, 0, fade_len))
        
        # Blend the old tail with the new head
        overlap = (self.prev_tail * fade_out) + (new_audio[:fade_len] * fade_in)
        
        # Store new tail for next time
        self.prev_tail = new_audio[-fade_len:]
        
        # Combine: [Body of previous] [Overlap] [Body of current]
        combined = np.concatenate([overlap, new_audio[fade_len:-fade_len]])
        return (combined * 32767).astype(np.int16).tobytes()

    async def generate_speech_stream(self, description: str, text: str, **kwargs) -> AsyncGenerator[bytes, None]:
        text_chunks = self._split_text_contextually(text)
        loop = asyncio.get_event_loop()
        
        # Internal async queue to bridge the multiprocess audio_q
        internal_audio_q = asyncio.Queue()

        # Background task to drain the multiprocessing queue into the async queue
        async def queue_drainer():
            while True:
                try:
                    # Use run_in_executor to avoid blocking the event loop
                    chunk = await loop.run_in_executor(None, self.audio_q.get)
                    if chunk is None: break # Sentinel to stop
                    await internal_audio_q.put(chunk)
                except Exception as e:
                    logger.error(f"Drainer error: {e}")
                    break

        drain_task = asyncio.create_task(queue_drainer())

        try:
            for i, chunk in enumerate(text_chunks):
                logger.info(f"Processing Text Chunk {i+1}/{len(text_chunks)}")
                prompt = self.prompt_builder.build_prefix(description=description, text=chunk)
                sampling_params = SamplingParams(
                    temperature=kwargs.get('temperature', DEFAULT_TEMPERATURE),
                    top_p=kwargs.get('top_p', DEFAULT_TOP_P),
                    max_tokens=4096, # Ensure this is high enough for the whole sentence
                    stop_token_ids=[CODE_END_TOKEN_ID],
                )

                snac_buffer = []
                last_pos = 0

                async for output in self.model.generate_stream(prompt, sampling_params):
                    all_tokens = output.outputs[0].token_ids
                    new_tokens = all_tokens[last_pos:]
                    last_pos = len(all_tokens)
                    
                    for t in new_tokens:
                        if SNAC_MIN_ID <= t <= SNAC_MAX_ID:
                            snac_buffer.append(t)
                    
                    # Handoff frames in groups of 7
                    if len(snac_buffer) >= 7:
                        num_frames = len(snac_buffer) // 7
                        to_send = snac_buffer[:num_frames * 7]
                        snac_buffer = snac_buffer[num_frames * 7:]
                        await loop.run_in_executor(None, self.token_q.put, to_send)

                    # Yield any audio that the background drainer has found
                    while not internal_audio_q.empty():
                        audio = await internal_audio_q.get()
                        yield self._apply_micro_fade(audio)

                # Flush remaining tokens for the sentence
                if snac_buffer:
                    await loop.run_in_executor(None, self.token_q.put, snac_buffer)

            # Final drain to catch the last bits of audio
            await asyncio.sleep(1.0) 
            while not internal_audio_q.empty():
                yield self._apply_micro_fade(await internal_audio_q.get())

        finally:
            await loop.run_in_executor(None, self.audio_q.put, None) # Stop drainer
            await drain_task
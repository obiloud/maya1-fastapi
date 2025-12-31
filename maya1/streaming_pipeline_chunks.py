import re
import logging
from typing import AsyncGenerator, Optional, List
from vllm import SamplingParams
import numpy as np
from .utils import recursive_word_chunker, parse_pause_tags, generate_silent_bytes
import asyncio
from dataclasses import dataclass
import time

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
QUEUE_MAX_SIZE = 2000
AUDIO_CHUNK_SIZE = 8192

# # Buffering Config
BUFFER_DURATION_SEC = 3.0
BYTES_PER_SEC = RATE * 2 * CHANNELS
MIN_START_BYTES = BYTES_PER_SEC * BUFFER_DURATION_SEC 
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
        self.previous_chunk_tail = np.zeros(CROSSFADE_SAMPLES, dtype=np.float32) 
        
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

    async def fetch_audio_manager(self, audio_queue, description: str, pipeline_items: List[PipelineItem], **kwargs):
        logger.info("🚀 Producer Started")
        # This queue allows the LLM to hand off tokens to the Decoder
        # and immediately start the next generation call.
        decode_queue = asyncio.Queue(maxsize=2)

        async def decoder_worker():
            """
            CONSUMER TASK: Runs in the background.
            It pulls tokens from the queue and decodes them while the LLM 
            is busy generating the next chunk.
            """
            while True:
                item_data = await decode_queue.get()
                if item_data is None:  # Stop signal
                    decode_queue.task_done()
                    break
                
                idx, p_item, snac_codes = item_data
                
                try:
                    if p_item.type == 'pause':
                        # Flush tail before silence
                        if self.previous_chunk_tail is not None:
                            await audio_queue.put(self.previous_chunk_tail.astype(np.int16).tobytes())
                            self.previous_chunk_tail = None
                        
                        audio_bytes = generate_silent_bytes(p_item.duration)
                    else:
                        # While this is awaiting, the loop below is hitting the LLM again!
                        audio_bytes = await self.snac_decoder.decode_single_async(snac_codes)
                    
                    if audio_bytes:
                        logger.info(f"✅ Producer: Item {idx} decoded")
                        processed = self._crossfade_chunks(audio_bytes)
                        for i in range(0, len(processed), AUDIO_CHUNK_SIZE):
                            await audio_queue.put(processed[i:i+AUDIO_CHUNK_SIZE])
                finally:
                    decode_queue.task_done()

        # 1. Start the Decoder worker in the background
        worker_task = asyncio.create_task(decoder_worker())

        try:
            # 2. PRODUCER LOOP: Focus only on the LLM (The Bottleneck)
            for i, item in enumerate(pipeline_items):
                logger.info(f"📦 Producer: Processing item {i}")

                if item.type == 'pause':
                    await decode_queue.put((i, item, None))
                    continue

                prompt = self.prompt_builder.build_prefix(description, item.content)

                sampling_params = SamplingParams(
                    temperature=kwargs.get("temperature", DEFAULT_TEMPERATURE),
                    top_p=kwargs.get("top_p", DEFAULT_TOP_P),
                    max_tokens=kwargs.get("max_tokens", DEFAULT_MAX_TOKENS),
                    min_tokens=kwargs.get("min_tokens", DEFAULT_MIN_TOKENS),
                    stop_token_ids=[CODE_END_TOKEN_ID],
                )
                
                # This is your 0.5s stage
                start_inference = time.perf_counter()
                outputs = await self.model.generate(prompt, sampling_params)
                end_inference = time.perf_counter()
                logger.info(f"⏱️ Inference for item {i} took {end_inference - start_inference:.2f}s")
                
                if outputs:
                    snac_codes = self._extract_snac_codes(outputs[0].outputs[0].token_ids)
                    # HAND OFF to the worker and IMMEDIATELY loop back to 'model.generate'
                    await decode_queue.put((i, item, snac_codes))
                    logger.info(f"📦 Producer: item {i} SNAC codes handed off")
            
            # 3. Wait for the background worker to finish all hand-offs
            await decode_queue.join()
            logger.info("🏁 Producer: All items finished")
            
        except Exception as e:
            logger.error(f"❌ Producer CRASHED: {e}", exc_info=True)
        finally:
            # 4. Cleanup
            await decode_queue.put(None)
            await worker_task
            
            # Final tail flush
            if self.previous_chunk_tail is not None:
                await audio_queue.put(self.previous_chunk_tail.astype(np.int16).tobytes())
                self.previous_chunk_tail = None
                
            await audio_queue.put(None)
        
    async def generate_speech_stream(self, description: str, text: str, **kwargs) -> AsyncGenerator[bytes, None]:
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
        logger.info(f"🏁 Starting stream for text: {text[:50]}...")

        audio_queue = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)

        chunks_sent = 0

        start_time = asyncio.get_event_loop().time()

        pipeline_items = self.prepare_pipeline(text)

        producer_task = asyncio.create_task(self.fetch_audio_manager(audio_queue, description, pipeline_items, **kwargs))

        async def watchdog():
            """Logs status every 2s while the producer is running."""
            while not producer_task.done():
                elapsed = asyncio.get_event_loop().time() - start_time
                q_size = audio_queue.qsize()
                logger.info(
                    f"🐕 Watchdog: T+{elapsed:.1f}s | "
                    f"Queue: {q_size} chunks | "
                    f"Producer Alive: {not producer_task.done()}"
                )
                health = await self.model.get_engine_health_status()
                logger.info(f"🐕 Watchdog Engine Status: {health}")
                await asyncio.sleep(2.0)

        # Start the watchdog
        watchdog_task = asyncio.create_task(watchdog())

        try:
            while True:
                current_buffer_size = audio_queue.qsize() * AUDIO_CHUNK_SIZE
                required_buffer = MIN_START_BYTES if chunks_sent > 0 else (BYTES_PER_SEC * 0.5)

                # Re-buffering logic based on queue size, not global byte counter
                if current_buffer_size < required_buffer and producer_task.done() is False:
                    await asyncio.sleep(0.05)
                    continue
                    
                data = await audio_queue.get()
                if data is None: 
                    logger.info("🛑 Sentinel received. End of stream.")
                    audio_queue.task_done()
                    break

                yield data
                chunks_sent += 1

        except Exception as e:
            logger.error(f"❌ Producer status: {not producer_task.done()}. Error: {e}")

        finally:
            watchdog_task.cancel()
            if not producer_task.done():
                producer_task.cancel()
                try:
                    await producer_task
                except asyncio.CancelledError:
                    pass
            logger.info(f"🔚 Generator closed. Sent {chunks_sent} chunks.")
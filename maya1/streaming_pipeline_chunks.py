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
        
        logger.info("Pipeline initialized")

    async def pre_warm(self):
        """
        Runs a dummy inference through the SNAC decoder to 
        initialize PyTorch kernels and prevent cold-start timeouts.
        """
        logger.info("🔥 Pre-warming SNAC Decoder...")
        # 28 tokens of silence/dummy data (4 frames)
        dummy_tokens = [1000] * 28 
        try:
            # This will trigger kernel compilation/loading
            await self.snac_decoder.decode_single_async(dummy_tokens)
            logger.info("✅ SNAC Decoder warmed up and ready.")
        except Exception as e:
            logger.error(f"⚠️ Pre-warm failed: {e}")

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
        """
        Extracts SNAC codes within the specific CODE_START and CODE_END markers,
        ensuring all tokens fall within the valid SNAC_MIN/MAX range.
        """
        # 1. Handle potential Mock objects in tests
        # If token_ids is a MagicMock, it won't support .index(). 
        # We convert to a list or handle the Mock gracefully.
        if hasattr(token_ids, "__getitem__") and not isinstance(token_ids, list):
            try:
                token_ids = list(token_ids)
            except Exception:
                logger.warning("token_ids is a Mock that cannot be converted to list.")
                return []

        # 2. Find Start Marker (SOS)
        try:
            # Start looking from the beginning
            sos_idx = token_ids.index(CODE_START_TOKEN_ID)
            start_from = sos_idx + 1
        except (ValueError, AttributeError):
            # Fallback for mocks/incomplete streams: start from 0
            start_from = 0
        
        # 3. Find End Marker (EOS)
        try:
            # Only look for EOS *after* the SOS
            eos_idx = token_ids.index(CODE_END_TOKEN_ID, start_from)
        except (ValueError, AttributeError):
            # If no EOS, take everything to the end
            eos_idx = len(token_ids)
        
        # 4. Range Extraction
        snac_tokens = token_ids[start_from:eos_idx]
        
        # 5. Strict Validation & Integer Filtering
        # This prevents the ">=" TypeError by ensuring we only compare real ints
        snac_codes = []
        for t in snac_tokens:
            try:
                # Ensure t is an int (handles cases where Mock returns Mocks)
                val = int(t) 
                if SNAC_MIN_ID <= val <= SNAC_MAX_ID:
                    snac_codes.append(val)
            except (ValueError, TypeError):
                continue

        # 6. Alignment Check
        # SNAC requires 7 tokens per frame. We trim any trailing partial frames
        # to avoid the sliding window getting misaligned.
        num_frames = len(snac_codes) // 7
        if num_frames == 0 and len(snac_codes) > 0:
            logger.debug(f"Received {len(snac_codes)} tokens, but not enough for a full SNAC frame.")
        
        return snac_codes[:num_frames * 7]

    async def fetch_audio_manager(self, audio_queue, description: str, pipeline_items: List[PipelineItem], **kwargs):
        logger.info("🚀 Producer Started")

        # sliding window configuration
        TOKENS_PER_FRAME = 7
        WINDOW_FRAMES = 4 
        WINDOW_SIZE = WINDOW_FRAMES * TOKENS_PER_FRAME # 28 tokens

        # This queue allows the LLM to hand off tokens to the Decoder
        # and immediately start the next generation call.
        decode_queue = asyncio.Queue(maxsize=2)

        async def decoder_worker():
            token_buffer = []
            is_first_chunk = True
            
            while True:
                item_data = await decode_queue.get()
                
                # --- STOP SIGNAL & FINAL FLUSH ---
                if item_data is None:
                    if len(token_buffer) >= 7:
                        logger.info(f"Final flush: {len(token_buffer)} tokens")
                        # Use standard decode for the tail to avoid cropping end-of-speech
                        audio_bytes = await self.snac_decoder.decode_single_async(
                            token_buffer, 
                            use_sliding_window=False,
                        )
                        if audio_bytes:
                            await audio_queue.put(audio_bytes)
                    decode_queue.task_done()
                    break
                
                idx, p_item, snac_codes = item_data
                
                try:
                    if p_item.type == 'pause':
                        token_buffer = [] # Reset buffer on pause
                        await audio_queue.put(generate_silent_bytes(p_item.duration))
                    else:
                        token_buffer.extend(snac_codes)
                        logger.info(f"Buffer size: {len(token_buffer)}")

                        # --- FAST PATH: Get sound playing immediately ---
                        if is_first_chunk and len(token_buffer) >= 7:
                            first_frame = token_buffer[:7]
                            audio_bytes = await self.snac_decoder.decode_single_async(
                                first_frame, use_sliding_window=False, trim_warmup=True
                            )
                            if audio_bytes:
                                await audio_queue.put(audio_bytes)
                            token_buffer = token_buffer[7:]
                            is_first_chunk = False

                        # --- SLIDING WINDOW: Smooth continuous playback ---
                        while len(token_buffer) >= WINDOW_SIZE:
                            window = token_buffer[:WINDOW_SIZE]
                            audio_bytes = await self.snac_decoder.decode_single_async(
                                window, use_sliding_window=True
                            )
                            if audio_bytes:
                                await audio_queue.put(audio_bytes)
                            
                            token_buffer = token_buffer[TOKENS_PER_FRAME:]
                            await asyncio.sleep(0) # Keep event loop alive
                        logger.info(f"✅ Decoder: Chunk {idx} processed into window stream")
                finally:
                    decode_queue.task_done()

        # 1. Start the Decoder worker in the background
        worker_task = asyncio.create_task(decoder_worker())

        def handle_worker_result(task):
            try:
                task.result()
            except Exception as e:
                logger.error(f"💥 Decoder Worker DIED: {e}", exc_info=True)

        worker_task.add_done_callback(handle_worker_result)

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
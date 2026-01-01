import re
import logging
from typing import AsyncGenerator, Optional, List
from vllm import SamplingParams
import numpy as np
from .utils import recursive_word_chunker, parse_pause_tags, generate_silent_bytes
import asyncio
from dataclasses import dataclass
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import functools
from .worker import init_worker, worker_decode_task

from .constants import (
    CODE_START_TOKEN_ID,
    CODE_END_TOKEN_ID,
    SNAC_MIN_ID,
    SNAC_MAX_ID,
    SOA_ID,
    SOH_ID,
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
    
    def __init__(self, model, prompt_builder, snac_decoder_class=None, **decoder_kwargs):
        """
        Initialize sliding window streaming pipeline.
        
        Args:
            model: Maya1Model instance
            prompt_builder: Maya1PromptBuilder instance
            snac_decoder: SNACDecoder class
            **kwargs: Additional keyword argument are passed to the SNACDecoder constructor.

        Other Parameters:
            device (str): Location where the `torch.Tensor` will be allocated, such as "cpu" or "cuda". 
        """
        self.model = model
        self.prompt_builder = prompt_builder
        self.executor = ProcessPoolExecutor(
            max_workers=1,
            initializer=init_worker,
            initargs=(snac_decoder_class, decoder_kwargs)
        )
        self.decode_lock = asyncio.Lock()
        
        logger.info("Pipeline initialized")

    async def shutdown(self):
        """Call this when the server closes."""
        self.executor.shutdown(wait=False)

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

    def _extract_snac_codes_streaming(self, raw_tokens: List[int]):
        """
        Returns (list_of_codes, total_raw_tokens_consumed)
        Handles skipping start tags and aligning to 7-token frames.
        """
        if not raw_tokens:
            return [], 0

        valid_audio_codes = []
        raw_consumed = 0
        
        for token in raw_tokens:
            # 1. Skip metadata/start tags but mark them as consumed
            if token == CODE_START_TOKEN_ID:
                raw_consumed += 1
                continue
                
            # 2. Stop at end markers
            if token == CODE_END_TOKEN_ID:
                raw_consumed += 1
                break
                
            # 3. Collect valid SNAC audio codes
            if SNAC_MIN_ID <= token <= SNAC_MAX_ID:
                valid_audio_codes.append(token)
                raw_consumed += 1
            else:
                # Consume unknown/emotion tokens to move the stream forward
                raw_consumed += 1

        # 4. Alignment: We can only yield multiples of 7
        num_frames = len(valid_audio_codes) // 7
        final_codes = valid_audio_codes[:num_frames * 7]
        
        # IMPORTANT: If we didn't use some audio tokens because they didn't 
        # form a full frame, we "un-consume" them from the raw count 
        # so they appear in the next slice.
        unused_audio_count = len(valid_audio_codes) % 7
        final_raw_consumed = raw_consumed - unused_audio_count
        
        return final_codes, final_raw_consumed
    
    async def decoder_worker(self, decode_queue, audio_queue):
        # sliding window configuration
        TOKENS_PER_FRAME = 7
        WINDOW_FRAMES = 4 
        WINDOW_SIZE = WINDOW_FRAMES * TOKENS_PER_FRAME # 28 tokens

        token_buffer = []
        is_first_chunk = True
        loop = asyncio.get_running_loop()
        
        while True:
            item_data = await decode_queue.get()
            
            # --- STOP SIGNAL & FINAL FLUSH ---
            if item_data is None:
                if len(token_buffer) >= TOKENS_PER_FRAME:
                    logger.debug(f"Final flush: {len(token_buffer)} tokens")
                    # Final Flush: Offload to executor
                    audio_bytes = await loop.run_in_executor(
                        self.executor,
                        functools.partial(
                            worker_decode_task,
                            token_buffer,
                            use_sliding_window=False,
                        )
                    )
                    if audio_bytes:
                        await audio_queue.put(audio_bytes)
                decode_queue.task_done()
                break
            
            idx, p_item, snac_codes = item_data
            
            try:
                async with self.decode_lock:
                    if p_item.type == 'pause':
                        token_buffer = [] # Reset buffer on pause
                        await audio_queue.put(generate_silent_bytes(p_item.duration))
                    else:
                        token_buffer.extend(snac_codes)
                        logger.debug(f"Buffer size: {len(token_buffer)}")

                        # --- FAST PATH: Get sound playing immediately ---
                        if is_first_chunk and len(token_buffer) >= TOKENS_PER_FRAME:
                            first_frame = token_buffer[:TOKENS_PER_FRAME]
                            
                            logger.info(f"🚀 FAST PATH: Tokens {first_frame[:3]}... | Buffer: {len(token_buffer)}")

                            token_buffer = token_buffer[TOKENS_PER_FRAME:]
                            is_first_chunk = False

                            # Offload to process pool
                            audio_bytes = await loop.run_in_executor(
                                self.executor,
                                functools.partial(
                                    worker_decode_task,
                                    first_frame,
                                    use_sliding_window=False,
                                    trim_warmup=True
                                )
                            )
                            if audio_bytes:
                                await audio_queue.put(audio_bytes)

                        # --- SLIDING WINDOW: Smooth continuous playback ---
                        while len(token_buffer) >= WINDOW_SIZE:
                            window = token_buffer[:WINDOW_SIZE]

                            logger.info(f"🪟 WINDOW: Tokens {window[:3]}... | Buffer: {len(token_buffer)}")

                            # Advance the buffer BEFORE the await to prevent re-processing
                            token_buffer = token_buffer[TOKENS_PER_FRAME:]

                            # Parallelize decoding!
                            audio_bytes = await loop.run_in_executor(
                                self.executor,
                                functools.partial(
                                    worker_decode_task,
                                    window,
                                    use_sliding_window=True,
                                    trim_warmup=False
                                )
                            )
                            if audio_bytes:
                                await audio_queue.put(audio_bytes)
                            
                    logger.debug(f"✅ Decoder: Chunk {idx} processed into window stream")
            finally:
                decode_queue.task_done()

    async def fetch_audio_manager(self, audio_queue, description: str, pipeline_items: List[PipelineItem], **kwargs):
        logger.info("🚀 Producer Started")

        # This queue allows the LLM to hand off tokens to the Decoder
        # and immediately start the next generation call.
        decode_queue = asyncio.Queue(maxsize=2)
        # 1. Start the Decoder worker in the background
        worker_task = asyncio.create_task(self.decoder_worker(decode_queue, audio_queue))

        def handle_worker_result(task):
            try:
                task.result()
            except Exception as e:
                logger.error(f"💥 Decoder Worker DIED: {e}", exc_info=True)

        worker_task.add_done_callback(handle_worker_result)

        try:
            # 2. PRODUCER LOOP: Focus only on the LLM (The Bottleneck)
            for i, item in enumerate(pipeline_items):                
                if item.type == 'pause':
                    await decode_queue.put((i, item, None))
                else:
                    logger.info(f"📦 Producer: Processing item {i}")

                    # vllm_pointer must be absolute to the cumulative list provided by vLLM
                    vllm_pointer = 0
                    prompt = self.prompt_builder.build_prefix(description, item.content)

                    sampling_params = SamplingParams(
                        temperature=kwargs.get("temperature", DEFAULT_TEMPERATURE),
                        top_p=kwargs.get("top_p", DEFAULT_TOP_P),
                        max_tokens=kwargs.get("max_tokens", DEFAULT_MAX_TOKENS),
                        min_tokens=kwargs.get("min_tokens", DEFAULT_MIN_TOKENS),
                        repetition_penalty=kwargs.get('repetition_penalty', DEFAULT_REPETITION_PENALTY),
                        stop_token_ids=[CODE_END_TOKEN_ID],
                    )
                    
                    start_inference = time.perf_counter()
                    async for request_output in self.model.generate_stream(prompt, sampling_params):
                        all_tokens = request_output.outputs[0].token_ids

                        # Only take the new tokens generated in this step
                        new_tokens = all_tokens[vllm_pointer:]

                        if new_tokens:
                            counts = Counter(new_tokens)
                            # Get the top 3 most frequent tokens
                            top_tokens = counts.most_common(3)
                            
                            # Calculate "Burst Density"
                            # If one token is > 80% of the burst, it's a guaranteed audio artifact
                            most_common_id, freq = top_tokens[0]
                            density = freq / len(new_tokens)
                            
                            if density > 0.8 and len(new_tokens) > 7:
                                logger.warning(
                                    f"⚠️ STUCK STREAM: Token {most_common_id} occupies {density:.0%} "
                                    f"of the burst ({freq}/{len(new_tokens)} tokens)."
                                )
                            
                            # Log the IDs to see if they are SNAC (1000-12000) or Tags (13000+)
                            logger.debug(f"Burst Top Tokens: {top_tokens}")

                        # Extract codes (handles alignment and SOS/EOS)
                        snac_codes, raw_consumed = self._extract_snac_codes_streaming(new_tokens)
                        
                        # We need at least 7 tokens for a single SNAC frame
                        if len(snac_codes) >= 7:
                            await decode_queue.put((i, item, snac_codes))
                            # logger.info(f"📦 {len(snac_codes)} SNAC codes handed off.")
                                
                        vllm_pointer += raw_consumed
                        logger.debug(f"vllm_pointer {vllm_pointer}")
                    
                    end_inference = time.perf_counter()
                    logger.info(f"⏱️ Producer: Inference for item {i} took {end_inference - start_inference:.2f}s.")
                    
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
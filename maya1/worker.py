import time
import numpy as np
import logging

logger = logging.getLogger(__name__)


_worker_decoder = None

class MockSNACDecoder:
    def __init__(self, **config):
        self.config = config
        self.sample_rate = 24000

    def decode(self, tokens, use_sliding_window=False, trim_warmup=False):
        # Simulate the 75ms CPU bottleneck
        time.sleep(0.075) 
        num_samples = 8192 if use_sliding_window else 2048
        return np.zeros(num_samples * 2)

    def decode_to_bytes(self, tokens, use_sliding_window=False, trim_warmup=False):
        # Simulate the 75ms CPU bottleneck
        time.sleep(0.075) 
        num_samples = 8192 if use_sliding_window else 2048
        return b'\x00' * (num_samples * 2)

def init_worker(decoder_class, decoder_kwargs):
    global _worker_decoder
    # Initialize the class passed in
    _worker_decoder = decoder_class(**decoder_kwargs)
    # Warmup
    _worker_decoder.decode_to_bytes([1000]*7)

def worker_decode_task(tokens, use_sliding_window, trim_warmup=False, speed_up=False):
    global _worker_decoder
    if _worker_decoder is None:
        raise RuntimeError("Worker not initialized")
    audio_data = _worker_decoder.decode(tokens, use_sliding_window, trim_warmup)

    audio_int16 = (audio_data * 32767).astype(np.int16)

    if speed_up:
        # Define the silence threshold
        # For int16, a value around 300-500 is usually silent background noise
        SILENCE_THRESHOLD = 500 
        
        original_len = len(audio_int16)

        # Look at the tail of the chunk (last ~10ms / 240 samples at 24kHz)
        tail_size = 240
        if len(audio_int16) > tail_size:
            tail = audio_int16[-tail_size:]
            
            # If the average absolute amplitude is below threshold, it's 'silent'
            if np.abs(tail).mean() < SILENCE_THRESHOLD:
                # Remove the silent tail to finish this chunk faster
                audio_int16 = audio_int16[:-tail_size]
                # logger.debug("Speed-up: Truncated 10ms of silence")
            
        trimmed_samples = original_len - len(audio_int16)
        if trimmed_samples > 0:
            # Log this so you can see the "Time Gained"
            logger.debug(f"⚡ Speed-up: Gained {trimmed_samples/24:.2f}ms of lead time")

    return audio_int16.tobytes()
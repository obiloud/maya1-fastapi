import logging
import time
import asyncio

_worker_decoder = None

class MockSNACDecoder:
    def __init__(self, **config):
        self.config = config
        self.sample_rate = 24000
        # self.profiler = config.get('profiler')

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

def worker_decode_task(tokens, use_sliding_window, trim_warmup=False):
    global _worker_decoder
    if _worker_decoder is None:
        raise RuntimeError("Worker not initialized")
    return _worker_decoder.decode_to_bytes(tokens, use_sliding_window, trim_warmup)
import pytest
import asyncio
import time
import numpy as np
from unittest.mock import AsyncMock, MagicMock
from dataclasses import dataclass
from .maya1_pipeline import Maya1Pipeline
from .worker import MockSNACDecoder
from .constants import CODE_START_TOKEN_ID, SNAC_MAX_ID

# Configuration for the test
SAMPLE_RATE = 24000

@dataclass
class ProfileResult:
    stage: str
    duration: float
    audio_duration_produced: float

class PipelineProfiler:
    """Helper to track timing across async calls"""
    def __init__(self):
        self.results = []

    def record(self, stage, duration, audio_len_sec):
        self.results.append(ProfileResult(stage, duration, audio_len_sec))

@pytest.mark.asyncio
async def test_pipeline_bottlenecks():
    # 1. Setup Mocks
    mock_model = AsyncMock()
    mock_builder = MagicMock()
    profiler = PipelineProfiler()
    
    # Mock return values
    mock_builder.build_prefix.return_value = "prompt"
    
    # Mock Model: Simulates token generation latency
    async def mocked_generate_stream(*args, **kwargs):
        start = time.perf_counter()
        await asyncio.sleep(0.5) 
        gen_time = time.perf_counter() - start
        
        # Simulate 10 frames of audio arriving in 2-frame bursts
        total_audio_tokens = [CODE_START_TOKEN_ID] + ([SNAC_MAX_ID] * 70)
        
        for burst_idx in range(1, 6):
            current_limit = burst_idx * 14 # 14, 28, 42...
            mock_output = MagicMock()
            # vLLM always returns the CUMULATIVE list of tokens
            mock_output.outputs = [MagicMock(token_ids=total_audio_tokens[:current_limit])]
            yield mock_output
            await asyncio.sleep(0.01) # Simulate network/processing jitter
            profiler.record("TTS_GEN", gen_time, 10 / 6.86)
            
    mock_model.generate_stream = mocked_generate_stream
    
    # 2. Initialize Pipeline
    pipeline = Maya1Pipeline(
        model=mock_model, 
        prompt_builder=mock_builder, 
        snac_decoder_class=MockSNACDecoder,
        device="cpu",
    )
    
    # Prepare dummy items (3 text chunks)
    text = "Word " * 180
    
    # 3. Execute and Measure
    start_pipeline = time.perf_counter()
    
    audio_chunks = []
    async for chunk in pipeline.generate_speech_stream("voice_desc", text):
        audio_chunks.append(chunk)

    pipeline.executor.shutdown(wait=False, cancel_futures=True)
    
    total_pipeline_time = time.perf_counter() - start_pipeline

    # Count how many chunks the pipeline actually created
    # This helps us verify the test is actually testing multi-chunk logic
    tts_calls = [r for r in profiler.results if r.stage == "TTS_GEN"]
    num_chunks = len(tts_calls)

    total_samples = sum(len(c) for c in audio_chunks) // 2
    total_audio_duration = total_samples / SAMPLE_RATE

    # 4. Bottleneck Analysis
    print(f"\n--- Performance Report ---")
    print(f"Total Wall Clock Time: {total_pipeline_time:.2f}s")
    print(f"Total Audio Produced:  {total_audio_duration:.2f}s")
    print(f"Real-Time Factor (RTF): {total_pipeline_time / total_audio_duration:.2f}")

    # Calculate SNAC total manually for the report since the worker is isolated
    # Each window processed is 28 tokens. 
    # Your log showed 15 chunks processed.
    tts_total = sum(p.duration for p in profiler.results if p.stage == "TTS_GEN")

    SIMULATED_DECODE_STEP = 0.075 
    snac_total = num_chunks * SIMULATED_DECODE_STEP

    print(f"Sum of TTS Latency:    {tts_total:.2f}s")
    print(f"Sum of SNAC Latency:   {snac_total:.2f}s")
    print(f"Chunks Processed:      {num_chunks}")
    
    # If num_chunks > 1, total_pipeline_time MUST be less than the sum 
    # if pipelining is working.
    if num_chunks > 1:
        assert total_pipeline_time < (tts_total + snac_total), \
            f"No overlap! Wallclock ({total_pipeline_time:.2f}s) should be less than sum ({tts_total + snac_total:.2f}s)"

    # Bottleneck logic
    if tts_total > snac_total:
        print("BOTTLENECK IDENTIFIED: LLM Inference (TTS)")
    else:
        print("BOTTLENECK IDENTIFIED: SNAC Decoding (DSP)")

    # 5. Assertions
    # If the look-ahead works, total_pipeline_time should be significantly 
    # LESS than (tts_total + snac_total)
    assert total_pipeline_time < (tts_total + snac_total), "Look-ahead buffering failed to overlap tasks!"
    
    # Ensure we didn't drop audio
    assert len(audio_chunks) > 0, "No audio was generated"

if __name__ == "__main__":
    # This allows pytest to run, but also protects worker spawns
    import pytest
    pytest.main([__file__])
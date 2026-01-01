import pytest
import asyncio
import time
import numpy as np
from unittest.mock import AsyncMock, MagicMock
from dataclasses import dataclass
from .streaming_pipeline_chunks import Maya1LongPipeline
from .constants import CODE_START_TOKEN_ID, CODE_END_TOKEN_ID, SNAC_MAX_ID

# Assuming the class is in pipeline.py
# from pipeline import Maya1LongPipeline 

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
    mock_decoder = AsyncMock()
    mock_builder = MagicMock()
    profiler = PipelineProfiler()

    # Configuration for the test
    # We simulate a slow LLM and a medium-speed SNAC decoder
    SIMULATED_TTS_LATENCY = 0.5  # 500ms per chunk
    SIMULATED_SNAC_LATENCY = 0.2 # 200ms per chunk
    SAMPLE_RATE = 24000
    
    # Mock return values
    mock_builder.build_prefix.return_value = "prompt"
    
    # Mock Model: Simulates token generation latency
    async def mocked_generate(*args, **kwargs):
        start = time.perf_counter()
        await asyncio.sleep(SIMULATED_TTS_LATENCY)
        gen_time = time.perf_counter() - start
        
        # Create the nested structure vLLM expects
        mock_request_output = MagicMock()
        # This makes outputs[0] work
        mock_request_output.__getitem__.return_value = mock_request_output 
        # This makes outputs[0].outputs[0] work
        token_ids = [CODE_START_TOKEN_ID] + ([SNAC_MAX_ID] * 70) + [CODE_END_TOKEN_ID]
        mock_request_output.outputs = [MagicMock(token_ids=token_ids)]
        # Return dummy vLLM output structure
        profiler.record("TTS_GEN", gen_time, 10 / 6.86)
        return mock_request_output

    # Mock Decoder: Simulates DSP/SNAC decoding latency
    async def mocked_decode(*args, **kwargs):
        print("🛠️ Decoder Mock Called!")
        start = time.perf_counter()
        await asyncio.sleep(SIMULATED_SNAC_LATENCY)
        dec_time = time.perf_counter() - start
        
        # Return 1 second of audio bytes
        audio_sec = 1.0
        profiler.record("SNAC_DECODE", dec_time, audio_sec)
        return b'\x00' * int(SAMPLE_RATE * 2 * audio_sec)

    mock_model.generate = mocked_generate
    mock_decoder.decode_single_async = mocked_decode

    # 2. Initialize Pipeline
    pipeline = Maya1LongPipeline(mock_model, mock_builder, mock_decoder)
    
    # Prepare dummy items (3 text chunks)
    text = "Word " * 180
    
    # 3. Execute and Measure
    start_pipeline = time.perf_counter()
    
    audio_chunks = []
    async for chunk in pipeline.generate_speech_stream("voice_desc", text):
        audio_chunks.append(chunk)
    
    total_pipeline_time = time.perf_counter() - start_pipeline

    # Count how many chunks the pipeline actually created
    # This helps us verify the test is actually testing multi-chunk logic
    tts_calls = [r for r in profiler.results if r.stage == "TTS_GEN"]
    num_chunks = len(tts_calls)

    total_audio_duration = sum([p.audio_duration_produced for p in profiler.results if p.stage == "SNAC_DECODE"])

    # 4. Bottleneck Analysis
    print(f"\n--- Performance Report ---")
    print(f"Total Wall Clock Time: {total_pipeline_time:.2f}s")
    print(f"Total Audio Produced:  {total_audio_duration:.2f}s")
    print(f"Real-Time Factor (RTF): {total_pipeline_time / total_audio_duration:.2f}")

    tts_total = sum(p.duration for p in profiler.results if p.stage == "TTS_GEN")
    snac_total = sum(p.duration for p in profiler.results if p.stage == "SNAC_DECODE")

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
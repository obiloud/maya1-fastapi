import os
import io
import wave
import time
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from .model_loader import Maya1Model
from .prompt_builder import Maya1PromptBuilder
from .snac_decoder import SNACDecoder
from .pipeline import Maya1Pipeline
from .streaming_pipeline import Maya1SlidingWindowPipeline
from .constants import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_MAX_TOKENS,
    DEFAULT_REPETITION_PENALTY,
    AUDIO_SAMPLE_RATE,
    SNAC_BATCH_SIZE,
    SNAC_BATCH_TIMEOUT_MS
)

# Timeout settings (seconds)
GENERATE_TIMEOUT = 60

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Maya1 TTS API",
    description="Open source TTS inference for Maya1",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
model = None
prompt_builder = None
snac_decoder = None
pipeline = None
streaming_pipeline = None


# ============================================================================
# Startup/Shutdown
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    global model, prompt_builder, snac_decoder, pipeline, streaming_pipeline
    
    print("\n" + "="*60)
    print(" Starting Maya1 TTS API Server")
    print("="*60 + "\n")
    
    # Initialize components
    model = Maya1Model()
    prompt_builder = Maya1PromptBuilder(model.tokenizer, model)
    
    # Initialize SNAC decoder
    snac_device = os.environ.get("SNAC_DEVICE", "cuda")

    snac_decoder = SNACDecoder(device=snac_device, enable_batching=True, max_batch_size=SNAC_BATCH_SIZE, batch_timeout_ms=SNAC_BATCH_TIMEOUT_MS)
    await snac_decoder.start_batch_processor()
    
    # Initialize pipelines
    pipeline = Maya1Pipeline(model, prompt_builder, snac_decoder)
    streaming_pipeline = Maya1SlidingWindowPipeline(model, prompt_builder, snac_decoder)
    
    print("\n" + "="*60)
    print("Maya1 TTS API Server Ready")
    print("="*60 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("\nShutting down Maya1 TTS API Server")
    
    if snac_decoder and snac_decoder.is_running:
        await snac_decoder.stop_batch_processor()


# ============================================================================
# Utility Functions
# ============================================================================

def create_wav_header(sample_rate: int = 24000, channels: int = 1, bits_per_sample: int = 16, data_size: int = 0) -> bytes:
    """Create WAV file header."""
    import struct
    
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        36 + data_size,
        b'WAVE',
        b'fmt ',
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b'data',
        data_size
    )
    
    return header


# ============================================================================
# Request/Response Models
# ============================================================================

class TTSRequest(BaseModel):
    """TTS generation request."""
    description: str = Field(
        ...,
        description="Voice description (e.g., 'Male voice in their 30s with american accent')"
    )
    text: str = Field(
        ...,
        description="Text to synthesize (can include <emotion> tags)"
    )
    temperature: Optional[float] = Field(
        default=DEFAULT_TEMPERATURE,
        description="Sampling temperature"
    )
    top_p: Optional[float] = Field(
        default=DEFAULT_TOP_P,
        description="Nucleus sampling"
    )
    max_tokens: Optional[int] = Field(
        default=DEFAULT_MAX_TOKENS,
        description="Maximum tokens to generate"
    )
    repetition_penalty: Optional[float] = Field(
        default=DEFAULT_REPETITION_PENALTY,
        description="Repetition penalty"
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility",
        ge=0,
    )
    stream: bool = Field(
        default=False,
        description="Stream audio (True) or return complete WAV (False)"
    )


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Maya1 TTS API",
        "version": "1.0.0",
        "status": "running",
        "model": "Maya1-Voice (open source)",
        "endpoints": {
            "generate": "/v1/tts/generate (POST)",
            "health": "/health (GET)",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": "Maya1-Voice",
        "timestamp": time.time(),
    }


# ============================================================================
# TTS Generation Endpoint
# ============================================================================

@app.post("/v1/tts/generate")
async def generate_tts(request: TTSRequest):
    """Generate TTS audio from description and text."""
    
    try:
        # Route to streaming or non-streaming
        if request.stream:
            return await _generate_tts_streaming(
                description=request.description,
                text=request.text,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                repetition_penalty=request.repetition_penalty,
                seed=request.seed,
            )
        else:
            return await _generate_tts_complete(
                description=request.description,
                text=request.text,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                repetition_penalty=request.repetition_penalty,
                seed=request.seed,
            )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f" Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _generate_tts_complete(
    description: str,
    text: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    repetition_penalty: float,
    seed: Optional[int],
):
    """Generate complete WAV file (non-streaming)."""
    
    try:
        import asyncio
        
        # Generate audio
        audio_bytes = await asyncio.wait_for(
            pipeline.generate_speech(
                description=description,
                text=text,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                repetition_penalty=repetition_penalty,
                seed=seed,
            ),
            timeout=GENERATE_TIMEOUT
        )
        
        if audio_bytes is None:
            raise Exception("Audio generation failed")
        
        # Create WAV file
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(AUDIO_SAMPLE_RATE)
            wav_file.writeframes(audio_bytes)
        
        wav_buffer.seek(0)
        
        return StreamingResponse(
            wav_buffer,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=output.wav"}
        )
    
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Generation timeout")


async def _generate_tts_streaming(
    description: str,
    text: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    repetition_penalty: float,
    seed: Optional[int],
):
    """Generate streaming audio."""
    start_time = time.time()
    first_audio_time = None
    
    async def audio_stream_generator():
        """Generate audio stream with WAV header."""
        nonlocal first_audio_time
        
        # Send WAV header first
        yield create_wav_header(sample_rate=AUDIO_SAMPLE_RATE, channels=1, bits_per_sample=16)
        
        # Stream audio chunks
        async for audio_chunk in streaming_pipeline.generate_speech_stream(
            description=description,
            text=text,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
            seed=seed,
        ):
            if first_audio_time is None:
                first_audio_time = time.time()
                ttfb_ms = (first_audio_time - start_time) * 1000
                print(f"⏱️  TTFB: {ttfb_ms:.1f}ms")
            
            yield audio_chunk
    
    try:
        return StreamingResponse(
            audio_stream_generator(),
            media_type="audio/wav",
            headers={"Cache-Control": "no-cache"}
        )
    
    except Exception as e:
        print(f"Streaming error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# For running directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
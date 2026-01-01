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
import asyncio
import sys
import logging
import multiprocessing as mp
from contextlib import asynccontextmanager

from .model_loader import Maya1Model
from .prompt_builder import Maya1PromptBuilder
from .streaming_pipeline_chunks import Maya1LongPipeline
from .snac_decoder import SNACDecoder
from .constants import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_MAX_TOKENS,
    DEFAULT_REPETITION_PENALTY,
    AUDIO_SAMPLE_RATE,
)

# Timeout settings (seconds)
GENERATE_TIMEOUT = 60

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Imports the Cloud Logging client library
import google.cloud.logging

# Instantiates a client
client = google.cloud.logging.Client()

# Retrieves a Cloud Logging handler based on the environment
# you're running in and integrates the handler with the
# Python logging module. By default this captures all logs
# at INFO level and higher
client.setup_logging()

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger("api_v2")

# Load environment variables
load_dotenv()

if sys.platform != "win32":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # Method might already be set
        pass

# Global state
model = None
prompt_builder = None
snac_decoder = None
streaming_pipeline = None


# ============================================================================
# Startup/Shutdown
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI): # FIXED TYPO: lifspan -> lifespan
    global model, prompt_builder, snac_decoder, streaming_pipeline

    logger.info("\n" + "="*60 + "\n Starting Maya1 TTS API Server\n" + "="*60)
    
    # Load Model (vLLM Engine)
    model = Maya1Model() 

    prompt_builder = Maya1PromptBuilder(model.tokenizer, model)

    streaming_pipeline = Maya1LongPipeline(model, prompt_builder, SNACDecoder, device="cpu")

    logger.info("🚀 System fully initialized and ready for requests.")

    yield

    # Cleanup
    logger.info("Shutting down...")
    streaming_pipeline.shutdown()


# Initialize FastAPI app
app = FastAPI(
    title="Maya1 TTS API",
    description="Open source TTS inference for Maya1",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        logger.error(f" Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _generate_tts_complete(description, text, **kwargs):
    """
    Refactored to use the streaming pipeline logic for consistency,
    but collects all chunks before returning.
    """
    try:
        audio_segments = []
        # Use the same generator logic to ensure identical quality
        stream = streaming_pipeline.generate_speech_stream(
            description=description,
            text=text,
            **kwargs
        )
        
        async for chunk in stream:
            audio_segments.append(chunk)
            
        full_audio = b"".join(audio_segments)
        
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(AUDIO_SAMPLE_RATE)
            wav_file.writeframes(full_audio)
        
        wav_buffer.seek(0)
        return StreamingResponse(wav_buffer, media_type="audio/wav")
    except Exception as e:
        logger.error(f"Complete generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _generate_tts_streaming(description, text, **kwargs):
    
    async def audio_stream_generator():
        # Consider removing WAV header if the client supports raw PCM
        # yield create_wav_header(sample_rate=AUDIO_SAMPLE_RATE)

        # 1. Use anext() properly (Python 3.10+)
        stream_iter = streaming_pipeline.generate_speech_stream(
            description=description, text=text, **kwargs
        )

        while True:
            try:
                # Use a shorter timeout if the goal is strictly a keep-alive heartbeat
                # Cloud Run usually requires data every 10-30s
                audio_chunk = await asyncio.wait_for(anext(stream_iter), timeout=30.0)
                
                if audio_chunk:
                    yield audio_chunk
                    
            except StopAsyncIteration:
                logger.info("Stream completed normally.")
                break
            except asyncio.TimeoutError:
                # HEARTBEAT: We send a "Null" byte or a tiny silence.
                # WARNING: In a WAV container, sending random bytes can 
                # cause the decoder to desync.
                logger.info("💓 Heartbeat triggered.")
                # If using raw PCM, 2 bytes of 0 is a single sample of silence.
                yield b'\x00\x00' 
                continue
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                break
    
    # Add headers to help the browser handle the stream
    return StreamingResponse(
        audio_stream_generator(), 
        media_type="audio/l16; rate=24000", # Better for raw streaming
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Content-Type-Options": "nosniff"
        }
    )

# For running directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
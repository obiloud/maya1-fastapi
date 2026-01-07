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
import json
import logging
from contextlib import asynccontextmanager
import google.cloud.logging

from .logging import get_logging_config
from .model_loader import Maya1Model
from .prompt_builder import Maya1PromptBuilder
from .maya1_pipeline import Maya1Pipeline
from .snac_decoder import SNACDecoder
from .constants import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_MAX_TOKENS,
    DEFAULT_REPETITION_PENALTY,
    AUDIO_SAMPLE_RATE,
    MAX_WORDS_PER_CHUNK,
)

# Timeout settings (seconds)
GENERATE_TIMEOUT = 60

os.environ["TOKENIZERS_PARALLELISM"] = "false"

log_level_str = os.environ.get('LOG_LEVEL', 'WARNING').upper()
log_level = getattr(logging, log_level_str, logging.WARNING)
logging.config.dictConfig(get_logging_config(log_level_str))

# Initialize the Cloud Logging client
client = google.cloud.logging.Client()

# Captures all logs from the root logger at INFO level and higher
client.setup_logging(log_level=log_level)


logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

snac_device = os.environ.get('SNAC_DEVICE', 'cuda')

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

    streaming_pipeline = Maya1Pipeline(model, prompt_builder, SNACDecoder, device=snac_device)

    logger.info("🚀 System fully initialized and ready for requests.")

    yield

    # Cleanup
    logger.info("Shutting down...")
    await streaming_pipeline.shutdown()


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
    max_word_per_chunk: Optional[int] = Field(
        default=MAX_WORDS_PER_CHUNK,
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
                max_words_per_chunk=request.max_word_per_chunk
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
                max_words_per_chunk=request.max_word_per_chunk
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

# ============================================================================
# Token Streaming Endpoint
# ============================================================================

@app.post("/v1/tts/tokens")
async def generate_tts_tokens(request: TTSRequest):
    """
    Stream raw SNAC tokens to the client for browser-side ONNX decoding.
    """
    try:
        # Note: We ignore the 'stream' flag here because this endpoint 
        # is inherently streaming by design.
        return await _generate_tokens_streaming_handler(
            description=request.description,
            text=request.text,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            repetition_penalty=request.repetition_penalty,
            seed=request.seed,
            max_words_per_chunk=request.max_word_per_chunk
        )
    except Exception as e:
        logger.error(f"Token Endpoint Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _generate_tokens_streaming_handler(description, text, **kwargs):
    """
    Wraps the pipeline generator to yield JSON-formatted token packets.
    """
    async def token_stream_generator():
        stream_iter = streaming_pipeline.generate_token_stream(
            description=description, 
            text=text, 
            **kwargs
        )

        while True:
            try:
                # Use a timeout to detect stalls and send heartbeats
                packet = await asyncio.wait_for(anext(stream_iter), timeout=15.0)
                
                if packet:
                    # Yielding as a JSON line (NDJSON format) is easiest for browsers to parse
                    yield json.dumps(packet) + "\n"
                    
            except StopAsyncIteration:
                logger.info("Token stream completed normally.")
                break
            except asyncio.TimeoutError:
                # HEARTBEAT: Keeps the HTTP connection alive during long LLM inference
                logger.info("💓 Token Stream Heartbeat triggered.")
                yield json.dumps({"type": "heartbeat", "time": time.time()}) + "\n"
                continue
            except Exception as e:
                logger.error(f"Token streaming error: {e}", exc_info=True)
                yield json.dumps({"type": "error", "message": str(e)}) + "\n"
                break
    
    return StreamingResponse(
        token_stream_generator(), 
        media_type="application/x-ndjson", 
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no" # Essential for Nginx/Cloud Run proxying
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
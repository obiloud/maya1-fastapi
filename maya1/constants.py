"""
Maya1 Constants
Token IDs and special tokens used in the model.
Matches training configuration exactly.
"""

# Special control tokens
SOH_ID = 128259  # Start of Human turn
EOH_ID = 128260  # End of Human turn
SOA_ID = 128261  # Start of AI turn
EOA_ID = 128262  # End of AI turn (not used in maya1)
PAD_ID = 128263  # Padding token

# Text tokens
BOS_ID = 128000  # Begin of sequence (Llama BOS)
TEXT_EOT_ID = 128009  # End of text (appears in prefix, not a stop token!)

# Audio tokens
CODE_START_TOKEN_ID = 128257  # SOS - Start of Speech
CODE_END_TOKEN_ID = 128258   # EOS - End of Speech (audio stop token)
CODE_TOKEN_OFFSET = 128266   # Start of SNAC codes

# SNAC token range
SNAC_MIN_ID = 128266
SNAC_MAX_ID = 156937  # 128266 + (7 * 4096) - 1

# Stop tokens for generation
# CRITICAL: Only use CODE_END_TOKEN_ID (128258) for audio generation
# TEXT_EOT_ID (128009) appears in prefix and should NOT stop generation
TRAINING_STOP_TOKEN_IDS = [CODE_END_TOKEN_ID]  # [128258]
ALL_POSSIBLE_STOP_TOKENS = [TEXT_EOT_ID, CODE_END_TOKEN_ID]  # For reference only

# 20 Extended Emotion Tags (must be single tokens)
ALL_EMOTION_TAGS = [
    '<angry>',
    '<appalled>',
    '<chuckle>',
    '<cry>',
    '<curious>',
    '<disappointed>',
    '<excited>',
    '<exhale>',
    '<gasp>',
    '<giggle>',
    '<gulp>',
    '<laugh>',
    '<laugh_harder>',
    '<mischievous>',
    '<sarcastic>',
    '<scream>',
    '<sigh>',
    '<sing>',
    '<snort>',
    '<whisper>',
]

# Model configuration
DEFAULT_MODEL_PATH = "maya-research/maya1"
DEFAULT_CHECKPOINT = "checkpoint-25000"
DEFAULT_MAX_MODEL_LEN = 4096 # 8192

# SNAC configuration
SNAC_MODEL_NAME = "hubertsiuzdak/snac_24khz"
SNAC_SAMPLE_RATE = 24000
SNAC_TOKENS_PER_FRAME = 7
SNAC_LEVELS = 3

# Audio configuration
AUDIO_SAMPLE_RATE = 24000
AUDIO_CHANNELS = 1
AUDIO_BITS_PER_SAMPLE = 16

# Generation defaults
DEFAULT_TEMPERATURE = 0.4  # Lower temp for more stable generation
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_TOKENS = 2048  # Reasonable default for most use cases
DEFAULT_MIN_TOKENS = 28  # At least 4 SNAC frames
DEFAULT_REPETITION_PENALTY = 1.1
DEFAULT_SEED = None  # None = random, set integer for reproducibility

# IMPORTANT: Emotion tags consume audio time!
# <laugh> = ~4-6 seconds (~300-400 tokens)
# <excited>, <chuckle> = ~1-2 seconds (~50-150 tokens)

# Recommended max_tokens by use case:
# - Short phrases (< 10 words): 150-250 tokens (~3-5s)
# - Medium text (10-30 words): 250-500 tokens (~5-10s)
# - Long text (30+ words): 500-1500 tokens (~10-30s)
# - Very long text: 1500-2000 tokens (~30-42s)
# Note: 1 second ≈ 48 tokens (7 tokens/frame * 6.86 frames/sec)

# Streaming configuration
STREAM_BUFFER_SIZE = 28  # 4 frames (process every 28 tokens)
SNAC_BATCH_SIZE = 64
SNAC_BATCH_TIMEOUT_MS = 15
import numpy as np
import re

def get_word_count(text: str) -> int:
    return len(text.split())

def generate_silent_bytes(duration_sec: float) -> bytes:
    """Generates pure digital silence (zeros) for the specified duration."""
    num_samples = int(duration_sec * 24000)
    # create array of zeros (int16)
    silent_array = np.random.normal(0, 5, num_samples).astype(np.int16)
    return silent_array.tobytes()

def parse_pause_tags(text: str):
    """
    Splits text by [PAUSE:X] tags.
    Returns a list of mixed types: [str, float, str, float...]
    where float represents silence duration in seconds.
    """
    # Regex to find [PAUSE:2] or [PAUSE:0.5]
    pattern = r'\[PAUSE:\s?(\d+(?:\.\d+)?)\]'
    parts = re.split(pattern, text)
    
    parsed_sequence = []
    
    # re.split returns [text, duration, text, duration...]
    # We need to reconstruct this carefully
    i = 0
    while i < len(parts):
        text_segment = parts[i].strip()
        if text_segment:
            parsed_sequence.append(text_segment)
        
        # If there is a next part, it is the captured group (duration)
        if i + 1 < len(parts):
            try:
                duration = float(parts[i+1])
                parsed_sequence.append(duration)
            except ValueError:
                pass # Should not happen with strict regex
        i += 2
        
    return parsed_sequence

def recursive_word_chunker(text: str, max_words: int) -> list[str]:
    # (Existing chunker logic - abbreviated for brevity)
    # Note: This runs on the text segments *between* pauses
    delimiters = ["\n\n", r"(?<=[.!?])\s+", r"(?<=[,;])\s+", " "]
    chunks = []
    
    def split_recursive(text_segment, delimiter_idx):
        if get_word_count(text_segment) <= max_words:
            if text_segment.strip(): chunks.append(text_segment.strip())
            return
        if delimiter_idx >= len(delimiters): # Hard split fallback
            words = text_segment.split()
            current = []
            for w in words:
                if len(current) + 1 > max_words:
                    chunks.append(" ".join(current)); current = [w]
                else: current.append(w)
            if current: chunks.append(" ".join(current))
            return
            
        delimiter = delimiters[delimiter_idx]
        parts = re.split(delimiter, text_segment)
        current_accumulation = ""
        for part in parts:
            part = part.strip()
            if not part: continue
            potential = (current_accumulation + " " + part).strip() if current_accumulation else part
            if get_word_count(potential) <= max_words:
                current_accumulation = potential
            else:
                if current_accumulation: chunks.append(current_accumulation); current_accumulation = ""
                if get_word_count(part) <= max_words: current_accumulation = part
                else: split_recursive(part, delimiter_idx + 1)
        if current_accumulation: chunks.append(current_accumulation)

    split_recursive(text, 0)
    
    # Merge optimization
    final = []
    if chunks:
        cur = chunks[0]
        for nxt in chunks[1:]:
            if get_word_count(cur) + get_word_count(nxt) <= max_words: cur += " " + nxt
            else: final.append(cur); cur = nxt
        final.append(cur)
    return final

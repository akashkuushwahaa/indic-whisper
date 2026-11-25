"""Text normalization shared by data prep, training labels and evaluation.

Do NOT use Whisper's BasicTextNormalizer for Indic text: it strips combining
marks (Unicode category M), which deletes vowel signs (matras) and turns words
into consonant skeletons, making WER meaningless.
"""

import unicodedata

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿"), None)

# Native digits -> ASCII, so "२०२४" and "2024" score as the same token.
_DIGITS = {
    base + i: ord("0") + i
    for base in (0x0966, 0x0AE6)  # Devanagari, Gujarati
    for i in range(10)
}


def normalize(text: str) -> str:
    """NFC, drop zero-width chars, ASCII digits, punctuation/symbols -> space
    (danda included), lowercase Latin, collapse whitespace."""
    text = unicodedata.normalize("NFC", text).translate(_ZERO_WIDTH).translate(_DIGITS)
    text = "".join(" " if unicodedata.category(ch)[0] in "PS" else ch for ch in text)
    return " ".join(text.lower().split())

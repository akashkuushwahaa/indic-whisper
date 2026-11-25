"""Language config, per-clip filters, train/val split and output schema."""

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa


@dataclass(frozen=True)
class Lang:
    name: str  # Shrutilipi config name
    code: str  # Whisper language code
    ranges: tuple[tuple[int, int], ...]  # Unicode blocks of the native script


LANGS = {
    "gujarati": Lang("gujarati", "gu", ((0x0A80, 0x0AFF),)),
    "hindi": Lang("hindi", "hi", ((0x0900, 0x097F), (0xA8E0, 0xA8FF))),
}


@dataclass
class FilterConfig:
    min_dur: float = 1.0
    max_dur: float = 30.0  # Whisper's input window
    min_script_ratio: float = 0.85
    # Normalized non-space chars per second. Real Shrutilipi gu/hi: p1 ~9, p99 ~17.5.
    min_cps: float = 6.0
    max_cps: float = 22.0
    max_label_tokens: int = 440  # Whisper decoder limit is 448 incl. special tokens


def script_ratio(text: str, lang: Lang) -> float:
    """Share of letters/marks that belong to the language's native script."""
    letters = [c for c in text if unicodedata.category(c)[0] in "LM"]
    if not letters:
        return 0.0
    inside = sum(any(lo <= ord(c) <= hi for lo, hi in lang.ranges) for c in letters)
    return inside / len(letters)


def chars_per_sec(norm_text: str, duration: float) -> float:
    return len(norm_text.replace(" ", "")) / duration if duration > 0 else 0.0


def reject_reason(norm_text: str, duration: float, n_tokens: int, lang: Lang,
                  cfg: FilterConfig) -> str | None:
    """Return why a clip is dropped, or None to keep it. Cheap checks first."""
    if not norm_text:
        return "empty_text"
    if duration < cfg.min_dur:
        return "too_short"
    if duration > cfg.max_dur:
        return "too_long"
    if script_ratio(norm_text, lang) < cfg.min_script_ratio:
        return "wrong_script"
    cps = chars_per_sec(norm_text, duration)
    if cps < cfg.min_cps:
        return "cps_low"  # transcript too short for the audio: likely misaligned
    if cps > cfg.max_cps:
        return "cps_high"  # transcript too long for the audio: likely misaligned
    if n_tokens > cfg.max_label_tokens:
        return "too_many_tokens"
    return None


_SEGMENT_SUFFIX = re.compile(r"[_-]\d+$")


def group_key(path: str) -> str:
    """Clips cut from the same bulletin should land in the same split. Strip the
    extension and a trailing segment index ("bulletin_0042" -> "bulletin").
    Verify against real paths with `iw-inspect` before trusting it."""
    stem = Path(path).with_suffix("").as_posix()
    return _SEGMENT_SUFFIX.sub("", stem)


def assign_split(path: str, val_fraction: float) -> str:
    h = int(hashlib.sha1(group_key(path).encode()).hexdigest()[:8], 16)
    return "val" if h / 0xFFFFFFFF < val_fraction else "train"


# datasets-compatible schema: `audio` decodes as an Audio feature when loaded.
SCHEMA = pa.schema(
    [
        ("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
        ("text", pa.string()),
        ("text_raw", pa.string()),
        ("duration", pa.float32()),
        ("lang", pa.string()),
        ("n_tokens", pa.int32()),
    ],
    metadata={
        "huggingface": json.dumps({"info": {"features": {
            "audio": {"_type": "Audio", "sampling_rate": 16000},
            "text": {"dtype": "string", "_type": "Value"},
            "text_raw": {"dtype": "string", "_type": "Value"},
            "duration": {"dtype": "float32", "_type": "Value"},
            "lang": {"dtype": "string", "_type": "Value"},
            "n_tokens": {"dtype": "int32", "_type": "Value"},
        }}})
    },
)


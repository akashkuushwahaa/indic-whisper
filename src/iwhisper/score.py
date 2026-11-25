"""Phase 1b: score every kept clip with a reference model (IndicWhisper) and
write per-shard sidecars. Training later drops clips above a threshold.

Two methods:
  nll (default)  teacher-forced mean per-token loss of the label given the audio.
                 One forward pass per batch, ~10x+ faster than generating.
                 Misaligned pairs get a high loss.
  cer            transcribe, then CER vs the label. Slow (autoregressive; Gujarati
                 needs ~2.7 tokens/char) but interpretable; used to calibrate nll.

    iw-score --lang gujarati --model models/whisper-medium-gu_alldata_multigpu
    iw-score ... --device cuda:1 --part 1 --num-parts 2     # second GPU

Output: <data>/<lang>/scores_<method>/<split>/<shard>.parquet ("scores/" for cer,
kept for compatibility). Resumable: finished shards are skipped.
"""

import argparse
import io
import sys
import time
from pathlib import Path
from typing import Callable

import jiwer
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

from .data import LANGS
from .normalize import normalize

MAX_LABEL_TOKENS = 448
PROMPT_TOKENS = 4  # <|startoftranscript|><|lang|><|transcribe|><|notimestamps|>
BASE_BY_WIDTH = {384: "openai/whisper-tiny", 512: "openai/whisper-base", 768: "openai/whisper-small",
                 1024: "openai/whisper-medium", 1280: "openai/whisper-large-v2"}
SCORE_DIRS = {"cer": "scores", "nll": "scores_nll"}


def cer(ref: str, hyp: str) -> float:
    if not hyp:
        return 1.0
    return float(jiwer.cer(ref, hyp))


class Scorer:
    def __init__(self, model: str, lang_code: str, device: str):
        import torch
        from transformers import GenerationConfig, WhisperForConditionalGeneration, WhisperProcessor

        self.torch, self.device = torch, device
        self.dtype = torch.float16 if device.startswith(("cuda", "mps")) else torch.float32
        self.proc = WhisperProcessor.from_pretrained(model)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model, torch_dtype=self.dtype).to(device).eval()
        # IndicWhisper ships a transformers-4.27 generation config without
        # lang_to_id, so generate(language=...) can't force the language and
        # falls back to detection. Token ids are the same across multilingual
        # Whisper sizes, so borrow the config from the matching OpenAI model.
        # Only when missing: large-v3 models shift special-token ids by one.
        if not getattr(self.model.generation_config, "lang_to_id", None):
            self.model.generation_config = GenerationConfig.from_pretrained(
                BASE_BY_WIDTH[self.model.config.d_model])
        self.model.config.forced_decoder_ids = None
        self.model.generation_config.forced_decoder_ids = None
        self.lang_code = lang_code
        tok = self.proc.tokenizer
        self.prompt = [tok.convert_tokens_to_ids(t) for t in
                       ("<|startoftranscript|>", f"<|{lang_code}|>", "<|transcribe|>", "<|notimestamps|>")]
        self.eot = tok.convert_tokens_to_ids("<|endoftext|>")

    def _features(self, audios):
        feats = self.proc.feature_extractor(audios, sampling_rate=16000, return_tensors="pt").input_features
        return feats.to(self.device, self.dtype)

    def transcribe(self, audios: list[np.ndarray], max_new_tokens: int) -> list[str]:
        with self.torch.inference_mode():
            out = self.model.generate(
                self._features(audios), language=self.lang_code, task="transcribe",
                max_new_tokens=min(max_new_tokens, MAX_LABEL_TOKENS - PROMPT_TOKENS),
                num_beams=1, do_sample=False)
        return self.proc.tokenizer.batch_decode(out, skip_special_tokens=True)

    def nll(self, audios: list[np.ndarray], texts: list[str]) -> list[float]:
        """Mean per-token cross-entropy of `prompt + text + <eot>` given the audio.
        Prompt tokens are inputs only; text tokens and <eot> are scored."""
        torch = self.torch
        ids = self.proc.tokenizer(texts, add_special_tokens=False).input_ids
        seqs = [(self.prompt + x)[: MAX_LABEL_TOKENS - 1] + [self.eot] for x in ids]
        width = max(map(len, seqs)) - 1
        inp = torch.full((len(seqs), width), self.eot, dtype=torch.long)
        tgt = torch.full((len(seqs), width), -100, dtype=torch.long)
        for i, s in enumerate(seqs):
            inp[i, : len(s) - 1] = torch.tensor(s[:-1])
            tgt[i, PROMPT_TOKENS - 1: len(s) - 1] = torch.tensor(s[PROMPT_TOKENS:])
        # Right padding is safe without a mask: the decoder is causal.
        with torch.inference_mode():
            logits = self.model(input_features=self._features(audios),
                                decoder_input_ids=inp.to(self.device)).logits
            loss = torch.nn.functional.cross_entropy(
                logits.float().transpose(1, 2), tgt.to(self.device), ignore_index=-100, reduction="none")
            mask = (tgt != -100).to(self.device)
            return ((loss * mask).sum(1) / mask.sum(1)).tolist()


BatchFn = Callable[[list[dict], list[np.ndarray]], list[dict]]


def cer_batch(scorer: Scorer) -> BatchFn:
    def fn(rows, audios):
        # Allow the hypothesis to run ~25% past the longest label in the batch.
        max_new = int(max(r["n_tokens"] for r in rows) * 1.25) + 8
        hyps = [normalize(h) for h in scorer.transcribe(audios, max_new)]
        return [{"cer": cer(r["text"], h), "hyp": h} for r, h in zip(rows, hyps)]
    return fn


def nll_batch(scorer: Scorer) -> BatchFn:
    def fn(rows, audios):
        return [{"nll": x} for x in scorer.nll(audios, [r["text_raw"] for r in rows])]
    return fn


def score_shard(src: Path, dst: Path, score_batch: BatchFn, batch_size: int) -> tuple[int, float]:
    t = pq.read_table(src, columns=["audio", "text", "text_raw", "duration", "n_tokens"]).to_pylist()
    # Sort by label length so a batch has similar decode lengths / padding.
    order = sorted(range(len(t)), key=lambda i: t[i]["n_tokens"])
    results: list[dict] = [{}] * len(t)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        audios = [sf.read(io.BytesIO(t[i]["audio"]["bytes"]), dtype="float32")[0] for i in idx]
        for i, res in zip(idx, score_batch([t[i] for i in idx], audios)):
            results[i] = res
    cols = {"path": [r["audio"]["path"] for r in t]}
    for key in results[0] if results else []:
        cols[key] = [r[key] for r in results]
    table = pa.table(cols)
    for name in ("cer", "nll"):
        if name in cols:
            table = table.set_column(table.schema.get_field_index(name), name,
                                     pa.array(cols[name], pa.float32()))
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    pq.write_table(table, tmp)
    tmp.rename(dst)  # atomic: a crash never leaves a half-written sidecar
    return len(t), sum(r["duration"] for r in t)


def shards_for(lang_dir: Path, part: int, num_parts: int, method: str = "nll") -> list[tuple[Path, Path]]:
    srcs = sorted(lang_dir.glob("train/*.parquet")) + sorted(lang_dir.glob("val/*.parquet"))
    pairs = [(s, lang_dir / SCORE_DIRS[method] / s.parent.name / s.name) for s in srcs]
    return [p for i, p in enumerate(pairs) if i % num_parts == part]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lang", required=True, choices=sorted(LANGS))
    p.add_argument("--model", required=True)
    p.add_argument("--method", choices=sorted(SCORE_DIRS), default="nll")
    p.add_argument("--data", type=Path, default=Path("data/shrutilipi"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--part", type=int, default=0)
    p.add_argument("--num-parts", type=int, default=1)
    p.add_argument("--max-shards", type=int, help="stop after this many shards")
    a = p.parse_args()

    lang = LANGS[a.lang]
    todo = [(s, d) for s, d in shards_for(a.data / lang.name, a.part, a.num_parts, a.method)
            if not d.exists()]
    todo = todo[: a.max_shards] if a.max_shards else todo
    print(f"[{lang.name}] {len(todo)} shards to score ({a.method}) on {a.device}", file=sys.stderr, flush=True)
    if not todo:
        return
    scorer = Scorer(a.model, lang.code, a.device)
    batch_fn = (nll_batch if a.method == "nll" else cer_batch)(scorer)
    total_audio, t_start = 0.0, time.time()
    for k, (src, dst) in enumerate(todo, 1):
        t0 = time.time()
        n, secs = score_shard(src, dst, batch_fn, a.batch_size)
        total_audio += secs
        el = time.time() - t_start
        print(f"[{lang.name}] {k}/{len(todo)} {src.parent.name}/{src.name}: {n} clips "
              f"{time.time() - t0:.0f}s | {total_audio / el:.0f}x realtime, "
              f"ETA {(len(todo) - k) * el / k / 3600:.1f} h", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()

"""Phase 2: fine-tune Whisper on the prepared Shrutilipi shards (Gujarati + Hindi).

    iw-train --preset rtx4050-6gb --out runs/small-gu-hi          # Windows/Linux laptop, 6 GB
    iw-train --preset t4 --out runs/small-gu-hi                   # Colab/Kaggle T4
    iw-train --preset smoke --out runs/smoke                      # 10 steps of whisper-tiny on CPU
    iw-train ... --resume                                         # continue from the last checkpoint

Data streams from <data>/<lang>/train/*.parquet: shards are read in random
order and the two languages are mixed by --lang-weights, through a shuffle
buffer. Every --eval-steps the model transcribes a fixed sample of each
language's val split (language forced) and logs WER/CER; the checkpoint with
the lowest mean WER is kept as best. Checkpoints include the processor, so
`iw-eval --model runs/<run>/checkpoint-N` works directly.
"""

import argparse
import io
import json
import random
import sys
from pathlib import Path

import jiwer
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch

from .bench import max_new_tokens
from .data import LANGS
from .normalize import normalize

PRESETS = {
    # 6 GB laptop GPU (Ada): bf16, 8-bit AdamW, gradient checkpointing.
    "rtx4050-6gb": dict(model="openai/whisper-small", batch_size=8, grad_accum=4, precision="bf16",
                        optim="adamw_bnb_8bit", grad_ckpt=True, workers=4, eval_batch_size=16),
    # 16 GB T4 (Turing: no bf16).
    "t4": dict(model="openai/whisper-small", batch_size=16, grad_accum=2, precision="fp16",
               optim="adamw_bnb_8bit", grad_ckpt=True, workers=2, eval_batch_size=32),
    # Wiring check on any machine.
    "smoke": dict(model="openai/whisper-tiny", batch_size=2, grad_accum=1, precision="fp32",
                  optim="adamw_torch", grad_ckpt=False, workers=0, eval_batch_size=4,
                  max_steps=10, eval_steps=5, save_steps=5, warmup_steps=2, val_per_lang=4,
                  logging_steps=1),
}
DEFAULTS = dict(max_steps=5000, eval_steps=500, save_steps=500, warmup_steps=500, lr=1e-5,
                val_per_lang=200, logging_steps=25)


# ---------- data ----------

def label_ids(tokenizer, lang_code: str, text: str) -> list[int]:
    """Whisper target WITHOUT the leading <|startoftranscript|>: the model adds it
    as decoder_start when shifting labels right."""
    prefix = [tokenizer.convert_tokens_to_ids(t) for t in
              (f"<|{lang_code}|>", "<|transcribe|>", "<|notimestamps|>")]
    eot = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    return prefix + tokenizer(text, add_special_tokens=False).input_ids + [eot]


def _rows(shard: Path, columns: list[str]):
    for batch in pq.ParquetFile(shard).iter_batches(batch_size=256, columns=columns):
        yield from batch.to_pylist()


def mixed_stream(shards: dict[str, list[Path]], weights: dict[str, float], seed: int,
                 buffer: int = 1000, columns=("audio", "text", "text_raw", "lang")):
    """Endless stream: pick a language by weight, take its next row; each language
    cycles through its shards in a fresh random order every pass."""
    rng = random.Random(seed)
    langs = [l for l in shards if shards[l]]
    w = [weights[l] for l in langs]

    def cycle(lang):
        while True:
            order = shards[lang][:]
            rng.shuffle(order)
            for s in order:
                yield from _rows(s, list(columns))

    its = {l: cycle(l) for l in langs}
    buf = []
    while True:
        buf.append(next(its[rng.choices(langs, w)[0]]))
        if len(buf) >= buffer:
            yield buf.pop(rng.randrange(len(buf)))


class ShardStream(torch.utils.data.IterableDataset):
    """Train-shard stream. Module-level so it pickles into DataLoader workers
    under spawn (the only start method on Windows)."""

    def __init__(self, shards, weights, seed, target):
        self.shards, self.weights, self.seed, self.target = shards, weights, seed, target

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid, n = (info.id, info.num_workers) if info else (0, 1)
        # Each worker owns a disjoint slice of shards and its own RNG.
        mine = {l: s[wid::n] or s for l, s in self.shards.items()}
        for row in mixed_stream(mine, self.weights, self.seed + 1000 * wid):
            yield {"audio": row["audio"]["bytes"], "lang": row["lang"],
                   "text": row["text_raw"] if self.target == "raw" else row["text"]}


class Collator:
    def __init__(self, processor):
        self.fe, self.tok = processor.feature_extractor, processor.tokenizer

    def __call__(self, items):
        audios = [sf.read(io.BytesIO(x["audio"]), dtype="float32")[0] for x in items]
        feats = self.fe(audios, sampling_rate=16000, return_tensors="pt").input_features
        labels = [label_ids(self.tok, x["lang"], x["text"]) for x in items]
        width = max(map(len, labels))
        lab = torch.full((len(labels), width), -100, dtype=torch.long)
        for i, ids in enumerate(labels):
            lab[i, : len(ids)] = torch.tensor(ids)
        return {"input_features": feats, "labels": lab}


def load_val(data: Path, langs: list[str], per_lang: int, seed: int) -> dict[str, list[dict]]:
    out = {}
    for lang in langs:
        rows = [r for s in sorted((data / lang / "val").glob("*.parquet"))
                for r in _rows(s, ["audio", "text", "duration"])]
        random.Random(seed).shuffle(rows)
        out[lang] = rows[:per_lang]
    return out


# ---------- training ----------

def build_args(a) -> dict:
    cfg = {**DEFAULTS, **PRESETS[a.preset]}
    for k, v in vars(a).items():  # explicit CLI flags win over the preset
        if v is not None and k in cfg:
            cfg[k] = v
    return cfg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", required=True, choices=sorted(PRESETS))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path("data/shrutilipi"))
    p.add_argument("--langs", nargs="+", default=["gujarati", "hindi"], choices=sorted(LANGS))
    p.add_argument("--lang-weights", nargs="+", default=["gujarati=0.5", "hindi=0.5"])
    p.add_argument("--target", choices=["raw", "norm"], default="raw",
                   help="train on text with punctuation (raw) or normalized text")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    for k in ("model", "optim", "precision"):
        p.add_argument(f"--{k}")
    for k in ("batch_size", "grad_accum", "max_steps", "eval_steps", "save_steps", "warmup_steps",
              "val_per_lang", "workers", "eval_batch_size", "logging_steps"):
        p.add_argument(f"--{k.replace('_', '-')}", dest=k, type=int)
    p.add_argument("--lr", type=float)
    a = p.parse_args()
    cfg = build_args(a)
    weights = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in a.lang_weights}

    from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration,
                              WhisperProcessor)

    shards = {l: sorted((a.data / l / "train").glob("*.parquet")) for l in a.langs}
    if not all(shards.values()):
        sys.exit(f"no train shards under {a.data}/<lang>/train for {[l for l in shards if not shards[l]]}; "
                 "run iw-prepare first")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if a.preset == "smoke":
        device = "cpu"
    elif device != "cuda":
        sys.exit(f"preset {a.preset} needs a CUDA GPU (found {device}); check torch.cuda.is_available()")

    processor = WhisperProcessor.from_pretrained(cfg["model"])
    model = WhisperForConditionalGeneration.from_pretrained(cfg["model"])
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False  # incompatible with gradient checkpointing; generate re-enables it
    # Light SpecAugment: masks a few time/frequency bands of the log-mel input.
    model.config.apply_spec_augment, model.config.mask_time_prob = True, 0.05

    val = load_val(a.data, a.langs, cfg["val_per_lang"], a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "run_config.json").write_text(json.dumps(
        {**cfg, "langs": a.langs, "lang_weights": weights, "target": a.target, "seed": a.seed,
         "shards": {l: len(s) for l, s in shards.items()}, "val": {l: len(v) for l, v in val.items()}},
        indent=1))

    class WhisperTrainer(Seq2SeqTrainer):
        """Eval = forced-language greedy transcription of the val sample, per language."""

        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
            m = self.model
            m.eval()
            metrics = {}
            for lang, rows in val.items():
                refs, hyps = [], []
                for s in range(0, len(rows), cfg["eval_batch_size"]):
                    batch = rows[s:s + cfg["eval_batch_size"]]
                    audios = [sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")[0] for r in batch]
                    feats = processor.feature_extractor(audios, sampling_rate=16000, return_tensors="pt")
                    feats = feats.input_features.to(m.device, next(m.parameters()).dtype)
                    with torch.inference_mode(), torch.autocast(m.device.type, enabled=False):
                        out = m.generate(feats, language=LANGS[lang].code, task="transcribe", use_cache=True,
                                         max_new_tokens=max_new_tokens(lang, [r["duration"] for r in batch]))
                    hyps += [normalize(h) for h in processor.tokenizer.batch_decode(out, skip_special_tokens=True)]
                    refs += [r["text"] for r in batch]
                code = LANGS[lang].code
                metrics[f"{metric_key_prefix}_wer_{code}"] = round(100 * jiwer.wer(refs, hyps), 2)
                metrics[f"{metric_key_prefix}_cer_{code}"] = round(100 * jiwer.cer(refs, hyps), 2)
                print(f"  [{lang}] e.g. ref: {refs[0][:70]}\n  [{lang}]      hyp: {hyps[0][:70]}", flush=True)
            metrics[f"{metric_key_prefix}_wer"] = round(float(np.mean(
                [v for k, v in metrics.items() if "_wer_" in k])), 2)
            m.train()
            self.log(metrics)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
            return metrics

    args = Seq2SeqTrainingArguments(
        output_dir=str(a.out), max_steps=cfg["max_steps"], learning_rate=cfg["lr"],
        warmup_steps=cfg["warmup_steps"], lr_scheduler_type="linear",
        per_device_train_batch_size=cfg["batch_size"], gradient_accumulation_steps=cfg["grad_accum"],
        bf16=cfg["precision"] == "bf16", fp16=cfg["precision"] == "fp16",
        gradient_checkpointing=cfg["grad_ckpt"], optim=cfg["optim"],
        eval_strategy="steps", eval_steps=cfg["eval_steps"], save_strategy="steps",
        save_steps=cfg["save_steps"], save_total_limit=3, load_best_model_at_end=True,
        metric_for_best_model="wer", greater_is_better=False,
        logging_steps=cfg["logging_steps"], report_to=["tensorboard"] if a.preset != "smoke" else [],
        dataloader_num_workers=cfg["workers"], dataloader_persistent_workers=cfg["workers"] > 0,
        remove_unused_columns=False, seed=a.seed, use_cpu=a.preset == "smoke",
    )
    trainer = WhisperTrainer(
        model=model, args=args, train_dataset=ShardStream(shards, weights, a.seed, a.target),
        eval_dataset=[0],  # placeholder: evaluate() uses `val`
        data_collator=Collator(processor), processing_class=processor)
    resume = a.resume and any(a.out.glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=True if resume else None)
    trainer.save_model(str(a.out / "best"))
    processor.save_pretrained(str(a.out / "best"))
    print(f"best checkpoint: {trainer.state.best_model_checkpoint} "
          f"(val WER {trainer.state.best_metric}) -> saved to {a.out / 'best'}", flush=True)


if __name__ == "__main__":
    main()

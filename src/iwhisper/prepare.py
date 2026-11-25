"""Download Shrutilipi parquet files for a language, filter noisy clips, split
train/val by bulletin, and write datasets-compatible parquet shards.

Works file by file and is resumable: each finished source file leaves a marker
in <out>/<lang>/_done/, and a rerun skips it. Raw downloads are deleted after
processing unless --keep-raw.

    iw-prepare --lang gujarati
    iw-prepare --lang hindi --max-hours 400      # random subset of files
"""

import argparse
import json
import random
import statistics
import sys
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from .data import LANGS, SCHEMA, FilterConfig, Lang, assign_split, chars_per_sec, reject_reason
from .normalize import normalize

DATASET = "ai4bharat/Shrutilipi"
# Whisper label = <|startoftranscript|><|lang|><|transcribe|><|notimestamps|> text <|endoftext|>
SPECIAL_TOKENS = 5


def list_source_files(lang: Lang) -> list[str]:
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(DATASET, repo_type="dataset")
    return sorted(f for f in files if f.startswith(f"{lang.name}/") and f.endswith(".parquet"))


def hf_fetch(raw_dir: Path) -> Callable[[str], Path]:
    from huggingface_hub import hf_hub_download

    return lambda f: Path(hf_hub_download(DATASET, f, repo_type="dataset", local_dir=raw_dir))


def whisper_token_counter(model: str = "openai/whisper-small") -> Callable[[str], int]:
    from transformers import WhisperTokenizerFast

    tok = WhisperTokenizerFast.from_pretrained(model)
    return lambda text: len(tok(text, add_special_tokens=False).input_ids) + SPECIAL_TOKENS


def clean_raw(text: str) -> str:
    """Light cleanup that keeps punctuation (a possible training target)."""
    text = unicodedata.normalize("NFC", text).replace("​", "").replace("﻿", "")
    return " ".join(text.split())


def filter_rows(rows: Iterable[dict], lang: Lang, cfg: FilterConfig,
                count_tokens: Callable[[str], int], val_fraction: float,
                seen: set) -> tuple[dict[str, list], Counter]:
    """Apply filters to source rows; return kept rows per split and drop counts."""
    out, reasons = {"train": [], "val": []}, Counter()
    for row in rows:
        audio = row["audio_filepath"]
        raw = clean_raw(row["text"] or "")
        norm = normalize(raw)
        dur = float(row["duration"])
        n_tok = 0

        if not audio or not audio.get("bytes"):
            reason = "no_audio"
        else:
            reason = reject_reason(norm, dur, 0, lang, cfg)
            if reason is None:
                key = (norm, round(dur, 1))
                if key in seen:
                    reason = "duplicate"
                else:
                    seen.add(key)
                    n_tok = count_tokens(raw)  # raw is the longer label; check it
                    reason = reject_reason(norm, dur, n_tok, lang, cfg)

        if reason:
            reasons[reason] += 1
            continue
        split = assign_split(audio.get("path") or "", val_fraction)
        out[split].append({
            "audio": {"bytes": audio["bytes"], "path": audio.get("path")},
            "text": norm, "text_raw": raw, "duration": dur,
            "lang": lang.code, "n_tokens": n_tok,
        })
    return out, reasons


def _pct(values: list[float]) -> dict:
    if len(values) < 2:
        return {}
    q = statistics.quantiles(values, n=100)
    return {"p1": round(q[0], 2), "p5": round(q[4], 2), "p50": round(q[49], 2),
            "p95": round(q[94], 2), "p99": round(q[98], 2)}


def _read_kept(lang_dir: Path, columns: list[str]) -> pa.Table | None:
    shards = sorted(lang_dir.glob("[tv]*/*.parquet"))
    return pa.concat_tables([pq.read_table(s, columns=columns) for s in shards]) if shards else None


def summarize(lang_dir: Path, cfg: FilterConfig, val_fraction: float) -> dict:
    done = [json.loads(p.read_text()) for p in sorted((lang_dir / "_done").glob("*.json"))]
    dropped, hours = Counter(), Counter()
    for d in done:
        dropped.update(d["dropped"])
        hours.update(d["hours"])
    stats = {"source_files": len(done), "rows_read": sum(d["rows_read"] for d in done),
             "rows_kept": sum(d["rows_kept"] for d in done),
             "dropped": dict(dropped.most_common()),
             "hours": {k: round(v, 2) for k, v in hours.items()},
             "filters": vars(cfg), "val_fraction": val_fraction}
    t = _read_kept(lang_dir, ["text", "duration", "n_tokens"])
    if t is not None:
        dur = t["duration"].to_pylist()
        stats["duration_s"] = _pct(dur)
        stats["chars_per_sec"] = _pct([chars_per_sec(x, d) for x, d in zip(t["text"].to_pylist(), dur)])
        tok = t["n_tokens"].to_pylist()
        stats["label_tokens"] = {**_pct(tok), "max": max(tok)}
    (lang_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    return stats


def run(lang: Lang, files: list[str], out: Path, fetch: Callable[[str], Path],
        count_tokens: Callable[[str], int], cfg: FilterConfig = FilterConfig(),
        val_fraction: float = 0.007, max_hours: float | None = None,
        keep_raw: bool = False, prefetch: int = 3) -> dict:
    lang_dir = out / lang.name
    done_dir = lang_dir / "_done"
    done_dir.mkdir(parents=True, exist_ok=True)

    def stem(f: str) -> str:
        return Path(f).stem

    # Resume: rebuild the dedup set and hour count from what is already written.
    seen: set = set()
    t = _read_kept(lang_dir, ["text", "duration"])
    if t is not None:
        seen = {(x, round(d, 1)) for x, d in zip(t["text"].to_pylist(), t["duration"].to_pylist())}
    hours_so_far = sum(sum(json.loads(p.read_text())["hours"].values()) for p in done_dir.glob("*.json"))
    todo = [f for f in files if not (done_dir / f"{stem(f)}.json").exists()]
    print(f"[{lang.name}] {len(files) - len(todo)} files done ({hours_so_far:.1f} h), "
          f"{len(todo)} to go", file=sys.stderr, flush=True)

    with ThreadPoolExecutor(max_workers=prefetch) as pool:
        futures = [pool.submit(fetch, f) for f in todo[:prefetch]]
        for i, f in enumerate(todo):
            if max_hours and hours_so_far >= max_hours:
                break
            t0 = time.time()
            path = futures[i].result()
            if i + prefetch < len(todo):
                futures.append(pool.submit(fetch, todo[i + prefetch]))

            rows = pq.read_table(path).to_pylist()
            kept, reasons = filter_rows(rows, lang, cfg, count_tokens, val_fraction, seen)
            hours = {}
            for split, split_rows in kept.items():
                if split_rows:
                    (lang_dir / split).mkdir(parents=True, exist_ok=True)
                    pq.write_table(pa.Table.from_pylist(split_rows, schema=SCHEMA),
                                   lang_dir / split / f"{stem(f)}.parquet")
                hours[split] = sum(r["duration"] for r in split_rows) / 3600
            # Marker last: a crash before this line just redoes the file.
            (done_dir / f"{stem(f)}.json").write_text(json.dumps({
                "file": f, "rows_read": len(rows), "rows_kept": sum(map(len, kept.values())),
                "dropped": dict(reasons), "hours": hours}))
            if not keep_raw:
                path.unlink(missing_ok=True)
            hours_so_far += sum(hours.values())
            print(f"[{lang.name}] {i + 1}/{len(todo)} {stem(f)}: kept "
                  f"{sum(map(len, kept.values()))}/{len(rows)}, total {hours_so_far:.1f} h, "
                  f"{time.time() - t0:.0f}s, dropped {dict(reasons)}", file=sys.stderr, flush=True)
        for fut in futures[i:] if todo else []:  # unstarted prefetches after a --max-hours stop
            fut.cancel()

    return summarize(lang_dir, cfg, val_fraction)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lang", required=True, choices=sorted(LANGS))
    p.add_argument("--out", type=Path, default=Path("data/shrutilipi"))
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--max-hours", type=float, help="stop after keeping this many hours")
    p.add_argument("--max-files", type=int, help="process at most this many source files")
    p.add_argument("--val-fraction", type=float, default=0.007)
    p.add_argument("--seed", type=int, default=42, help="file order for --max-hours sampling")
    p.add_argument("--keep-raw", action="store_true")
    p.add_argument("--min-cps", type=float, default=FilterConfig.min_cps)
    p.add_argument("--max-cps", type=float, default=FilterConfig.max_cps)
    p.add_argument("--min-script-ratio", type=float, default=FilterConfig.min_script_ratio)
    a = p.parse_args()

    lang = LANGS[a.lang]
    files = list_source_files(lang)
    random.Random(a.seed).shuffle(files)  # a --max-hours cap then samples the whole corpus
    files = files[: a.max_files] if a.max_files else files
    cfg = FilterConfig(min_cps=a.min_cps, max_cps=a.max_cps, min_script_ratio=a.min_script_ratio)
    stats = run(lang, files, a.out, hf_fetch(a.raw_dir), whisper_token_counter(), cfg,
                a.val_fraction, a.max_hours, a.keep_raw)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

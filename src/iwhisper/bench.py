"""Phase 0: Vistaar benchmarks for Gujarati and Hindi.

    iw-bench-build --out data/bench            # extract gu/hi test sets (~3 GB) to parquet
    iw-eval --model openai/whisper-small --name whisper-small --bench data/bench --out results
    iw-eval-report --out results               # markdown table of every result so far

Each benchmark is one zip for all languages on AI4Bharat's object store; only the
gu/hi folders are read, via HTTP range requests. Eval is resumable per
(model, benchmark, language) and writes predictions plus a results row.

Decoding is greedy with the language forced. Output length is capped from the
audio duration (never the reference), so a model stuck in a repetition loop
can't stall a batch or emit hundreds of junk tokens. Every model gets the same cap.
WER/CER are reported with two normalizations: ours (`normalize`) and Vistaar's
evaluation.py recipe (strip punctuation, then indicnlp normalize), so numbers
can be compared with the published IndicWhisper table.
"""

import argparse
import io
import json
import re
import string
import time
import zipfile
from pathlib import Path

import jiwer
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

from .data import LANGS
from .normalize import normalize

URL = "https://indicwhisper.objectstore.e2enetworks.net/vistaar_benchmarks/{}.zip"
BENCHMARKS = {  # benchmark -> languages it has (of gu/hi)
    "kathbath": ("gujarati", "hindi"),
    "kathbath_noisy": ("gujarati", "hindi"),
    "fleurs": ("gujarati", "hindi"),
    "indictts": ("gujarati", "hindi"),
    "mucs": ("gujarati", "hindi"),
    "commonvoice": ("hindi",),
}
# Max output tokens per second of audio: ~18 chars/s (Shrutilipi p99 speech rate)
# x Whisper tokens/char (gu ~2.7, hi ~1.2) x 1.3 slack.
TOKENS_PER_SEC = {"gujarati": 63, "hindi": 28}
MAX_NEW_TOKENS = 440

BENCH_SCHEMA = pa.schema([
    ("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
    ("text", pa.string()), ("duration", pa.float32()), ("lang", pa.string()),
])


# ---------- build ----------

def to_16k_mono_flac(data: bytes) -> bytes:
    import soxr

    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    audio = soxr.resample(audio.mean(axis=1), sr, 16000)
    buf = io.BytesIO()
    sf.write(buf, audio, 16000, format="FLAC")
    return buf.getvalue()


def build(out: Path, benchmarks: list[str], langs: list[str]) -> None:
    from .indicwhisper import HTTPRangeFile

    for b in benchmarks:
        todo = [l for l in BENCHMARKS[b] if l in langs and not (out / b / f"{l}.parquet").exists()]
        if not todo:
            continue
        zf = zipfile.ZipFile(io.BufferedReader(HTTPRangeFile(URL.format(b)), 16 << 20))
        for lang in todo:
            t0 = time.time()
            manifest = [json.loads(x) for x in zf.read(f"{b}/{lang}/manifest.json").decode().splitlines() if x.strip()]
            # Read members in archive order so the buffered reader streams sequentially.
            want = {m["audio_filepath"]: m for m in manifest}
            infos = sorted((i for i in zf.infolist() if i.filename in want), key=lambda i: i.header_offset)
            rows, resampled = [], 0
            for info in infos:
                m, data = want[info.filename], zf.read(info)
                meta = sf.info(io.BytesIO(data))
                if (meta.samplerate, meta.channels) != (16000, 1):
                    data, resampled = to_16k_mono_flac(data), resampled + 1
                rows.append({"audio": {"bytes": data, "path": info.filename}, "text": m["text"],
                             "duration": float(m["duration"]), "lang": LANGS[lang].code})
            if resampled:
                print(f"{b}/{lang}: resampled {resampled} files to 16 kHz mono", flush=True)
            (out / b).mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(rows, schema=BENCH_SCHEMA), out / b / f"{lang}.parquet")
            missing = len(manifest) - len(rows)
            print(f"{b}/{lang}: {len(rows)} utterances, {sum(r['duration'] for r in rows) / 3600:.2f} h"
                  f"{f', {missing} missing audio' if missing else ''} ({time.time() - t0:.0f}s)", flush=True)


# ---------- normalization + metrics ----------

_VISTAAR_PUNCT = str.maketrans("", "", string.punctuation + "।۔'-॥")
_indic_normalizers: dict = {}


def vistaar_normalize(text: str, lang_code: str) -> str:
    """Mirror of AI4Bharat vistaar evaluation.py: strip punctuation, indicnlp
    normalize, collapse spaces."""
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

    if lang_code not in _indic_normalizers:
        _indic_normalizers[lang_code] = IndicNormalizerFactory().get_normalizer(lang_code)
    text = _indic_normalizers[lang_code].normalize(text.translate(_VISTAAR_PUNCT))
    return re.sub(" +", " ", text).strip()


def deva_to_gujarati(text: str) -> str:
    """Map Devanagari characters to Gujarati (the blocks align almost 1:1).
    Vanilla Whisper writes Gujarati speech in Devanagari; this scores what it
    heard separately from the script it chose."""
    from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator

    return UnicodeIndicTransliterator.transliterate(text, "hi", "gu")


def metrics(refs: list[str], hyps: list[str]) -> dict:
    """Corpus-level WER/CER. Empty refs are kept as '<empty>' like Vistaar does."""
    refs = [r or "<empty>" for r in refs]
    return {"wer": round(100 * jiwer.wer(refs, hyps), 2), "cer": round(100 * jiwer.cer(refs, hyps), 2)}


def max_new_tokens(lang: str, durations: list[float]) -> int:
    return min(MAX_NEW_TOKENS, int(TOKENS_PER_SEC[lang] * max(durations)) + 10)


# ---------- eval ----------

def evaluate(transcribe, name: str, bench: Path, out: Path, benchmarks: list[str],
             langs: list[str], batch_size: int, on_result=None) -> None:
    for b in benchmarks:
        for lang in (l for l in BENCHMARKS[b] if l in langs):
            dst = out / name / b / f"{lang}.json"
            src = bench / b / f"{lang}.parquet"
            if dst.exists():
                continue
            if not src.exists():
                print(f"skip {b}/{lang}: {src} missing (run iw-bench-build)", flush=True)
                continue
            rows = pq.read_table(src).to_pylist()
            order = sorted(range(len(rows)), key=lambda i: rows[i]["duration"])
            hyps = [""] * len(rows)
            t0 = time.time()
            for s in range(0, len(order), batch_size):
                idx = order[s:s + batch_size]
                audios = [sf.read(io.BytesIO(rows[i]["audio"]["bytes"]), dtype="float32")[0] for i in idx]
                cap = max_new_tokens(lang, [rows[i]["duration"] for i in idx])
                for i, h in zip(idx, transcribe(lang, audios, cap)):
                    hyps[i] = h
            secs = time.time() - t0
            refs = [r["text"] for r in rows]
            code = LANGS[lang].code
            result = {
                "model": name, "benchmark": b, "lang": lang, "n": len(rows),
                "hours": round(sum(r["duration"] for r in rows) / 3600, 2),
                "ours": metrics([normalize(r) for r in refs], [normalize(h) for h in hyps]),
                "vistaar": metrics([vistaar_normalize(r, code) for r in refs],
                                   [vistaar_normalize(h, code) for h in hyps]),
                "empty_hyps": sum(1 for h in hyps if not h), "seconds": round(secs),
            }
            if lang == "gujarati":
                result["ours_translit"] = metrics([normalize(r) for r in refs],
                                                  [normalize(deva_to_gujarati(h)) for h in hyps])
            dst.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.table({"path": [r["audio"]["path"] for r in rows], "ref": refs, "hyp": hyps}),
                           dst.with_suffix(".parquet"))
            dst.write_text(json.dumps(result, ensure_ascii=False, indent=1))  # marker last
            if on_result:
                on_result(result)
            print(f"{name} {b}/{lang}: WER {result['ours']['wer']} (vistaar-norm "
                  f"{result['vistaar']['wer']}) CER {result['ours']['cer']} | {len(rows)} utt in {secs:.0f}s",
                  flush=True)


def report(out: Path) -> str:
    rows = [json.loads(p.read_text()) for p in sorted(out.glob("*/*/*.json"))]
    if not rows:
        return "no results yet"
    models = sorted({r["model"] for r in rows})
    cols = [(b, l) for b in BENCHMARKS for l in BENCHMARKS[b] if any(r["benchmark"] == b and r["lang"] == l for r in rows)]
    by = {(r["model"], r["benchmark"], r["lang"]): r for r in rows}
    lines = []
    titles = {"ours": "ours normalization", "vistaar": "Vistaar normalization",
              "ours_translit": "Gujarati only: Devanagari output transliterated to Gujarati first"}
    for norm in ("ours", "vistaar", "ours_translit"):
        if norm == "ours_translit":
            cols = [(b, l) for b, l in cols if l == "gujarati"]
        lines.append(f"\n### WER % ({titles[norm]})\n")
        lines.append("| model | " + " | ".join(f"{b} {LANGS[l].code}" for b, l in cols) + " |")
        lines.append("|---" * (len(cols) + 1) + "|")
        for m in models:
            cells = [str(by[(m, b, l)][norm]["wer"]) if norm in by.get((m, b, l), {}) else "" for b, l in cols]
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------- CLIs ----------

def _common(p):
    p.add_argument("--langs", nargs="+", default=["gujarati", "hindi"], choices=sorted(LANGS))
    p.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS), choices=list(BENCHMARKS))


def main_build() -> None:
    p = argparse.ArgumentParser(description="Extract Vistaar gu/hi benchmarks to parquet")
    p.add_argument("--out", type=Path, default=Path("data/bench"))
    _common(p)
    a = p.parse_args()
    build(a.out, a.benchmarks, a.langs)


# Phase 0 baselines: (label, model, languages). "indicwhisper:<lang>" is fetched from AI4Bharat.
BASELINES = [
    ("whisper-small", "openai/whisper-small", ["gujarati", "hindi"]),
    ("indicwhisper-medium", "indicwhisper:gujarati", ["gujarati"]),
    ("indicwhisper-medium", "indicwhisper:hindi", ["hindi"]),
    ("whisper-large-v3-turbo", "openai/whisper-large-v3-turbo", ["gujarati", "hindi"]),
]


def main_eval_job() -> None:
    """Cloud GPU job: build benchmarks, run every baseline, push each result to
    the Hub as it lands. Rerun after a disconnect to resume."""
    p = argparse.ArgumentParser(description=main_eval_job.__doc__)
    p.add_argument("--results-repo", required=True, help="HF dataset repo id for results")
    p.add_argument("--work", type=Path, default=Path("/content/iw"))
    p.add_argument("--only", nargs="*", help="run only these labels")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    from huggingface_hub import HfApi, snapshot_download

    from .indicwhisper import fetch
    from .score import Scorer

    api, out, bench = HfApi(), a.work / "results", a.work / "bench"
    api.create_repo(a.results_repo, repo_type="dataset", private=True, exist_ok=True)
    snapshot_download(a.results_repo, repo_type="dataset", local_dir=out, allow_patterns="*/*/*")
    build(bench, list(BENCHMARKS), ["gujarati", "hindi"])

    def upload(r):
        api.upload_folder(repo_id=a.results_repo, repo_type="dataset", folder_path=out,
                          allow_patterns=f"{r['model']}/{r['benchmark']}/{r['lang']}.*",
                          commit_message=f"{r['model']} {r['benchmark']}/{r['lang']}: WER {r['ours']['wer']}")

    for label, model, langs in BASELINES:
        if a.only and label not in a.only:
            continue
        pending = [l for l in langs for b in BENCHMARKS if l in BENCHMARKS[b]
                   and not (out / label / b / f"{l}.json").exists()]
        if not pending:
            continue
        if model.startswith("indicwhisper:"):
            model = str(fetch(model.split(":")[1], a.work / "models"))
        scorer = Scorer(model, LANGS[langs[0]].code, a.device)

        def transcribe(lang, audios, cap, scorer=scorer):
            scorer.lang_code = LANGS[lang].code
            return scorer.transcribe(audios, cap)

        evaluate(transcribe, label, bench, out, list(BENCHMARKS), langs, a.batch_size, upload)
        del scorer
        __import__("torch").cuda.empty_cache()
    report_md = report(out)
    (out / "README.md").write_text("# Phase 0 baselines\n" + report_md + "\n")
    api.upload_file(path_or_fileobj=str(out / "README.md"), path_in_repo="README.md",
                    repo_id=a.results_repo, repo_type="dataset")
    print(report_md)


def main_eval() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF id or local dir")
    p.add_argument("--name", required=True, help="short label for results")
    p.add_argument("--bench", type=Path, default=Path("data/bench"))
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    _common(p)
    a = p.parse_args()
    from .score import Scorer

    scorer = Scorer(a.model, LANGS[a.langs[0]].code, a.device)

    def transcribe(lang, audios, cap):
        scorer.lang_code = LANGS[lang].code  # same weights, different forced language
        return scorer.transcribe(audios, cap)

    evaluate(transcribe, a.name, a.bench, a.out, a.benchmarks, a.langs, a.batch_size)
    print(report(a.out))


def main_report() -> None:
    p = argparse.ArgumentParser(description="Print a WER table of all results")
    p.add_argument("--out", type=Path, default=Path("results"))
    print(report(p.parse_args().out))

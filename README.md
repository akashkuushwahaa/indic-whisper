# indic-whisper

Fine-tune Whisper for **Gujarati and Hindi** on [Shrutilipi](https://huggingface.co/datasets/ai4bharat/Shrutilipi)
(All India Radio news, mined audio/text pairs, CC-BY-4.0), evaluated on the
[Vistaar](https://github.com/AI4Bharat/vistaar) benchmarks against vanilla Whisper and IndicWhisper.

## Phase 1: data prep

Shrutilipi is gated. Accept the terms on its dataset page, then log in once:

```bash
uv sync
uv run hf auth login        # paste a read token from huggingface.co/settings/tokens
```

Look at real rows before a full run (checks paths, audio format, filter thresholds):

```bash
uv run iw-inspect --lang gujarati -n 20
uv run iw-inspect --lang hindi -n 20
```

Smoke run, then full runs:

```bash
uv run iw-prepare --lang gujarati --max-rows 2000 --out data/smoke
uv run iw-prepare --lang gujarati                    # all of it, ~17.7 GB streamed
uv run iw-prepare --lang hindi --max-hours 400       # random sample of a 95 GB corpus
```

Output: `data/shrutilipi/<lang>/{train,val}/part-*.parquet` plus `stats.json`
(rows dropped by reason, hours kept, duration / chars-per-second / token percentiles).
Shards load with `load_dataset("parquet", ...)` and `audio` is a 16 kHz `Audio` feature.

Columns: `audio`, `text` (normalized: training/eval target), `text_raw` (keeps punctuation),
`duration`, `lang` (Whisper code), `n_tokens`.

### What gets dropped

| reason | rule |
|---|---|
| `too_short` / `too_long` | outside 1–30 s (30 s is Whisper's window) |
| `wrong_script` | < 85% of letters in the language's native script |
| `cps_low` / `cps_high` | outside 3–25 normalized chars/sec: transcript likely misaligned |
| `too_many_tokens` | label > 440 Whisper tokens (decoder max is 448) |
| `duplicate` | same normalized text and duration |

Train/val are split by bulletin (`group_key` strips the trailing segment index from the
path) so clips from one bulletin never appear on both sides. Confirm the real path
format with `iw-inspect` and adjust `group_key` if it doesn't match.

### Known issues

- **Gujarati token cost.** Whisper's BPE spends ~2.7 tokens per Gujarati character (Hindi
  ~1.2), so Gujarati speech hits the 448-token decoder limit after roughly 11 s. Check
  `dropped.too_many_tokens` in `stats.json`.
- **Normalization.** Never use Whisper's `BasicTextNormalizer` on Indic text: it strips
  vowel signs. Use `iwhisper.normalize.normalize` for labels and for WER.

## Phase 1b: noise filter with IndicWhisper (GPU) — measured, then skipped

**Result (2026-11-25, Gujarati, 30,400 clips scored by CER):** after the cheap filters the data is
already clean. Median CER vs IndicWhisper is 3.5%, p99 0.25, and only 0.2% of clips exceed
0.4. Inspecting those showed mostly *model* failures, not bad labels: greedy decoding loops
("2012222222…", "સુધી સુધી સુધી…") on clips whose labels are correct. Teacher-forced NLL agrees:
the label fits the audio (NLL 0.03–0.1) for most high-CER clips. A CER filter would drop ~0.2%
of the data, much of it good, for ~50 free-GPU hours of scoring both languages, so training
uses the Phase 1 filters only. The scoring tools below stay for future corpora.

Caveat: IndicWhisper was probably trained on Shrutilipi, so both scores can be optimistic. A
human spot-check (listen to ~30 random clips per language) is the cheap independent check.

`iw-score --method nll` (default) is a single forward pass per batch; `--method cer` generates.

Transcribe every kept clip with AI4Bharat's IndicWhisper (whisper-medium) and store the
CER against its label. Training drops clips above a threshold (start ~0.4), which catches
misaligned pairs the cheap filters miss.

```bash
uv run iw-fetch-indicwhisper --lang gujarati --out models   # 1.5 GB, extracted from the zip
uv run iw-fetch-indicwhisper --lang hindi --out models      # 1.5 GB of a 4.3 GB zip
uv run iw-score --lang gujarati --model models/whisper-medium-gu_alldata_multigpu --device cuda
```

**On free Colab (T4):** open
[notebooks/score_colab.ipynb](https://colab.research.google.com/github/akashkuushwahaa/indic-whisper/blob/main/notebooks/score_colab.ipynb),
follow the four setup steps in its first cell and Run all. It runs `iw-score-job`, which
rebuilds the shards inside Colab (no local upload), scores them and pushes the CER
sidecars to a private HF dataset every few shards. After a disconnect, Run all again.

Two GPUs on one box: run `--device cuda:0 --part 0 --num-parts 2` and
`--device cuda:1 --part 1 --num-parts 2` side by side. Sidecars land in
`<lang>/scores/<split>/<shard>.parquet` (`path`, `cer`, `hyp`); reruns skip finished shards.

Caveat: Vistaar's training data likely includes Shrutilipi, so IndicWhisper may have seen
these clips. CER is then optimistic and the filter lenient. It still flags gross misalignment.

## Phase 0: baselines on Vistaar benchmarks

```bash
uv run iw-bench-build --out data/bench        # gu/hi from 6 Vistaar zips: 21k utterances, ~36 h
uv run iw-eval --model openai/whisper-small --name whisper-small --device cuda
uv run iw-eval-report
```

On Kaggle (2× T4, both used): import `notebooks/eval_kaggle.ipynb` from GitHub. On free Colab: [notebooks/eval_colab.ipynb](https://colab.research.google.com/github/akashkuushwahaa/indic-whisper/blob/main/notebooks/eval_colab.ipynb)
runs `iw-eval-job` over whisper-small, IndicWhisper-medium and whisper-large-v3-turbo, pushing each
result to a private HF dataset as it lands (resumable).

Decoding: greedy, language forced, output capped from audio duration (never the reference).
Metrics: corpus WER/CER with our normalizer, with Vistaar's evaluation.py recipe, and for
Gujarati also after transliterating Devanagari output to Gujarati script.

**Why the transliterated column exists:** forced to `gu`, vanilla whisper-small writes Gujarati
speech in Devanagari ("पानी मा पनी हारियों" for "પાણીમાં પનિહારીઓ"). On IndicTTS-gu (100 utt,
local CPU check): raw WER 148 / CER 116; transliterated WER 124 / CER 61; excluding 8 clips
that loop, WER 80 / CER 34. Script, word boundaries and loops are separate problems.

## Phase 2: fine-tune whisper-small (Gujarati + Hindi)

`iw-train --preset <name>`: `rtx4050-6gb` (6 GB laptop GPU: bf16, 8-bit AdamW, gradient
checkpointing, batch 8 × 4), `t4` (fp16, batch 16 × 2), `smoke` (10 steps of whisper-tiny on CPU).
Defaults: 5,000 steps, lr 1e-5, 500 warmup, 50/50 language mix, raw (punctuated) targets, val
WER/CER per language every 500 steps on 200 clips each, best checkpoint kept. Resume with `--resume`.

### Windows + RTX 4050 laptop setup

Install [git](https://git-scm.com/download/win) and uv (PowerShell:
`powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`), then in a new terminal:

```powershell
git clone https://github.com/akashkuushwahaa/indic-whisper; cd indic-whisper
uv venv; .venv\Scripts\activate
# CUDA build of PyTorch: copy the pip command for Windows + CUDA from https://pytorch.org/get-started/locally/
# and run it with `uv pip` instead of `pip`, e.g.  uv pip install torch --index-url https://download.pytorch.org/whl/cuXXX
uv pip install -e ".[train]"
python -c "import torch, bitsandbytes; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Use this activated venv (not `uv sync` / `uv run`, which would swap the CUDA build of torch for the default one).
The last line must print `True NVIDIA GeForce RTX 4050 Laptop GPU`.

Data (~45 GB download, ~40 GB on disk; same seed -> same shards as any other machine):

```powershell
hf auth login
iw-prepare --lang gujarati
iw-prepare --lang hindi --max-hours 400
```

Measure speed first (~15 min), then the real run:

```powershell
iw-train --preset rtx4050-6gb --out runs/speedtest --max-steps 60 --eval-steps 50 --save-steps 1000 --val-per-lang 50
iw-train --preset rtx4050-6gb --out runs/small-gu-hi            # add --resume after any interruption
tensorboard --logdir runs                                       # loss + val WER curves
```

Plugged in, Windows power mode "Best performance", sleep off. Out of memory: `--batch-size 4 --grad-accum 8`.

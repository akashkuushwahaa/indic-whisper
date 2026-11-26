# indic-whisper: context for Claude Code

Fine-tune `openai/whisper-small` for Gujarati + Hindi on Shrutilipi (AI4Bharat, CC-BY-4.0) and
compare against vanilla Whisper and IndicWhisper on the Vistaar benchmarks. Portfolio project:
fine-tuning, evals, model serving. README.md has every command; this file is status + rules.

## Rules
- Never add `Co-Authored-By: Claude` or "Generated with Claude Code" to commits or PRs.
- Use `iwhisper.normalize.normalize` for labels and WER. Never Whisper's `BasicTextNormalizer`
  (it deletes Indic vowel signs).
- Never train on or tune against benchmark test sets (`data/bench`).
- Long jobs (prep, training, eval) go in the background with logs under `data/logs/`.

## Status (2026-11-26)
- Phase 1 data prep: done on the Mac. Gujarati 176 h train / 0.9 h val (all 33 files), Hindi
  398 h / 2.6 h (58 of 196 files, seed 42, `--max-hours 400`). Same command + seed reproduces
  the same shards on any machine.
- Phase 1b noise filter: measured on 30k Gujarati clips and skipped. Median CER 3.5%, only 0.2%
  above 0.4, and most of those were IndicWhisper decoding loops on correct labels.
- Phase 0 baselines: tooling done (`iw-bench-build`, `iw-eval`). Local check, whisper-small on
  IndicTTS: Hindi WER 48; Gujarati WER 148 because it writes Gujarati speech in Devanagari
  (124 after transliteration). Full baseline table not run yet.
- Phase 2 training: `iw-train` written and smoke-tested on Mac CPU (whisper-tiny). Not yet run
  on a CUDA GPU.

## Next on the Windows RTX 4050 laptop (6 GB)
Follow README "Windows + RTX 4050 laptop setup":
1. venv + CUDA build of PyTorch from the pytorch.org index, then `uv pip install -e ".[train]"`.
   Use the activated venv, not `uv sync`/`uv run` (they replace CUDA torch with the CPU build).
   Check: `python -c "import torch, bitsandbytes; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`.
2. `hf auth login` (the account must have accepted the Shrutilipi terms), then
   `iw-prepare --lang gujarati` and `iw-prepare --lang hindi --max-hours 400` (~45 GB download).
3. Speed test: `iw-train --preset rtx4050-6gb --out runs/speedtest --max-steps 60 --eval-steps 50 --save-steps 1000 --val-per-lang 50`.
   Report seconds/step and peak VRAM (`nvidia-smi`). Out of memory -> `--batch-size 4 --grad-accum 8`.
4. Real run: `iw-train --preset rtx4050-6gb --out runs/small-gu-hi` (5,000 steps; `--resume` after any stop).
5. Baselines and fine-tuned eval on the GPU: `iw-bench-build`, then `iw-eval --device cuda` for
   `openai/whisper-small`, IndicWhisper (`iw-fetch-indicwhisper --lang gujarati|hindi`) and
   `runs/small-gu-hi/best`. `iw-eval-report` prints the table.

## Windows gotchas
- DataLoader workers use spawn: the dataset class must stay module-level (a test enforces it).
- If bitsandbytes fails to load on Windows, use `--optim adamw_torch_fused` (more VRAM, so also
  `--batch-size 4 --grad-accum 8`).
- Keep the laptop plugged in, power mode "Best performance", sleep off during training.

## Open questions
- Gujarati: Whisper's BPE spends ~2.7 tokens per character, so clips over ~11 s were dropped by
  the 448-token limit (about 10% of Gujarati). Check long-audio WER after training; vocabulary
  extension is the fix if it matters.

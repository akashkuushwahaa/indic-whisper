"""Phase 1b end to end on a cloud GPU box (Colab, Kaggle, Lightning...):
rebuild the filtered shards for one language from the Hub, fetch IndicWhisper,
score every clip, and push CER sidecars to a private HF dataset in small chunks.

Survives disconnects: scores already on the Hub are pulled first and skipped,
and each chunk is uploaded as soon as it finishes, so a dropped session loses
at most one chunk of work. Rerun the same command to resume.

    HF_TOKEN=... iw-score-job --lang gujarati --scores-repo <user>/shrutilipi-gu-hi-scores
"""

import argparse
import subprocess
import sys
from pathlib import Path

from .data import LANGS
from .indicwhisper import fetch
from .score import SCORE_DIRS, shards_for

# Must match the local prep so both sides select the same source files.
DEFAULT_MAX_HOURS = {"hindi": 400}


def sh(args: list[str]) -> int:
    print("$", " ".join(args), flush=True)
    return subprocess.call(args)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lang", required=True, choices=sorted(LANGS))
    p.add_argument("--scores-repo", required=True, help="HF dataset repo id for the sidecars")
    p.add_argument("--work", type=Path, default=Path("/content/iw"))
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--method", choices=sorted(SCORE_DIRS), default="nll")
    p.add_argument("--chunk", type=int, default=3, help="shards per GPU between uploads")
    p.add_argument("--batch-size", type=int, default=16)
    a = p.parse_args()

    from huggingface_hub import HfApi, snapshot_download

    lang, data = LANGS[a.lang], a.work / "data"
    scores = f"{lang.name}/{SCORE_DIRS[a.method]}/**"
    api = HfApi()
    api.create_repo(a.scores_repo, repo_type="dataset", private=True, exist_ok=True)
    snapshot_download(a.scores_repo, repo_type="dataset", local_dir=data,
                      allow_patterns=scores)

    prep = [sys.executable, "-m", "iwhisper.prepare", "--lang", lang.name,
            "--out", str(data), "--raw-dir", str(a.work / "raw")]
    if lang.name in DEFAULT_MAX_HOURS:
        prep += ["--max-hours", str(DEFAULT_MAX_HOURS[lang.name])]
    if sh(prep):
        sys.exit("prepare failed")
    model = fetch(lang.name, a.work / "models")

    def remaining() -> int:
        return sum(not dst.exists() for _, dst in shards_for(data / lang.name, 0, 1, a.method))

    total = len(shards_for(data / lang.name, 0, 1, a.method))
    while (left := remaining()):
        print(f"[{lang.name}] {total - left}/{total} shards scored", flush=True)
        procs = [subprocess.Popen([
            sys.executable, "-m", "iwhisper.score", "--lang", lang.name, "--model", str(model),
            "--method", a.method,
            "--data", str(data), "--device", f"cuda:{i}", "--part", str(i),
            "--num-parts", str(a.gpus), "--max-shards", str(a.chunk),
            "--batch-size", str(a.batch_size)]) for i in range(a.gpus)]
        if any(proc.wait() for proc in procs):
            sys.exit("a scoring process failed; see the log above")
        api.upload_folder(repo_id=a.scores_repo, repo_type="dataset", folder_path=data,
                          allow_patterns=[scores, f"{lang.name}/stats.json"],
                          commit_message=f"{lang.name}: {total - remaining()}/{total} shards scored ({a.method})")
        if remaining() == left:
            sys.exit("no progress in a full round; stopping")
    print(f"[{lang.name}] all {total} shards scored", flush=True)


if __name__ == "__main__":
    main()

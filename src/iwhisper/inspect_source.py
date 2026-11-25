"""Print the first N rows of one Shrutilipi file for a language: path, bulletin
group, audio format, duration, chars/sec, script ratio, raw and normalized text.

    iw-inspect --lang gujarati -n 20
"""

import argparse
import io
import itertools
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf

from .data import LANGS, chars_per_sec, group_key, script_ratio
from .normalize import normalize
from .prepare import hf_fetch, list_source_files


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lang", required=True, choices=sorted(LANGS))
    p.add_argument("-n", type=int, default=20)
    p.add_argument("--file-index", type=int, default=0)
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    a = p.parse_args()
    lang = LANGS[a.lang]

    path = hf_fetch(a.raw_dir)(list_source_files(lang)[a.file_index])
    batch = next(pq.ParquetFile(path).iter_batches(batch_size=a.n))
    for row in itertools.islice(batch.to_pylist(), a.n):
        audio, dur = row["audio_filepath"], float(row["duration"])
        info = sf.info(io.BytesIO(audio["bytes"]))
        norm = normalize(row["text"])
        print(f"path   {audio['path']}  ->  group {group_key(audio['path'])}")
        print(f"audio  {info.format}/{info.subtype} {info.samplerate} Hz {info.channels}ch  "
              f"dur {dur:.2f}s  lang {row['lang']}")
        print(f"cps {chars_per_sec(norm, dur):.1f}  script {script_ratio(norm, lang):.2f}")
        print(f"raw    {row['text']}")
        print(f"norm   {norm}\n")


if __name__ == "__main__":
    main()

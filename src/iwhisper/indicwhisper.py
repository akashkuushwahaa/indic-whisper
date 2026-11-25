"""Fetch one IndicWhisper checkpoint without downloading the whole zip.

AI4Bharat ships each language as one zip (Hindi is 4.3 GB because it also holds
a whisper-large model). HTTP range requests let us read the zip's index and
extract only the files of the model we want.

    iw-fetch-indicwhisper --lang gujarati --out models
    -> models/whisper-medium-gu_alldata_multigpu/
"""

import argparse
import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

URL = "https://indicwhisper.objectstore.e2enetworks.net/{lang}_models.zip"
MODELS = {  # whisper-medium for both languages, so the two filters are comparable
    "gujarati": "gujarati_models/whisper-medium-gu_alldata_multigpu/",
    "hindi": "hindi_models/whisper-medium-hi_alldata_multigpu/",
}
SKIP = ("rng_state", "training_args.bin", "trainer_state.json", "latest", "zero_to_fp32.py")


class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests."""

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        head = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"))
        self.size = int(head.headers["Content-Length"])

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size or not len(b):
            return 0
        end = min(self.pos + len(b), self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        data = urllib.request.urlopen(req).read()
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


def fetch(lang: str, out: Path) -> Path:
    prefix = MODELS[lang]
    dest = out / Path(prefix).name
    zf = zipfile.ZipFile(io.BufferedReader(HTTPRangeFile(URL.format(lang=lang)), 8 << 20))
    for info in zf.infolist():
        name = info.filename
        if info.is_dir() or not name.startswith(prefix) or Path(name).name.startswith(SKIP):
            continue
        target = dest / Path(name).name
        if target.exists() and target.stat().st_size == info.file_size:
            continue
        print(f"  {info.file_size / 1e6:8.1f} MB  {target}", flush=True)
        dest.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as src, open(target.with_suffix(".part"), "wb") as dst:
            shutil.copyfileobj(src, dst, 8 << 20)
        target.with_suffix(".part").rename(target)
    return dest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lang", required=True, choices=sorted(MODELS))
    p.add_argument("--out", type=Path, default=Path("models"))
    print(fetch(**vars(p.parse_args())))


if __name__ == "__main__":
    main()

import io
import json
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from datasets import Audio, load_dataset

from iwhisper.data import LANGS, FilterConfig, assign_split, group_key, reject_reason, script_ratio
from iwhisper.prepare import run

HI, GU = LANGS["hindi"], LANGS["gujarati"]
CFG = FilterConfig()
GOOD = "आज मौसम अच्छा है और बारिश होगी।"  # 25 normalized chars -> 8.3 cps at 3 s


def flac(seconds: float, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(seconds * sr), dtype=np.float32), sr, format="FLAC")
    return buf.getvalue()


def row(text, dur, path):
    return {"audio_filepath": {"bytes": flac(dur), "path": path}, "text": text,
            "duration": dur, "lang": "hi"}


def write_source(dir_, name, rows):
    """A parquet file shaped like a Shrutilipi source file."""
    p = dir_ / "src" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), p)
    return name


def fetcher(dir_, log=None):
    """Copy, because run() deletes the fetched file after processing."""
    def fetch(name):
        if log is not None:
            log.append(name)
        dst = dir_ / "raw" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(dir_ / "src" / name, dst)
        return dst
    return fetch


def test_script_ratio():
    assert script_ratio("नमस्ते दुनिया", HI) == 1.0
    assert script_ratio("hello world", HI) == 0.0
    assert script_ratio("નમસ્તે", HI) == 0.0
    assert script_ratio("નમસ્તે", GU) == 1.0


def test_reject_reasons():
    ok = "आज मौसम अच्छा है और बारिश होगी"  # 25 chars
    assert reject_reason(ok, 2.0, 20, HI, CFG) is None
    assert reject_reason("", 2.0, 0, HI, CFG) == "empty_text"
    assert reject_reason(ok, 0.5, 20, HI, CFG) == "too_short"
    assert reject_reason(ok, 31.0, 20, HI, CFG) == "too_long"
    assert reject_reason("this is english text", 2.0, 20, HI, CFG) == "wrong_script"
    assert reject_reason("आज", 10.0, 5, HI, CFG) == "cps_low"
    assert reject_reason(ok * 3, 2.5, 20, HI, CFG) == "cps_high"
    assert reject_reason(ok, 2.0, 500, HI, CFG) == "too_many_tokens"


def test_group_key_matches_real_shrutilipi_paths():
    path = "NSD-Gujarati-Gujarati-1950-2000-2019112621258_chunk_9.flac"
    assert group_key(path) == "NSD-Gujarati-Gujarati-1950-2000-2019112621258_chunk"
    first = assign_split("Regional-Shimla-Hindi-0915-202083110323_chunk_1.flac", 0.5)
    assert all(assign_split(f"Regional-Shimla-Hindi-0915-202083110323_chunk_{i}.flac", 0.5) == first
               for i in range(100))


def test_run_end_to_end(tmp_path):
    f1 = write_source(tmp_path, "hindi/train-00000.parquet", [
        row(GOOD, 3.0, "b1_chunk_1.flac"),
        row(GOOD, 3.0, "b1_chunk_2.flac"),  # duplicate
        row("दूसरी खबर दिल्ली से है", 2.0, "b2_chunk_1.flac"),
        row("english only text here", 2.0, "b3_chunk_1.flac"),
        row("", 2.0, "b4_chunk_1.flac"),
    ])
    f2 = write_source(tmp_path, "hindi/train-00001.parquet", [
        row(GOOD, 3.0, "b5_chunk_1.flac"),  # duplicate across files
        row("तीसरी खबर मुंबई से आई है", 2.0, "b6_chunk_1.flac"),
    ])
    out = tmp_path / "out"
    stats = run(HI, [f1, f2], out, fetcher(tmp_path), lambda t: len(t) // 2, val_fraction=0.0)

    assert stats["source_files"] == 2 and stats["rows_read"] == 7 and stats["rows_kept"] == 3
    assert stats["dropped"] == {"duplicate": 2, "wrong_script": 1, "empty_text": 1}
    assert json.loads((out / "hindi/stats.json").read_text())["rows_kept"] == 3
    assert not list((tmp_path / "raw").rglob("*.parquet")), "raw downloads should be deleted"

    ds = load_dataset("parquet", data_files=str(out / "hindi/train/*.parquet"), split="train")
    assert len(ds) == 3 and isinstance(ds.features["audio"], Audio)
    # Decoding via datasets needs torchcodec (present in the training env); check bytes directly.
    ds = ds.cast_column("audio", Audio(decode=False))
    assert ds[0]["text"] == "आज मौसम अच्छा है और बारिश होगी"
    assert ds[0]["text_raw"].endswith("।")  # punctuation kept in the raw label
    audio, sr = sf.read(io.BytesIO(ds[0]["audio"]["bytes"]))
    assert (len(audio), sr) == (48000, 16000)


def test_resume_skips_done_files_and_keeps_dedup(tmp_path):
    f1 = write_source(tmp_path, "hindi/a.parquet", [row(GOOD, 3.0, "b1_chunk_1.flac")])
    f2 = write_source(tmp_path, "hindi/b.parquet", [row(GOOD, 3.0, "b2_chunk_1.flac")])
    out, fetched = tmp_path / "out", []
    run(HI, [f1], out, fetcher(tmp_path, fetched), lambda t: 10)
    stats = run(HI, [f1, f2], out, fetcher(tmp_path, fetched), lambda t: 10)
    assert fetched == [f1, f2]  # f1 not fetched again
    assert stats["rows_kept"] == 1 and stats["dropped"] == {"duplicate": 1}


def test_max_hours_stops_between_files(tmp_path):
    files = [write_source(tmp_path, f"hindi/{i}.parquet",
                          [row(f"खबर नंबर {i} आज की ताज़ा", 2.0, f"b{i}_chunk_1.flac")])
             for i in range(10)]
    stats = run(HI, files, tmp_path / "out", fetcher(tmp_path), lambda t: 10,
                max_hours=2.5 * 2.0 / 3600, prefetch=1)  # cap falls inside file 3
    assert stats["source_files"] == 3

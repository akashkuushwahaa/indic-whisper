import collections

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from iwhisper.data import SCHEMA
from iwhisper.train import Collator, label_ids, mixed_stream
from test_prepare import flac


@pytest.fixture(scope="module")
def processor():
    from transformers import WhisperProcessor
    return WhisperProcessor.from_pretrained("openai/whisper-tiny")


def shard(path, lang, n):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"audio": {"bytes": flac(1.0), "path": f"{lang}{i}"}, "text": f"{lang} {i}", "text_raw": f"{lang} {i}।",
             "duration": 1.0, "lang": lang, "n_tokens": 5} for i in range(n)]
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path)
    return path


def test_label_ids_drop_sot_and_end_with_eot(processor):
    tok = processor.tokenizer
    ids = label_ids(tok, "gu", "નમસ્તે")
    assert tok.convert_ids_to_tokens(ids[:3]) == ["<|gu|>", "<|transcribe|>", "<|notimestamps|>"]
    assert ids[-1] == tok.convert_tokens_to_ids("<|endoftext|>")
    assert tok.decode(ids, skip_special_tokens=True) == "નમસ્તે"


def test_mixed_stream_weights_and_cycling(tmp_path):
    shards = {"gu": [shard(tmp_path / "gu.parquet", "gu", 5)],
              "hi": [shard(tmp_path / "hi1.parquet", "hi", 50), shard(tmp_path / "hi2.parquet", "hi", 50)]}
    stream = mixed_stream(shards, {"gu": 0.5, "hi": 0.5}, seed=0, buffer=20)
    langs = collections.Counter(next(stream)["lang"] for _ in range(2000))
    assert 0.45 < langs["gu"] / 2000 < 0.55  # small language is cycled, not exhausted


def test_collator_shapes_and_padding(processor):
    items = [{"audio": flac(1.0), "lang": "hi", "text": "नमस्ते"},
             {"audio": flac(2.0), "lang": "gu", "text": "નમસ્તે દુનિયા"}]
    batch = Collator(processor)(items)
    assert tuple(batch["input_features"].shape) == (2, 80, 3000)
    lab = batch["labels"]
    assert (lab[0] == -100).sum() > 0 and (lab[1] == -100).sum() == 0  # shorter label padded


def test_stream_pickles_for_windows_spawn_workers():
    import pickle

    from iwhisper.train import ShardStream
    pickle.loads(pickle.dumps(ShardStream({"gu": []}, {"gu": 1.0}, 0, "raw")))

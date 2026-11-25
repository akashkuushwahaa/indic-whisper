import pyarrow as pa
import pyarrow.parquet as pq

from iwhisper.data import SCHEMA
from iwhisper.score import cer, cer_batch, score_shard, shards_for
from test_prepare import flac


def write_kept(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path)


def kept(text, path, n_tokens):
    return {"audio": {"bytes": flac(1.5), "path": path}, "text": text, "text_raw": text,
            "duration": 1.5, "lang": "hi", "n_tokens": n_tokens}


def test_cer():
    assert cer("नमस्ते", "नमस्ते") == 0.0
    assert cer("नमस्ते", "") == 1.0
    assert 0 < cer("आज बारिश होगी", "आज बारिश") < 1


def test_score_shard_aligns_rows_despite_sorting(tmp_path):
    src = tmp_path / "hindi/train/a.parquet"
    write_kept(src, [kept("लंबा वाक्य यहाँ है", "p_long", 40), kept("छोटा", "p_short", 8),
                     kept("गलत लेबल", "p_bad", 10)])
    seen = []

    def fake(rows, audios):
        seen.append([r["n_tokens"] for r in rows])
        assert len(rows) == len(audios)
        return [{"nll": float(r["n_tokens"])} for r in rows]

    dst = tmp_path / "hindi/scores_nll/train/a.parquet"
    n, secs = score_shard(src, dst, fake, batch_size=2)
    got = {r["path"]: r["nll"] for r in pq.read_table(dst).to_pylist()}
    assert n == 3 and secs == 4.5
    assert got == {"p_long": 40.0, "p_short": 8.0, "p_bad": 10.0}  # results follow their rows
    assert seen == [[8, 10], [40]]  # batched in label-length order
    assert pq.read_schema(dst).field("nll").type == pa.float32()
    assert not list(dst.parent.glob("*.tmp"))


def test_cer_batch_normalizes_hypotheses():
    class FakeScorer:
        def transcribe(self, audios, max_new):
            assert max_new == int(10 * 1.25) + 8
            return ["छोटा।", "बिलकुल अलग बात"]

    rows = [{"text": "छोटा", "n_tokens": 8}, {"text": "गलत लेबल", "n_tokens": 10}]
    out = cer_batch(FakeScorer())(rows, [None, None])
    assert out[0] == {"cer": 0.0, "hyp": "छोटा"}
    assert out[1]["cer"] > 0.5


def test_shards_for_partitions_without_overlap(tmp_path):
    for s in ("train/a", "train/b", "train/c", "val/a"):
        write_kept(tmp_path / f"gujarati/{s}.parquet", [])
    for method, dirname in (("cer", "scores"), ("nll", "scores_nll")):
        parts = [shards_for(tmp_path / "gujarati", i, 2, method) for i in range(2)]
        flat = [src for p in parts for src, _ in p]
        assert len(flat) == 4 == len(set(flat))
        assert all(dst.parts[-3:-1] == (dirname, src.parent.name) for p in parts for src, dst in p)

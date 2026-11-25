import json

import pyarrow as pa
import pyarrow.parquet as pq

from iwhisper.bench import BENCH_SCHEMA, evaluate, max_new_tokens, metrics, report, vistaar_normalize
from test_prepare import flac


def test_vistaar_normalize_strips_punct_and_danda():
    assert vistaar_normalize("नमस्ते, दुनिया। कैसे हो?", "hi") == "नमस्ते दुनिया कैसे हो"
    assert vistaar_normalize("ગુજરાતી  ભાષા.", "gu") == "ગુજરાતી ભાષા"


def test_metrics_corpus_level():
    m = metrics(["a b c d", "e f"], ["a b x d", "e f"])
    assert m["wer"] == round(100 * 1 / 6, 2)
    assert metrics([""], ["x"])["wer"] == 100.0  # empty ref -> '<empty>'


def test_max_new_tokens_from_duration_only():
    assert max_new_tokens("hindi", [1.0, 5.0]) == 28 * 5 + 10
    assert max_new_tokens("gujarati", [30.0]) == 440


def test_evaluate_resumable_and_report(tmp_path):
    bench, out = tmp_path / "bench", tmp_path / "results"
    (bench / "fleurs").mkdir(parents=True)
    rows = [{"audio": {"bytes": flac(d), "path": f"fleurs/hindi/wavs/{i}.wav"}, "text": t,
             "duration": d, "lang": "hi"} for i, (t, d) in enumerate([("आज बारिश होगी।", 2.0), ("कल धूप", 1.0)])]
    pq.write_table(pa.Table.from_pylist(rows, schema=BENCH_SCHEMA), bench / "fleurs/hindi.parquet")
    calls = []

    def fake(lang, audios, cap):
        calls.append((lang, len(audios), cap))
        return ["कल धूप" if len(a) == 16000 else "आज बारिश" for a in audios]

    evaluate(fake, "m1", bench, out, ["fleurs", "kathbath"], ["hindi"], batch_size=8)
    r = json.loads((out / "m1/fleurs/hindi.json").read_text())
    assert r["n"] == 2 and r["ours"]["wer"] == 20.0 and r["empty_hyps"] == 0  # 1 deletion / 5 words
    assert calls == [("hindi", 2, 28 * 2 + 10)]
    preds = pq.read_table(out / "m1/fleurs/hindi.parquet").to_pylist()
    assert preds[0]["hyp"] == "आज बारिश"  # predictions stay aligned to rows after sorting
    evaluate(fake, "m1", bench, out, ["fleurs"], ["hindi"], batch_size=8)
    assert len(calls) == 1  # resumed: nothing re-run
    assert "| m1 | 20.0 |" in report(out)


def test_resample_to_16k_mono():
    import io

    import numpy as np
    import soundfile as sf

    from iwhisper.bench import to_16k_mono_flac
    buf = io.BytesIO()
    sf.write(buf, np.zeros((48000, 2), dtype=np.float32), 48000, format="WAV")
    audio, sr = sf.read(io.BytesIO(to_16k_mono_flac(buf.getvalue())))
    assert sr == 16000 and audio.ndim == 1 and abs(len(audio) - 16000) <= 1


def test_deva_to_gujarati():
    from iwhisper.bench import deva_to_gujarati
    assert deva_to_gujarati("पानी") == "પાની"
    assert deva_to_gujarati("ગુજરાતી") == "ગુજરાતી"  # already Gujarati: unchanged

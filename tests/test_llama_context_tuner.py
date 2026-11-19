'''Host-side self-tests for llama-context-tuner.py (no GPU, no real docker).

Mocks the shared launch primitive so the search logic -- exponential probe,
bisection, OOM classification, native-limit detection and non-OOM abort -- is
tested without loading a model. Run with:

    .venv/bin/python tests/test_llama_context_tuner.py
'''

import importlib.util
import io
import json
import tempfile
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("ctx_tuner",
                                              HERE / "llama-context-tuner.py")
ct = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ct)


def ok_log(tokens=4096):
    return "\n".join([
        f"slot  0: id 0 | task 1 | prompt processing, n_tokens = {tokens:6d}, "
        "progress = 1.00, t =   10.00 s / 409.60 tokens per second",
        f"slot  0: id 0 | task 1 | prompt eval time =  10000.00 ms / {tokens} tokens "
        "(    2.44 ms per token,   409.60 tokens per second)",
    ])


def ok_result(ctx):
    return {"status": "ok", "server_log": ok_log(min(ctx, 4096)),
            "response": {"prompt_tokens": 4096}}


def oom_result(ctx):
    return {"status": "error",
            "error": "model did not finish loading within load_timeout",
            "server_log": (f"llama_kv_cache_init: n_ctx = {ctx}\n"
                           "ggml_cuda_init: failed to allocate KV cache: "
                           "CUDA out of memory\n")}


def base_config(**overrides):
    cfg = {
        "model_name": "M", "model_file": "M.gguf",
        "native_ctx": 262144, "min_ctx": 4096,
        "ctx_granularity": 1024, "ctx_multiplier": 2,
        "server_parameters": {"--threads": "12"}, "ntg": 1,
    }
    cfg.update(overrides)
    return cfg


def fake_launcher(limit):
    '''Return a callable that is ok for ctx <= limit and OOM above it.'''
    def launch(config, cell, spec_flags, prompt_file, launch_id, model_file,
               ntg=None):
        assert "--ctx-size" in cell
        ctx = int(cell["--ctx-size"])
        return ok_result(ctx) if ctx <= limit else oom_result(ctx)
    return launch


def test_classify():
    assert ct.classify(ok_result(4096))[0] == "ok"
    kind, reason = ct.classify(oom_result(131072))
    assert kind == "oom" and "OOM" in reason, (kind, reason)
    kind, _ = ct.classify({"status": "error", "error": "bad flag --zzz",
                           "server_log": "usage: llama-server ..."})
    assert kind == "error"
    print("  test_classify  ok")


def test_align_and_ladder():
    assert ct.align(5000, 1024) == 4096
    assert ct.align(1023, 1024) == 1024
    assert ct.probe_sequence(4096, 20000, 2, 1024) == \
        [4096, 8192, 16384, 20000]
    print("  test_align_and_ladder  ok")


def test_oom_boundary_search():
    out = io.StringIO()
    summary = {}
    with mock.patch.object(ct.LLM, "run_bench_launch",
                           side_effect=fake_launcher(65536)):
        n, max_safe, first_bad, native, last_good = ct.find_max_ctx(
            base_config(), Path("/tmp/probe.txt"), out, summary)
    assert max_safe == 65536, max_safe
    assert first_bad is not None and 0 < first_bad - max_safe <= 1024, \
        (first_bad, max_safe)
    assert native is False and last_good["status"] == "ok"
    rows = [json.loads(x) for x in out.getvalue().splitlines()]
    assert any(r["phase"] == "probe" for r in rows)
    assert any(r["phase"] == "bisect" for r in rows)
    assert len(rows) == n, (len(rows), n)   # one row per launch, no duplicates
    print("  test_oom_boundary_search  ok")


def test_native_reached():
    out = io.StringIO()
    summary = {}
    with mock.patch.object(ct.LLM, "run_bench_launch",
                           side_effect=fake_launcher(262144)):
        n, max_safe, first_bad, native, last_good = ct.find_max_ctx(
            base_config(), Path("/tmp/probe.txt"), out, summary)
    assert max_safe == 262144 and native is True
    assert first_bad is None and n == 7   # 4096 .. 262144, no bisection
    assert summary["native_ctx"] == 262144
    print("  test_native_reached  ok")


def test_non_oom_error_aborts():
    def bad_launch(config, cell, spec_flags, prompt_file, launch_id,
                   model_file, ntg=None):
        return {"status": "error", "error": "bad flag --zzz",
                "server_log": "usage: llama-server"}
    with mock.patch.object(ct.LLM, "run_bench_launch", side_effect=bad_launch):
        try:
            ct.find_max_ctx(base_config(), Path("/tmp/probe.txt"),
                            io.StringIO(), {})
        except RuntimeError as e:
            assert "non-OOM" in str(e), e
        else:
            raise AssertionError("non-OOM error did not abort the search")
    print("  test_non_oom_error_aborts  ok")


def test_truncated_prompt():
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "src.txt"
        src.write_text("x" * 100000, encoding="utf-8")
        out = ct.truncated_prompt(src, Path(d), 8192, 1, 3.07)
        n = len(out.read_text())
        assert 256 <= n < 100000, n
        assert n <= int((8192 - 1 - 512) * 3.07)
    print("  test_truncated_prompt  ok")


if __name__ == "__main__":
    try:
        test_classify()
        test_align_and_ladder()
        test_oom_boundary_search()
        test_native_reached()
        test_non_oom_error_aborts()
        test_truncated_prompt()
        print("All self-tests passed.")
    finally:
        pass

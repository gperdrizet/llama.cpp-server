'''Host-side self-tests for inference-api-bench.py (no network, no API key).

Mocks urllib to feed synthetic SSE chunks and checks model filtering, body
construction, exact-timings math and the wall-clock fallback. Run with:

    .venv/bin/python tests/test_inference_api_bench.py
'''

import importlib.util
import json
import re
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("api_bench",
                                              HERE / "inference-api-bench.py")
api = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(api)


class FakeResponse:
    def __init__(self, lines):
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        return iter(self.lines)


def sse(chunks):
    out = []
    for c in chunks:
        out.append(("data: " + json.dumps(c) + "\n\n").encode())
    out.append(b"data: [DONE]\n\n")
    return out


def test_select_models():
    catalog = ["Qwen3.8-27B", "Qwen3.8-Flash-Next", "Qwen3-Coder-Next",
               "BGE-M3", "text-embedding-3-large"]
    kept, skipped = api.select_models(
        catalog, None, re.compile(r"(?i)(embed|bge|rerank)"), [], [])
    assert kept == catalog[:3], kept
    assert set(skipped) == {"BGE-M3", "text-embedding-3-large"}
    kept, _ = api.select_models(catalog, None, None, ["Qwen3.8-27B"], [])
    assert kept == ["Qwen3.8-27B"]
    kept, skipped = api.select_models(catalog, None, None,
                                      [], ["BGE-M3"])
    assert "BGE-M3" in skipped and "Qwen3.8-27B" in kept
    print("  test_select_models  ok")


def test_build_body():
    cfg = {"temperature": 0, "timings_per_token": True,
           "chat_template_kwargs": {"enable_thinking": False}}
    body = api.build_body(cfg, "hello", "M", 512, True)
    assert body["stream"] and body["timings_per_token"] is True
    assert body["stream_options"]["include_usage"] is True
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    body = api.build_body({}, "hello", "M", 8, False)
    assert body["stream"] is False and "timings_per_token" not in body
    print("  test_build_body  ok")


def test_exact_timings():
    chunks = [
        {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}],
         "model": "M",
         "timings": {"prompt_n": 100, "prompt_ms": 500.0,
                     "predicted_n": 10, "predicted_ms": 200.0}},
        {"choices": [], "usage": {"prompt_tokens": 100,
                                  "completion_tokens": 10},
         "timings": {"prompt_n": 100, "prompt_ms": 500.0,
                     "predicted_n": 10, "predicted_ms": 200.0}},
    ]
    with mock.patch.object(api.urllib.request, "urlopen",
                           return_value=FakeResponse(sse(chunks))):
        row = api.stream_completion("http://x/chat/completions", "sk",
                                    {"stream": True}, 30)
    assert row["status"] == "ok" and row["timing_source"] == "timings"
    assert row["pp_tps"] == 200.0, row
    assert row["gen_tps"] == 50.0, row
    assert row["prompt_tokens"] == 100 and row["completion_tokens"] == 10
    assert row["finish_reason"] is None
    assert len(row["token_arrivals_ms"]) == 1
    print("  test_exact_timings  ok")


def test_wall_clock_fallback():
    chunks = [
        {"choices": [{"delta": {"content": "He"}}], "model": "M"},
        {"choices": [{"delta": {"content": "llo"}}], "model": "M"},
        {"choices": [], "usage": {"prompt_tokens": 100,
                                  "completion_tokens": 10}},
    ]
    # t_start, first token, last token, end-of-stream wall
    times = iter([0.0, 0.5, 0.9, 1.0])
    with mock.patch.object(api.urllib.request, "urlopen",
                           return_value=FakeResponse(sse(chunks))), \
         mock.patch.object(api.time, "monotonic", side_effect=lambda: next(times)):
        row = api.stream_completion("http://x/chat/completions", "sk",
                                    {"stream": True}, 30)
    assert row["timing_source"] == "wall-clock", row
    assert row["pp_tps"] == 200.0, row       # 100 tokens / 0.5 s
    assert row["gen_tps"] == 25.0, row       # 10 tokens / 0.4 s
    assert len(row["token_arrivals_ms"]) == 2
    print("  test_wall_clock_fallback  ok")


def test_resolve_prompts():
    p = api.resolve_prompts({"prompt_file": "tests/prompts/qwen3_8_27b_v1.txt"})
    assert len(p) == 1 and p[0].is_file(), p
    print("  test_resolve_prompts  ok")


if __name__ == "__main__":
    test_select_models()
    test_build_body()
    test_exact_timings()
    test_wall_clock_fallback()
    test_resolve_prompts()
    print("All self-tests passed.")

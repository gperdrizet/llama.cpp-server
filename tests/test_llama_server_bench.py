'''Host-side self-test for llama-server-bench.py (no GPU, no real docker).

Verifies the pure logic (log parsing, sweep expansion) and drives run_launch
with run_bench_launch mocked to confirm result rows land in the JSONL and
failures are recorded. Run with:

    .venv/bin/python tests/test_llama_server_bench.py
'''

import importlib.util
import json
import sys
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("llm_srv_bench", HERE / "llama-server-bench.py")
b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b)


def synth_log():
    '''A realistic slice of a llama-server log for a ~131k-token PP run.'''
    return "\n".join([
        "llama_server: HTTP server will listen on 127.0.0.1:8080",
        "llama_model_loader: loaded",
        "slot  0: id 0 | task 1 | prompt processing, n_tokens =    4096, progress = 0.03, t =   11.94 s / 343.05 tokens per second",
        "slot  0: id 0 | task 1 | prompt processing, n_tokens =    8192, progress = 0.06, t =   24.00 s / 341.33 tokens per second",
        "slot  0: id 0 | task 1 | prompt processing, n_tokens =   16384, progress = 0.12, t =   50.00 s / 327.68 tokens per second",
        "slot  0: id 0 | task 1 | prompt processing, n_tokens =   85466, progress = 0.65, t =  367.36 s / 232.88 tokens per second",
        "slot  0: id 0 | task 1 | prompt processing, n_tokens =  131072, progress = 1.00, t =  560.00 s / 234.06 tokens per second",
        "slot  0: id 0 | task 1 | prompt eval time =  560000.00 ms / 131072 tokens (    4.27 ms per token,   234.06 tokens per second)",
        "slot  0: id 0 | task 1 |        eval time =    1200.00 ms /     64 tokens (   18.75 ms per token,    53.33 tokens per second)",
    ])


def test_parse_server_log():
    p = b.parse_server_log(synth_log())
    pts = p["points"]
    assert pts, "no points parsed"
    cums = [x["n_tokens_cum"] for x in pts]
    assert cums == sorted(cums) and cums[-1] == 131072, cums
    assert sum(x["n_tokens_seg"] for x in pts) == 131072, "segment tokens must sum to total"
    assert abs(pts[0]["tps_seg"] - 4096 / 11.94) < 1.0, pts[0]
    assert abs(pts[-1]["tps_cum"] - 131072 / 560.0) < 0.5, pts[-1]
    assert p["prompt_eval"]["n_tokens"] == 131072
    assert abs(p["prompt_eval"]["tps"] - 234.06) < 0.01
    assert p["gen_eval"]["n_tokens"] == 64
    assert abs(p["gen_eval"]["tps"] - 53.33) < 0.01
    print("  test_parse_server_log  ok")


def test_expand_cells():
    base = {
        "--flash-attn": "on",
        "--batch-size": "512,1024",
        "--ubatch-size": "512,1024",
        "--device": "CUDA0,CUDA1",   # atomic
    }
    cells = b.expand_cells(base)
    assert len(cells) == 4, cells
    assert all(c["--device"] == "CUDA0,CUDA1" for c in cells)
    cells = b.valid_batch_ubatch_cells(base)
    # only (512,512),(1024,512),(1024,1024) -- drop (512,1024)
    pairs = sorted((c["--batch-size"], c["--ubatch-size"]) for c in cells)
    assert pairs == [("1024", "1024"), ("1024", "512"), ("512", "512")], pairs
    print("  test_expand_cells  ok")


def test_model_specs():
    specs = b.model_specs_from_config(
        {"model_base": "Qwen3.8-27B", "quants": "UD-Q4_K_M,UD-IQ3_S"})
    assert specs == [
        {"quant": "UD-Q4_K_M", "model_file": "Qwen3.8-27B-UD-Q4_K_M.gguf"},
        {"quant": "UD-IQ3_S", "model_file": "Qwen3.8-27B-UD-IQ3_S.gguf"},
    ], specs
    # plain model_file still works (quant = the file stem)
    specs = b.model_specs_from_config({"model_file": "m.gguf"})
    assert specs == [{"quant": "m", "model_file": "m.gguf"}], specs
    print("  test_model_specs  ok")


def test_run_launch_ok():
    ok_result = {
        "status": "ok",
        "response": {"prompt_tokens": 131072, "completion_tokens": 0,
                     "wall_request_s": 560.5},
        "server_log": synth_log(),
    }
    config = {"model_name": "M", "model_base": "M",
              "models_path": str(HERE), "ntg": 0}
    cell = {"--batch-size": "1024", "--ubatch-size": "512"}

    with open(HERE / "_t_ok.jsonl", "w") as f, \
         mock.patch.object(b, "run_bench_launch", return_value=ok_result):
        counter, launch_sum = b.run_launch(
            "pp", config, cell, {"label": "spec-off", "flags": {}},
            str(HERE / "prompt.txt"), 1, 0, f,
            "M-UD-Q4_K_M.gguf", "UD-Q4_K_M")

    rows = [json.loads(x) for x in open(HERE / "_t_ok.jsonl").read().strip().splitlines()]
    point_rows = [r for r in rows if isinstance(r.get("point"), int)]
    # one row per progress point only — the per-launch summary fields are
    # stamped on every point row (so the last point row carries them)
    assert len(rows) == len(point_rows) == 5, \
        f"expected 5 point rows and nothing else, got {len(rows)} rows / {len(point_rows)} points"
    last = point_rows[-1]
    assert last["status"] == "ok"
    assert last["pp_tps"] == 234.06
    assert last["gen_tps"] == 53.33
    # swept flags are tagged so curves can be separated per condition
    assert last["batch_size"] == "1024"
    assert last["ubatch_size"] == "512"
    assert last["quant"] == "UD-Q4_K_M"
    # pp_auc = the sum of the per-segment marginal rates ("area under the
    # curve" of the decay curve) -- the winner metric of all three phases
    tps_segs = [4096 / 11.94, 4096 / (24.00 - 11.94),
                8192 / (50.00 - 24.00), 69082 / (367.36 - 50.00),
                45606 / (560.00 - 367.36)]
    assert abs(last["pp_auc"] - round(sum(tps_segs), 3)) < 0.5, last["pp_auc"]
    # the returned launch summary is the same dict stamped on the rows
    assert counter == 1 and launch_sum["pp_tps"] == 234.06, (counter, launch_sum)
    print("  test_run_launch_ok  ok")


def test_run_launch_error_recorded():
    config = {"model_name": "M", "model_base": "M",
              "models_path": str(HERE)}
    with open(HERE / "_t_err.jsonl", "w") as f, \
         mock.patch.object(b, "run_bench_launch",
                           return_value={"status": "error", "error": "boom",
                                         "server_log": "x" * 10}):
        counter, launch_sum = b.run_launch(
            "pp", config, {"--batch-size": "1024"},
            {"label": "spec-off", "flags": {}},
            str(HERE / "prompt.txt"), 1, 0, f,
            "M-UD-Q4_K_M.gguf", "UD-Q4_K_M")
    row = json.loads(open(HERE / "_t_err.jsonl").read().strip())
    assert row["status"] == "error" and row["error"] == "boom", row
    # the server log is salvaged too -- a load-timeout / early crash leaves
    # its cause there, and the old code dropped it on this path
    assert row["server_log_tail"] == "x" * 10, row
    assert counter == 1 and launch_sum is None, (counter, launch_sum)
    print("  test_run_launch_error_recorded  ok")


def test_run_launch_no_output():
    config = {"model_name": "M", "model_base": "M",
              "models_path": str(HERE)}
    with open(HERE / "_t_noout.jsonl", "w") as f, \
         mock.patch.object(b, "run_bench_launch",
                           return_value={"status": "ok", "response": {},
                                         "server_log": "no timing lines here"}):
        b.run_launch("pp", config, {"--batch-size": "1024"},
                     {"label": "spec-off", "flags": {}},
                     str(HERE / "prompt.txt"), 1, 0, f,
                     "M-UD-Q4_K_M.gguf", "UD-Q4_K_M")
    row = json.loads(open(HERE / "_t_noout.jsonl").read().strip())
    assert row["status"] == "no_output", row
    print("  test_run_launch_no_output  ok")


def test_run_bench_launch_vram_gate():
    # With less than min_free_vram_gib (default 24 GiB) of total VRAM free —
    # e.g. the production model is resident — run_bench_launch fails fast
    # with a clear error and never touches docker. The model and prompt
    # pre-checks run first, so use files that actually exist.
    prompt = HERE / "_t_vram_prompt.txt"
    prompt.write_text("x")
    try:
        config = {"model_name": "M", "models_path": str(HERE)}
        with mock.patch.object(b, "free_vram_gib", return_value=7.0):
            res = b.run_bench_launch(config, {}, {}, str(prompt), 1,
                                     "test_llama_server_bench.py")
        assert res["status"] == "error" and "VRAM free" in res["error"], res
    finally:
        prompt.unlink()
    print("  test_run_bench_launch_vram_gate  ok")


def test_pick_pp_auc_winner():
    # per-launch rows as the phase loops return them: the launch summary
    # (which carries pp_auc) tagged with its FULL cell (every flag, swept or
    # not) -- so two cells differing only in flash-attn stay separate
    # conditions. Winners are picked by avg pp_auc in ALL three phases.
    cellA = {"--ctx-size": "131072", "--flash-attn": "off",
             "--split-mode": "layer"}
    cellB = {"--ctx-size": "131072", "--flash-attn": "on",
             "--split-mode": "layer"}
    # A averages 5000 (across prompts/runs), B 4800, same quant --
    # flash-attn must separate them; one of A's runs failed.
    rows = [
        {"quant": "UD-Q4_K_M", "cell": cellA, "pp_auc": 5200.0},
        {"quant": "UD-Q4_K_M", "cell": cellA, "pp_auc": 4800.0},  # avg 5000
        {"quant": "UD-IQ3_S", "cell": cellB, "pp_auc": 4800.0},   # avg 4800
        {"quant": "UD-Q4_K_M", "cell": cellA, "pp_auc": None},    # failed, skipped
    ]
    w = b.pick_pp_auc_winner(rows)
    assert w["quant"] == "UD-Q4_K_M" and w["pp_auc"] == 5000.0, w
    # the winning cell is the COMPLETE flag set (later phases inherit it)
    assert w["cell"] == cellA, w
    # no data -> None (the runner then parser.errors before chaining)
    assert b.pick_pp_auc_winner([{"quant": "q", "cell": cellA,
                                  "pp_auc": None}]) is None
    print("  test_pick_pp_auc_winner  ok")


def test_pick_spec_variant_winner():
    variants = [{"label": "spec-off", "flags": {}},
                {"label": "spec-nmax2", "flags": {"--spec-type": "draft-mtp"}}]
    # Winner by pp_auc: spec-off avg 5000, spec-nmax2 avg 4950 (one failed
    # run) -> the CONTROL wins (the expected outcome: spec shouldn't move PP).
    # gen_tps is reference only: spec-off avg 50, nmax2 avg 70.
    rows = {
        "spec-off": [{"pp_auc": 5100.0, "gen_tps": 55.0},
                     {"pp_auc": 4900.0, "gen_tps": 45.0}],
        "spec-nmax2": [{"pp_auc": 5000.0, "gen_tps": 80.0},
                       {"pp_auc": None, "gen_tps": None},
                       {"pp_auc": 4900.0, "gen_tps": 60.0}],
    }
    w = b.pick_spec_variant_winner(rows, variants)
    assert w["variant"]["label"] == "spec-off", w
    assert abs(w["pp_auc"] - 5000.0) < 1e-9, w
    # gen_tps is a per-label reference dict; accept estimate from it
    # (70/50 - 1 = 0.4, upper bound assuming verify is free)
    assert abs(w["gen_tps"]["spec-nmax2"] - 70.0) < 1e-9, w
    assert w["accept_estimate"] == {"spec-nmax2": 0.4}, w
    assert dict(w["ranking"]) == {"spec-off": 5000.0,
                                  "spec-nmax2": 4950.0}, w
    # if a spec variant actually beats the control on pp_auc, it can win
    rows["spec-nmax2"] = [{"pp_auc": 5300.0, "gen_tps": 80.0}]
    w = b.pick_spec_variant_winner(rows, variants)
    assert w["variant"]["label"] == "spec-nmax2", w
    # every variant failed -> None (runner parser.errors, no silent winner)
    assert b.pick_spec_variant_winner(
        {"spec-off": [{"pp_auc": None, "gen_tps": None}],
         "spec-nmax2": [{"pp_auc": None, "gen_tps": None}]},
        variants) is None
    print("  test_pick_spec_variant_winner  ok")


class _FakeContainer:
    '''Minimal stand-in for a docker-py Container: tracks status and the
    stop/start calls so the stop/restore logic is testable without docker.'''

    def __init__(self, name, status="running"):
        self.name = name
        self.status = status
        self.stop_calls = 0
        self.start_calls = 0

    def stop(self):
        self.stop_calls += 1
        self.status = "exited"

    def start(self):
        self.start_calls += 1
        self.status = "running"


class _FakeContainers:
    def __init__(self, containers):
        self._c = containers

    def list(self, all=False):
        assert all, "must list stopped containers too"
        return list(self._c)


class _FakeClient:
    def __init__(self, containers):
        self.containers = _FakeContainers(containers)


def test_production_container_names():
    # default: the production inference server
    assert b.production_container_names({}) == ["llama-cuda"]
    # explicit list passes through (exact names, no comma-splitting applied)
    assert b.production_container_names(
        {"production_containers": ["llama-cuda", "litellm-router"]}) == \
        ["llama-cuda", "litellm-router"]
    # a comma-separated string is split and stripped
    assert b.production_container_names(
        {"production_containers": " llama-cuda , other-server "}) == \
        ["llama-cuda", "other-server"]
    print("  test_production_container_names  ok")


def test_stop_and_restore_production():
    llama = _FakeContainer("llama-cuda", "running")
    llama_down = _FakeContainer("llama-cuda-bak", "exited")   # already down
    router = _FakeContainer("litellm-router", "running")      # not ours
    client = _FakeClient([llama, llama_down, router])
    config = {"stop_production": True}   # default is to stop production

    with mock.patch.object(b.docker, "from_env", return_value=client), \
         mock.patch.object(b.time, "sleep") as slp:
        stopped = b.stop_production_containers(config)

    # only the RUNNING, named container was stopped (exact-name match, so the
    # router — which merely coexists — is untouched)
    assert stopped == [llama], stopped
    assert llama.stop_calls == 1 and llama.status == "exited"
    assert llama_down.stop_calls == 0 and llama_down.status == "exited"
    assert router.stop_calls == 0 and router.status == "running"
    # a brief VRAM-settle wait happens when something was stopped
    assert slp.called, "expected a short settle sleep after stopping"

    # restore uses the captured objects (no docker needed); only the container
    # we actually stopped comes back up
    b.restore_production_containers(stopped)
    assert llama.start_calls == 1 and llama.status == "running"
    assert llama_down.start_calls == 0     # never stopped, never restarted
    assert router.start_calls == 0

    # restore is idempotent: it won't double-start an already-running container
    b.restore_production_containers(stopped)
    assert llama.start_calls == 1, llama.start_calls
    print("  test_stop_and_restore_production  ok")


def test_stop_production_disabled():
    # stop_production=False is a no-op (and never touches docker)
    fe = mock.patch.object(b.docker, "from_env").start()
    try:
        stopped = b.stop_production_containers(
            {"stop_production": False})
        assert stopped == []
        assert not fe.called, "docker must not be touched when disabled"
    finally:
        fe.stop()
    print("  test_stop_production_disabled  ok")


def test_stop_production_docker_unreachable():
    # docker down at stop time: no exception, loud report, empty result (so
    # the run proceeds — the VRAM gate will surface the residency)
    with mock.patch.object(b.docker, "from_env",
                           side_effect=b.docker.errors.DockerException(
                               "no daemon")):
        stopped = b.stop_production_containers({"stop_production": True})
    assert stopped == []
    print("  test_stop_production_docker_unreachable  ok")


if __name__ == "__main__":
    print("Running llama-server-bench self-tests:")
    try:
        test_parse_server_log()
        test_expand_cells()
        test_model_specs()
        test_run_launch_ok()
        test_run_launch_error_recorded()
        test_run_launch_no_output()
        test_run_bench_launch_vram_gate()
        test_pick_pp_auc_winner()
        test_pick_spec_variant_winner()
        test_production_container_names()
        test_stop_and_restore_production()
        test_stop_production_disabled()
        test_stop_production_docker_unreachable()
        print("All self-tests passed.")
    finally:
        for name in ("_t_ok.jsonl", "_t_err.jsonl", "_t_noout.jsonl",
                     "_t_vram_prompt.txt"):
            p = HERE / name
            if p.exists():
                p.unlink()

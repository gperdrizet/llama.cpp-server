'''Host-side self-test for llama-server-optuna-bench.py (no GPU, no real docker).

Verifies the config→search-space logic, the objective math, and drives a
real study.optimize with `run_bench_launch` MOCKED (the launch returns a
synthetic server log) — so the whole Optuna plumbing (SQLite DB, sampling,
per-run files, pruning) is exercised without a GPU. Run with:

    .venv/bin/python tests/test_llama_server_optuna_bench.py
'''

import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path
from unittest import mock

import optuna
import optuna.storages

# Same version shim as the tool under test.
if hasattr(optuna, "TrialPruned"):
    TrialPrunedError = optuna.TrialPruned
else:  # pragma: no cover - older Optuna
    TrialPrunedError = optuna.TrialPrunedError

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "llm_srv_optuna", HERE / "llama-server-optuna-bench.py")
o = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(o)


def synth_log():
    '''A realistic llama-server log slice (same shape as the sibling test).'''
    return "\n".join([
        "slot  0: id 0 | task 1 | prompt processing, n_tokens =    4096, progress = 0.06, t =   11.94 s / 343.05 tokens per second",
        "slot  0: id 0 | task 1 | prompt processing, n_tokens =    8192, progress = 0.12, t =   24.00 s / 341.33 tokens per second",
        "slot  0: id 0 | task 1 | prompt processing, n_tokens =   32768, progress = 1.00, t =  140.00 s / 234.06 tokens per second",
        "slot  0: id 0 | task 1 | prompt eval time =  140000.00 ms / 32768 tokens (    4.27 ms per token,   234.06 tokens per second)",
        "slot  0: id 0 | task 1 |        eval time =    5000.00 ms /    128 tokens (   39.06 ms per token,    25.60 tokens per second)",
    ])


CFG = {
    "model_name": "TEST",
    "model_base": "TEST",
    "quants": "A,B",
    "fixed": {"--ctx-size": "131072", "--device": "CUDA0,CUDA1"},
    "space": {
        "--n-gpu-layers": "-1,12",
        "--flash-attn": "on,off",
        "--batch-size": "4096",
        "--ubatch-size": "512,1024",
    },
    "spec": {"modes": [
        {"name": "off"},
        {"name": "mtp",
         "base_flags": {"--model-draft": "/models/TEST-mtp.gguf",
                        "--spec-type": "draft-mtp"},
         "params": {"--spec-draft-n-max": {"type": "int", "low": 1, "high": 5}}},
        {"name": "dflash",
         "base_flags": {"--model-draft": "/models/other.gguf",
                        "--spec-type": "draft-eagle"},
         "params": {"--spec-top-k": {"type": "choice", "choices": [1, 2, 4]}}},
    ]},
    "prompt_file": ["tests/prompts/qwen3_8_27b_v1.txt",
                    "tests/prompts/qwen3_8_27b_v2.txt"],
    "ntg": 128,
}


def test_space_helpers():
    qmap = o.quant_map(CFG)
    assert qmap == {"A": "TEST-A.gguf", "B": "TEST-B.gguf"}, qmap
    # comma strings and lists both accepted
    sp = o.space_choices({"space": {"--a": "x,y", "--b": ["z"]}})
    assert sp == {"--a": ["x", "y"], "--b": ["z"]}, sp
    try:
        o.space_choices({"space": {"--a": " , "}})
        raise AssertionError("expected ValueError for empty choice list")
    except ValueError:
        pass
    # missing spec block → single "off" mode
    assert o.spec_modes({}) == [{"name": "off"}]
    assert [m["name"] for m in o.spec_modes(CFG)] == ["off", "mtp", "dflash"]
    # prompt list: str comma list and real list
    assert o.prompt_list({"prompt_file": "a.txt,b.txt"}) == ["a.txt", "b.txt"]
    assert o.prompt_list({"prompt_file": ["c.txt"]}) == ["c.txt"]
    print("  test_space_helpers  ok")


def test_sample_params_deterministic_and_constraints():
    class T:
        '''Records suggests; answers them deterministically by param name.'''
        def __init__(self):
            self.calls = []
            self.user = {}
        def suggest_categorical(self, name, choices):
            self.calls.append((name, "cat"))
            if name == "quant":
                return "A"
            if name == "spec.mode":
                return "dflash"
            if name == "spec:--spec-top-k":
                return 4
            return choices[0]
        def suggest_int(self, name, low, high, step=1):
            self.calls.append((name, "int"))
            return low
        def set_user_attr(self, k, v):
            self.user[k] = v

    t = T()
    with mock.patch.object(optuna.Trial, "suggest_categorical",
                           lambda self, n, c: t.suggest_categorical(n, c)), \
         mock.patch.object(optuna.Trial, "suggest_int",
                           lambda self, n, lo, hi, step=1: t.suggest_int(n, lo, hi, step)), \
         mock.patch.object(optuna.Trial, "set_user_attr",
                           lambda self, k, v: t.set_user_attr(k, v)):
        params, spec = o.sample_params(t, CFG)

    assert params["quant"] == "A" and params["--n-gpu-layers"] == "-1"
    assert params["--batch-size"] == "4096" and params["--ubatch-size"] == "512"
    # ubatch(512) <= batch(4096) → not pruned
    # dflash mode: base flags + the choice-typed sub-param
    assert spec == {"--model-draft": "/models/other.gguf",
                    "--spec-type": "draft-eagle", "--spec-top-k": 4}, spec
    # every space key was sampled exactly once
    cats = sorted(n for n, k in t.calls if k == "cat"
                  and n in o.space_choices(CFG))
    assert cats == sorted(o.space_choices(CFG)), t.calls
    print("  test_sample_params_deterministic_and_constraints  ok")


def test_sample_params_ubatch_gt_batch_prunes():
    '''ubatch > batch must prune BEFORE any launch is spent.'''
    class T2:
        user = {}
        def suggest_categorical(self, name, choices):
            if name == "quant":
                return "A"
            if name == "spec.mode":
                return "off"
            if name == "--batch-size":
                return "512"          # force batch=512 < ubatch=1024
            if name == "--ubatch-size":
                return "1024"
            return choices[0]
        def suggest_int(self, name, low, high, step=1):
            return low
        def set_user_attr(self, k, v):
            self.user[k] = v

    t2 = T2()
    try:
        o.sample_params(t2, CFG)
        raise AssertionError("expected TrialPrunedError for ubatch > batch")
    except TrialPrunedError:
        pass
    assert t2.user.get("prune_reason", "").startswith("ubatch 1024 > batch 512"), t2.user
    print("  test_sample_params_ubatch_gt_batch_prunes  ok")


def test_sample_params_quant_cache_without_flash_attn_prunes():
    '''quantized K/V cache with flash-attn off can never load → prune pre-launch.'''
    # its own config: the shared CFG has no cache-type flags, and adding
    # quant ones there would prune inside the full-study tests.
    cfg_fa = dict(CFG)
    cfg_fa["space"] = {
        "--flash-attn": "on,off",
        "--cache-type-k": "q8_0,f16",
        "--cache-type-v": "q8_0,bf16",
        "--batch-size": "4096",
        "--ubatch-size": "512,1024",
    }

    class T3:
        def __init__(self, fa, ck="q8_0", cv="q8_0"):
            self.fa, self.ck, self.cv = fa, ck, cv
            self.user = {}
        def suggest_categorical(self, name, choices):
            if name == "quant":
                return "A"
            if name == "spec.mode":
                return "off"
            if name == "--flash-attn":
                return self.fa
            if name == "--cache-type-k":
                return self.ck
            if name == "--cache-type-v":
                return self.cv
            if name == "--batch-size":
                return "4096"
            if name == "--ubatch-size":
                return "512"
            return choices[0]
        def suggest_int(self, name, low, high, step=1):
            return low
        def set_user_attr(self, k, v):
            self.user[k] = v

    # quantized cache + flash-attn off → pruned
    t3 = T3("off", "q8_0", "q8_0")
    try:
        o.sample_params(t3, cfg_fa)
        raise AssertionError("expected TrialPrunedError for quant cache + fa off")
    except TrialPrunedError:
        pass
    assert "flash-attn" in t3.user.get("prune_reason", ""), t3.user

    # same quant cache but flash-attn on → NOT pruned (valid config)
    t4 = T3("on", "q8_0", "q8_0")
    params4, _ = o.sample_params(t4, cfg_fa)
    assert params4["--cache-type-k"] == "q8_0"

    # flash-attn off but f16/bf16 cache → valid (no quantization)
    t5 = T3("off", "f16", "bf16")
    params5, _ = o.sample_params(t5, cfg_fa)
    assert params5["--cache-type-v"] == "bf16"
    print("  test_sample_params_quant_cache_without_flash_attn_prunes  ok")


def test_sample_params_tensor_split_without_flash_attn_prunes():
    '''tensor split-mode with flash-attn off can never load → prune pre-launch.

    Isolated from the quant-cache prune: this config has NO cache-type flags,
    so only the SPLIT_MODE_TENSOR-requires-flash-attn constraint can fire.
    '''
    cfg_t = dict(CFG)
    cfg_t["space"] = {
        "--flash-attn": "on,off",
        "--split-mode": "layer,tensor",
        "--batch-size": "4096",
        "--ubatch-size": "512,1024",
    }

    class TT:
        def __init__(self, fa, sm):
            self.fa, self.sm = fa, sm
            self.user = {}
        def suggest_categorical(self, name, choices):
            if name == "quant":
                return "A"
            if name == "spec.mode":
                return "off"
            if name == "--flash-attn":
                return self.fa
            if name == "--split-mode":
                return self.sm
            if name == "--batch-size":
                return "4096"
            if name == "--ubatch-size":
                return "512"
            return choices[0]
        def suggest_int(self, name, low, high, step=1):
            return low
        def set_user_attr(self, k, v):
            self.user[k] = v

    # tensor + flash-attn off → pruned (the exact trial-8 failure)
    t = TT("off", "tensor")
    try:
        o.sample_params(t, cfg_t)
        raise AssertionError("expected TrialPrunedError for tensor + fa off")
    except TrialPrunedError:
        pass
    assert "tensor" in t.user.get("prune_reason", "").lower(), t.user

    # tensor + flash-attn on → NOT pruned (production's Qwen3.8-27B config)
    p_on, _ = o.sample_params(TT("on", "tensor"), cfg_t)
    assert p_on["--split-mode"] == "tensor"

    # layer + flash-attn off → NOT pruned (layer split doesn't need flash)
    p_layer, _ = o.sample_params(TT("off", "layer"), cfg_t)
    assert p_layer["--split-mode"] == "layer"
    print("  test_sample_params_tensor_split_without_flash_attn_prunes  ok")


def test_is_full_offload():
    cfg = {"full_offload_ngl": ["999"]}
    assert o.is_full_offload("-1", {})          # sentinel, no config needed
    assert o.is_full_offload("-1", cfg)
    assert o.is_full_offload("999", cfg)        # listed value
    assert o.is_full_offload(" 999 ", cfg)      # whitespace-tolerant
    assert not o.is_full_offload("12", cfg)     # partial
    assert not o.is_full_offload("999", {})     # not listed + not sentinel
    assert o.is_full_offload("999", {"full_offload_ngl": "999,-1"})  # str form
    print("  test_is_full_offload  ok")


def test_sample_params_tensor_partial_offload_prunes():
    '''tensor at PARTIAL offload → prune (the NCCL-crash regime); full stays.'''
    cfg = dict(CFG)
    cfg["full_offload_ngl"] = ["999"]
    cfg["space"] = {
        "--n-gpu-layers": "12,999",
        "--flash-attn": "on",
        "--split-mode": "layer,tensor",
        "--batch-size": "4096",
        "--ubatch-size": "512,1024",
    }

    class TP:
        def __init__(self, sm, ngl):
            self.sm, self.ngl = sm, ngl
            self.user = {}
        def suggest_categorical(self, name, choices):
            if name == "quant":
                return "A"
            if name == "spec.mode":
                return "off"
            if name == "--split-mode":
                return self.sm
            if name == "--n-gpu-layers":
                return self.ngl
            if name == "--flash-attn":
                return "on"
            if name == "--batch-size":
                return "4096"
            if name == "--ubatch-size":
                return "512"
            return choices[0]
        def suggest_int(self, name, low, high, step=1):
            return low
        def set_user_attr(self, k, v):
            self.user[k] = v

    # tensor + partial (12) → pruned
    t = TP("tensor", "12")
    try:
        o.sample_params(t, cfg)
        raise AssertionError("expected TrialPrunedError for tensor + partial")
    except TrialPrunedError:
        pass
    assert "tensor" in t.user.get("prune_reason", "").lower(), t.user

    # tensor + full (999) → NOT pruned (production's regime)
    p_full, _ = o.sample_params(TP("tensor", "999"), cfg)
    assert p_full["--n-gpu-layers"] == "999"

    # tensor + full (-1 sentinel) → NOT pruned
    p_sent, _ = o.sample_params(TP("tensor", "-1"), cfg)
    assert p_sent["--n-gpu-layers"] == "-1"

    # layer + partial (12) → NOT pruned (layer split is the safe path)
    p_layer, _ = o.sample_params(TP("layer", "12"), cfg)
    assert p_layer["--n-gpu-layers"] == "12"
    print("  test_sample_params_tensor_partial_offload_prunes  ok")


def test_sample_params_spec_min_gt_max_prunes():
    '''A spec mode's n_min > n_max can never be honoured → prune pre-launch.'''
    cfg = dict(CFG)
    cfg["spec"] = {"modes": [
        {"name": "off"},
        {"name": "mtp",
         "base_flags": {"--spec-type": "draft-mtp",
                        "--model-draft": "/models/TEST-mtp.gguf"},
         "params": {"--spec-draft-n-max": {"type": "int", "low": 1, "high": 6},
                    "--spec-draft-n-min": {"type": "int", "low": 0, "high": 3}}},
        {"name": "ngram-mod",
         "base_flags": {"--spec-type": "ngram-mod"},
         "params": {"--spec-ngram-mod-n-max": {"type": "int", "low": 16, "high": 96},
                    "--spec-ngram-mod-n-min": {"type": "int", "low": 8, "high": 64}}},
    ]}

    class TS:
        def __init__(self, mode, nmin, nmax):
            self.mode, self.nmin, self.nmax = mode, nmin, nmax
            self.user = {}
        def suggest_categorical(self, name, choices):
            if name == "quant":
                return "A"
            if name == "spec.mode":
                return self.mode
            return choices[0]
        def suggest_int(self, name, low, high, step=1):
            if name.endswith("n-min"):
                return self.nmin
            if name.endswith("n-max"):
                return self.nmax
            return low
        def set_user_attr(self, k, v):
            self.user[k] = v

    for mode, nmin, nmax, pruned in [
            ("mtp", 3, 2, True),
            ("mtp", 1, 4, False),
            ("ngram-mod", 64, 16, True),
            ("ngram-mod", 8, 32, False)]:
        t = TS(mode, nmin, nmax)
        if pruned:
            try:
                o.sample_params(t, cfg)
                raise AssertionError(f"expected prune for {mode} {nmin}>{nmax}")
            except TrialPrunedError:
                pass
            assert "n-min" in t.user.get("prune_reason", ""), t.user
        else:
            o.sample_params(t, cfg)
    print("  test_sample_params_spec_min_gt_max_prunes  ok")


def test_row_is_noop():
    '''Row's no-op signature is a runtime property of the server log.'''
    cell = {"--split-mode": "row"}
    two_gpus = synth_log() + "\n".join([
        "llama_model_load: using device CUDA0 (Tesla P100) - 16000 MiB free",
        "llama_model_load: using device CUDA1 (Tesla P100) - 16000 MiB free",
    ])
    assert not o.row_is_noop(cell, two_gpus)
    assert o.row_is_noop(cell, synth_log())  # no devices listed
    assert o.row_is_noop(cell, "using device CUDA0 only")
    assert o.row_is_noop(cell, "warning: Setting the split mode has no effect")
    assert o.row_is_noop(cell, "device CUDA0 does not support split buffers")
    # other split modes are never checked
    assert not o.row_is_noop({"--split-mode": "layer"}, "no device lines")
    assert not o.row_is_noop({}, "no device lines")
    print("  test_row_is_noop  ok")


def test_first_error_lines():
    log = "\n".join([
        "0.04.290.140 I spec common_specu: adding speculative implementation 'draft-mtp'",
        "0.04.290.146 E llama_init_from_model: quantized V cache requires flash_attn to be enabled",
        "0.04.290.146 E cmn  common_init_: failed to create context with model '/models/x.gguf'",
        "0.04.290.158 I srv    operator(): cleaning up before exit...",
        "0.04.290.893 E srv  server_run_i: exiting due to model loading error",
        "0.55.960.998 E   current device: 0, in function ggml_backend_cuda_comm_allreduce_nccl",
    ])
    assert o.first_error_lines(log, limit=3) == [
        "llama_init_from_model: quantized V cache requires flash_attn to be enabled",
        "cmn  common_init_: failed to create context with model '/models/x.gguf'",
        "srv  server_run_i: exiting due to model loading error",
    ]
    # limit honored; I/W/D lines ignored; empty/None safe
    assert len(o.first_error_lines(log, limit=2)) == 2
    assert o.first_error_lines(log, limit=10)  # 4 E lines total
    assert o.first_error_lines("") == []
    assert o.first_error_lines(None) == []
    assert o.first_error_lines("no severity markers here") == []
    print("  test_first_error_lines  ok")


def test_build_cell():
    cell = o.build_cell({"quant": "A", "--n-gpu-layers": "-1",
                         "--flash-attn": "on"}, CFG)
    assert cell["--ctx-size"] == "131072"
    assert cell["--device"] == "CUDA0,CUDA1"
    assert cell["--n-gpu-layers"] == "-1"
    assert cell["--flash-attn"] == "on"
    assert "quant" not in cell
    assert all(isinstance(v, str) for v in cell.values())
    print("  test_build_cell  ok")


def test_objective_math():
    # the RAW wall-clock helper is retained (recorded as wall_objective_s)
    r = {"status": "ok",
         "response": {"wall_request_s": 175.5, "prompt_tokens": 32768,
                      "completion_tokens": 128},
         "server_log": synth_log()}
    p = o.LLM.parse_server_log(r["server_log"])
    assert o.launch_objective(r, p) == 175.5
    # no wall clock → server summary lines (140.0 s pp + 5.0 s gen)
    r2 = {"status": "ok", "response": {}, "server_log": synth_log()}
    p2 = o.LLM.parse_server_log(r2["server_log"])
    assert abs(o.launch_objective(r2, p2) - 145.0) < 1e-6
    # nothing usable → None (pruned by the caller)
    assert o.launch_objective({"status": "ok", "response": {},
                               "server_log": "no lines here"},
                              o.LLM.parse_server_log("no lines here")) is None

    # the STUDY objective is the standardized T* (not the raw wall clock):
    # P(32768)=140.0 s from synth_log + ~4.0 s generation, ignoring wall=999
    cfg = {"objective": {"prompt_tokens": 32768, "output_tokens": 128,
                         "min_gen_tokens": 64, "gen_fit_min_steps": 2}}
    resp = {"prompt_tokens": 32768, "completion_tokens": 128,
            "finish_reason": "length", "wall_request_s": 999.0,
            "timings": {"predicted_n": 128, "predicted_ms": 4000.0},
            "gen_curve": [[1, 0.0], [32, 1000.0], [64, 2000.0],
                          [96, 3000.0], [128, 4000.0]]}
    std = o.LLM.standardized_objective(p, resp, cfg, 128)
    assert std["objective_mode"] == "stream-fit", std
    assert abs(std["objective_s"] - 144.0) < 1.0, std
    assert std["objective_s"] != 999.0
    assert std["objective_low_confidence"] is False, std
    print("  test_objective_math  ok")


def fake_run_bench_launch(config, cell, spec_flags, prompt_file, launch_id,
                          model_file, ntg=None):
    '''Mocked launch: always "ok" with the synthetic log, a wall clock, and a
    per-token decode curve (flat 31.25 ms/step) so the standardized objective
    is deterministic: P(32768)=140.0 s + 127 steps * 31.25 ms = 143.96875 s.'''
    fake_run_bench_launch.calls.append({
        "prompt": Path(prompt_file).name, "cell": cell,
        "spec_flags": spec_flags, "model_file": model_file, "ntg": ntg})
    return {"status": "ok",
            "response": {"wall_request_s": 100.0 + launch_id,
                         "prompt_tokens": 32768, "completion_tokens": 128,
                         "finish_reason": "length",
                         "timings": {"prompt_n": 32768, "predicted_n": 128,
                                     "predicted_ms": 3000.0},
                         "gen_curve": [[32, 0.0], [64, 1000.0],
                                       [96, 2000.0], [128, 3000.0]]},
            "server_log": synth_log()}


def fake_run_bench_launch_all_fail(config, cell, spec_flags, prompt_file,
                                   launch_id, model_file, ntg=None):
    return {"status": "error", "error": "boom (arg rejected)",
            "server_log": "tail"}


def test_study_runs_and_writes_files():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    tmp = Path(tempfile.mkdtemp(prefix="optuna_test_"))
    try:
        cfg = dict(CFG)
        cfg["storage"] = str(tmp / "study.db")
        cfg["output_dir"] = str(tmp / "out")
        fake_run_bench_launch.calls = []

        with mock.patch.object(o.LLM, "run_bench_launch",
                               fake_run_bench_launch):
            study = o.run_study(cfg, n_trials=4, seed=1)

        assert len(study.trials) == 4
        assert all(t.state.name == "COMPLETE" for t in study.trials), \
            [ (t.number, t.state.name, t.user_attrs.get("error")) for t in study.trials ]
        # objective = mean over the 2 prompts of the standardized T* (NOT the
        # raw wall clock): P(32768)=140.0 s + 127 decode steps * 31.25 ms
        t0 = study.trials[0]
        expected = round(140.0 + 127 * 31.25 / 1000.0, 4)
        assert abs(t0.value - expected) < 1e-6, (t0.value, expected)
        # per-run files: one per (trial, prompt), valid JSON, carry the curve
        files = sorted((tmp / "out" / "runs").glob("*.json"))
        assert len(files) == 8, [f.name for f in files]
        rec = json.loads(files[0].read_text())
        assert rec["trial"] == 0 and rec["run"] in (1, 2)
        assert rec["params"]["quant"] in ("A", "B")
        assert rec["spec_mode"] in ("off", "mtp", "dflash")
        assert rec["status"] == "ok" and rec["objective_s"] is not None
        assert rec["points"] and rec["points"][-1]["n_tokens_cum"] == 32768
        assert rec["prompt_eval"]["n_tokens"] == 32768
        # the study DB is resumable
        s2 = optuna.load_study(study_name=study.study_name,
                               storage=optuna.storages.RDBStorage(f"sqlite:///{cfg['storage']}"))
        assert len(s2.trials) == 4
        # summary.json
        summary = o.write_summary(study, Path(cfg["storage"]), Path(cfg["output_dir"]))
        assert summary["n_complete"] == 4 and summary["best"]["value_s"] > 0
        assert (tmp / "out" / "summary.json").is_file()
        # the launches really got the right ntg + cell shape
        for c in fake_run_bench_launch.calls:
            assert c["ntg"] == 128
            assert c["cell"]["--ctx-size"] == "131072"
            assert c["model_file"] in ("TEST-A.gguf", "TEST-B.gguf")
        print("  test_study_runs_and_writes_files  ok")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        optuna.logging.set_verbosity(optuna.logging.INFO)


def test_all_failed_trial_is_pruned_and_file_written():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    tmp = Path(tempfile.mkdtemp(prefix="optuna_test_"))
    try:
        cfg = dict(CFG)
        cfg["storage"] = str(tmp / "study.db")
        cfg["output_dir"] = str(tmp / "out")

        with mock.patch.object(o.LLM, "run_bench_launch",
                               fake_run_bench_launch_all_fail):
            study = o.run_study(cfg, n_trials=2, seed=1)

        assert len(study.trials) == 2
        assert all(t.state.name == "PRUNED" for t in study.trials), \
            [ (t.number, t.state.name) for t in study.trials ]
        assert study.trials[0].user_attrs.get("prune_reason") == "all prompts failed"
        # per-run files were still written, with the error recorded
        files = sorted((tmp / "out" / "runs").glob("*.json"))
        assert len(files) == 4, [f.name for f in files]
        rec = json.loads(files[0].read_text())
        assert rec["status"] == "error" and rec["error"] == "boom (arg rejected)"
        assert rec["server_log_tail"] == "tail"
        assert rec["objective_s"] is None
        print("  test_all_failed_trial_is_pruned_and_file_written  ok")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        optuna.logging.set_verbosity(optuna.logging.INFO)


def fake_run_bench_launch_row_noop(config, cell, spec_flags, prompt_file,
                                   launch_id, model_file, ntg=None):
    '''Mocked launch: "ok" but the log proves row split did nothing.'''
    return {"status": "ok",
            "response": {"wall_request_s": 100.0 + launch_id,
                         "prompt_tokens": 32768, "completion_tokens": 128},
            "server_log": synth_log() + "\nwarning: Setting the split mode "
                                        "has no effect\n"}


def test_row_noop_trial_is_pruned_and_file_written():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    tmp = Path(tempfile.mkdtemp(prefix="optuna_row_test_"))
    try:
        cfg = dict(CFG)
        cfg["storage"] = str(tmp / "study.db")
        cfg["output_dir"] = str(tmp / "out")
        cfg["space"] = {"--split-mode": "row",
                        "--batch-size": "4096",
                        "--ubatch-size": "512"}
        with mock.patch.object(o.LLM, "run_bench_launch",
                               fake_run_bench_launch_row_noop):
            study = o.run_study(cfg, n_trials=2, seed=1)

        assert len(study.trials) == 2
        assert all(t.state.name == "PRUNED" for t in study.trials), \
            [(t.number, t.state.name) for t in study.trials]
        assert "row" in study.trials[0].user_attrs.get("prune_reason", "")
        files = sorted((tmp / "out" / "runs").glob("*.json"))
        assert len(files) == 4, [f.name for f in files]
        rec = json.loads(files[0].read_text())
        assert rec["status"] == "pruned-noop" and rec["objective_s"] is None
        print("  test_row_noop_trial_is_pruned_and_file_written  ok")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        optuna.logging.set_verbosity(optuna.logging.INFO)


def test_objective_version_guard():
    '''Resuming a study built with a different objective must fail fast.'''
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    tmp = Path(tempfile.mkdtemp(prefix="optuna_guard_"))
    try:
        db = tmp / "study.db"
        storage = optuna.storages.RDBStorage(f"sqlite:///{db}")
        # a fresh DB (no study yet) never raises
        o.check_study_objective(CFG, db, "G")

        study = optuna.create_study(study_name="G", storage=storage,
                                    direction="minimize")
        study.set_user_attr("objective_version", "wall-time-v0")
        try:
            o.check_study_objective(CFG, db, "G")
            raise AssertionError("expected RuntimeError on objective mismatch")
        except RuntimeError as e:
            assert "objective_version" in str(e), e
        # an explicit opt-out is honoured
        o.check_study_objective(
            {"objective": {"allow_objective_migration": True}}, db, "G")
        # a matching version is fine
        study.set_user_attr("objective_version", o.OBJECTIVE_VERSION)
        o.check_study_objective(CFG, db, "G")
        print("  test_objective_version_guard  ok")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        optuna.logging.set_verbosity(optuna.logging.INFO)


def test_main_dry_run():
    '''--dry-run validates and prints the space without touching production.'''
    tmp_path = Path(tempfile.mkdtemp(prefix="optuna_dryrun_"))
    try:
        cfg_file = tmp_path / "cfg.json"
        cfg = dict(CFG)
        # real prompt paths so the dry-run validation passes
        cfg["prompt_file"] = [str(HERE / "prompts" / "qwen3_8_27b_v1.txt"),
                              str(HERE / "prompts" / "qwen3_8_27b_v2.txt")]
        # model + draft files need to exist; point at the prompts as stand-ins
        models = tmp_path / "models"
        models.mkdir()
        for name in ("TEST-A.gguf", "TEST-B.gguf", "TEST-mtp.gguf", "other.gguf"):
            (models / name).write_text("x")
        cfg["models_path"] = str(models)
        cfg_file.write_text(json.dumps(cfg))

        with mock.patch.object(sys, "argv",
                               ["prog", "--config", str(cfg_file), "--dry-run"]):
            rc = o.main()
        assert rc == 0
        print("  test_main_dry_run  ok")
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


if __name__ == "__main__":
    print("llama-server-optuna-bench self-test:")
    test_space_helpers()
    test_sample_params_deterministic_and_constraints()
    test_sample_params_ubatch_gt_batch_prunes()
    test_sample_params_quant_cache_without_flash_attn_prunes()
    test_sample_params_tensor_split_without_flash_attn_prunes()
    test_is_full_offload()
    test_sample_params_tensor_partial_offload_prunes()
    test_sample_params_spec_min_gt_max_prunes()
    test_row_is_noop()
    test_first_error_lines()
    test_build_cell()
    test_objective_math()
    test_study_runs_and_writes_files()
    test_all_failed_trial_is_pruned_and_file_written()
    test_row_noop_trial_is_pruned_and_file_written()
    test_objective_version_guard()
    test_main_dry_run()
    print("all tests passed")

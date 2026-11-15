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
    # wall clock is preferred
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
    print("  test_objective_math  ok")


def fake_run_bench_launch(config, cell, spec_flags, prompt_file, launch_id,
                          model_file, ntg=None):
    '''Mocked launch: always "ok" with the synthetic log + a wall clock.'''
    fake_run_bench_launch.calls.append({
        "prompt": Path(prompt_file).name, "cell": cell,
        "spec_flags": spec_flags, "model_file": model_file, "ntg": ntg})
    return {"status": "ok",
            "response": {"wall_request_s": 100.0 + launch_id,
                         "prompt_tokens": 32768, "completion_tokens": 128},
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
        # objective = mean wall clock over the 2 prompts
        t0 = study.trials[0]
        n_prompts = len(cfg["prompt_file"])
        # trial 0 launches first: launch_id 0 and 1 → wall 100.0, 101.0
        expected = sum([100.0 + 0, 100.0 + 1]) / n_prompts
        assert abs(t0.value - expected) < 1e-9, (t0.value, expected)
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
    test_build_cell()
    test_objective_math()
    test_study_runs_and_writes_files()
    test_all_failed_trial_is_pruned_and_file_written()
    test_main_dry_run()
    print("all tests passed")

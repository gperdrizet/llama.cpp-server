#!/usr/bin/env python3
'''Benchmark every text model exposed by a live OpenAI-compatible inference API.

Defaults to promptlyapi.com (see /home/siderealyear/promptly). Unlike the local
tools in this directory this one drives NO docker: it discovers the model
catalog with ``GET /v1/models``, skips embedding models (the catalog doesn't
flag them, so exclusion is a configurable regex / explicit list), and sends a
short streaming chat-completion to each remaining model.

Timing model
------------
Each request streams with ``stream_options.include_usage`` and (for llama.cpp
backends) ``timings_per_token``. When the backend returns a ``timings`` object
the EXACT server-side figures are used:

    pp_tps  = timings.prompt_n / timings.prompt_ms
    gen_tps = timings.predicted_n / timings.predicted_ms

Otherwise the tool falls back to wall-clock instrumentation of the SSE stream:
time-to-first-token (which includes prompt processing) gives PP rate, and the
first-to-last-token span gives generation rate. The raw fields (usage, finish
reason, final timings, per-token arrival curve, wall time) are always retained
so any rate can be recomputed offline.

A cold model load is excluded by a tiny warmup request per model before its
measured requests; the warmup duration is recorded in the summary.

Output: ``tests/results/inference-api-bench/<name>_<timestamp>.jsonl`` (one row
per measured request) plus a ``<name>_<timestamp>.summary.json`` with per-model
means. Failed models are recorded, never silently dropped.
'''

import argparse
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# Works both as `python tests/bench/inference_api_bench.py` and as an import.
try:
    from . import paths
except ImportError:  # direct script execution: this file's dir is sys.path[0]
    import paths

DEFAULT_BASE_URL = "https://promptlyapi.com/v1"
# The router's /v1/models has no capability flag; these names are embeddings.
DEFAULT_EXCLUDE_RE = r"(?i)(embed|bge|rerank)"


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_prompts(cfg: dict) -> list:
    raw = cfg.get("prompt_file")
    if raw is None:
        return []
    items = raw if isinstance(raw, list) else str(raw).split(",")
    out = []
    for item in items:
        p = Path(item.strip()).expanduser()
        p = p if p.is_absolute() else paths.REPO_ROOT / p
        out.append(p.resolve())
    return out


def compile_pattern(pattern, default=None):
    if not pattern and default is None:
        return None
    return re.compile(pattern if pattern else default)


def select_models(models: list, include_re, exclude_re,
                  include: list, exclude: list) -> tuple[list, dict]:
    '''Apply explicit lists first, then regex filters. Returns (kept, reasons).'''
    reasons = {}
    kept = []
    for name in models:
        if include and name not in include:
            reasons[name] = "not in --models / include_models"
            continue
        if name in (exclude or []):
            reasons[name] = "explicitly excluded"
            continue
        if include_re and not include_re.search(name):
            reasons[name] = "does not match include_regex"
            continue
        if exclude_re and exclude_re.search(name):
            reasons[name] = "matches exclude regex (embedding/rerank model)"
            continue
        kept.append(name)
    return kept, reasons


def request_json(url: str, api_key: str, body: dict, timeout: int) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"},
        method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def list_models(base_url: str, api_key: str, timeout: int) -> list:
    url = base_url.rstrip("/") + "/models"
    req = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = json.loads(r.read().decode("utf-8", "replace"))
    return [m["id"] for m in payload.get("data", []) if m.get("id")]


def error_text(exc) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        try:
            body = exc.read().decode("utf-8", "replace")[:1000]
        except Exception:
            body = ""
        return f"HTTP {exc.code} {exc.reason}: {body}"
    return repr(exc)


def stream_completion(url: str, api_key: str, body: dict,
                      timeout: int) -> dict:
    '''One streaming chat completion -> parsed measurement dict.

    Raises on HTTP/transport errors; the caller records them.
    '''
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"},
        method="POST")
    chunks = []
    raw_lines = []
    arrivals = []           # [ms since first token, chars seen] per content chunk
    first_t = last_t = None
    t_start = time.monotonic()

    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            raw_lines.append(line)
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            chunks.append(chunk)
            choices = chunk.get("choices") or []
            delta = choices[0].get("delta", {}) if choices else {}
            content = delta.get("content") or delta.get("reasoning_content") or ""
            if content:
                now = time.monotonic()
                if first_t is None:
                    first_t = now
                last_t = now
                arrivals.append([round((now - first_t) * 1000, 2), len(content)])

    wall = time.monotonic() - t_start
    if not chunks:
        # A non-streaming response: still useful for usage/timings.
        try:
            chunks = [json.loads("".join(raw_lines))]
        except json.JSONDecodeError:
            pass

    usage, timings, finish_reason, served_model = {}, {}, None, None
    for c in chunks:
        if c.get("usage"):
            usage = c["usage"]
        if c.get("timings"):
            timings = c["timings"]
        if c.get("model"):
            served_model = c["model"]
        for choice in c.get("choices") or []:
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

    prompt_n = usage.get("prompt_tokens") or timings.get("prompt_n")
    comp_n = usage.get("completion_tokens") or timings.get("predicted_n")
    prompt_ms = timings.get("prompt_ms")
    gen_ms = timings.get("predicted_ms")
    source = "timings" if (prompt_ms and gen_ms) else "wall-clock"

    if prompt_ms and gen_ms:
        pp_s, gen_s = float(prompt_ms) / 1000.0, float(gen_ms) / 1000.0
        ttft_s = pp_s
    else:
        pp_s = ttft_s = (first_t - t_start) if first_t else wall
        gen_s = (last_t - first_t) if (first_t and last_t) else 0.0

    return {
        "status": "ok",
        "served_model": served_model,
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_n,
        "completion_tokens": comp_n,
        "pp_s": round(pp_s, 4),
        "gen_s": round(gen_s, 4),
        "ttft_s": round(ttft_s, 4),
        "wall_request_s": round(wall, 4),
        "pp_tps": round(prompt_n / pp_s, 2) if prompt_n and pp_s > 0 else None,
        "gen_tps": round(comp_n / gen_s, 2) if comp_n and gen_s > 0 else None,
        "timing_source": source,
        "timings": timings or None,
        "usage": usage or None,
        "token_arrivals_ms": arrivals,
        "stream_chunks": len(chunks),
    }


def build_body(cfg: dict, prompt: str, model: str, max_tokens: int,
               stream: bool) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": cfg.get("temperature", 0),
        "stream": stream,
    }
    if cfg.get("chat_template_kwargs"):
        body["chat_template_kwargs"] = cfg["chat_template_kwargs"]
    if cfg.get("reasoning_effort") is not None:
        body["reasoning_effort"] = cfg["reasoning_effort"]
    if stream:
        opts = {"include_usage": True}
        if cfg.get("timings_per_token", True):
            body["timings_per_token"] = True
        body["stream_options"] = opts
    return body


def measure_model(base_url: str, api_key: str, cfg: dict, model: str,
                  prompts: list, runs: int, result_file, summary: dict,
                  timeout: int):
    url = base_url.rstrip("/") + "/chat/completions"
    max_tokens = int(cfg.get("ntg", 512))
    warm = {"status": "skipped"}

    if cfg.get("warmup", True):
        warm_body = build_body(cfg, "Reply with exactly one word: pong", model,
                               int(cfg.get("warmup_max_tokens", 8)), False)
        t0 = time.monotonic()
        try:
            reply = request_json(url, api_key, warm_body, timeout)
            warm = {"status": "ok",
                    "wall_s": round(time.monotonic() - t0, 3),
                    "served_model": reply.get("model")}
        except Exception as exc:
            warm = {"status": "error", "error": error_text(exc)}
        print(f"  warmup {model}: {warm['status']}"
              + (f" ({warm['wall_s']}s)" if warm.get("wall_s") else "")
              + (f" -- {warm['error'][:120]}" if warm.get("error") else ""))

    entry = {"warmup": warm, "runs": []}
    summary["models"][model] = entry
    if warm["status"] == "error":
        entry["status"] = "warmup_failed"
        return

    for prompt in prompts:
        for run_id in range(1, runs + 1):
            body = build_body(cfg, prompt.read_text(encoding="utf-8"), model,
                              max_tokens, True)
            row = {
                "name": cfg.get("name"), "base_url": base_url,
                "model": model, "prompt": prompt.name, "run": run_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            try:
                row.update(stream_completion(url, api_key, body, timeout))
            except urllib.error.HTTPError as exc:
                # Some endpoints reject the llama.cpp-only timings field.
                if (cfg.get("timings_per_token", True)
                        and "timings_per_token" in body):
                    retry = dict(body)
                    retry.pop("timings_per_token", None)
                    try:
                        row.update(stream_completion(url, api_key, retry,
                                                     timeout))
                        row["timings_retry"] = "dropped timings_per_token"
                    except Exception as exc2:
                        row.update({"status": "error",
                                    "error": error_text(exc2)})
                else:
                    row.update({"status": "error", "error": error_text(exc)})
            except Exception as exc:
                row.update({"status": "error", "error": error_text(exc)})

            result_file.write(json.dumps(row, separators=(",", ":")) + "\n")
            result_file.flush()
            entry["runs"].append({
                "prompt": prompt.name, "run": run_id,
                "status": row["status"],
                "pp_tps": row.get("pp_tps"), "gen_tps": row.get("gen_tps"),
                "error": row.get("error"),
            })
            print(f"    {prompt.name} run {run_id}: {row['status']}"
                  + (f"  pp={row['pp_tps']} t/s  gen={row['gen_tps']} t/s"
                     if row["status"] == "ok" else f"  {row.get('error', '')[:120]}"))

    ok = [r for r in entry["runs"] if r["status"] == "ok"]
    entry["status"] = "ok" if ok else "all_failed"
    for key in ("pp_tps", "gen_tps"):
        vals = [r[key] for r in ok if r.get(key) is not None]
        entry[f"mean_{key}"] = round(statistics.mean(vals), 2) if vals else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark every text model on a live OpenAI-compatible API")
    parser.add_argument("--config", required=True, help="benchmark JSON config")
    parser.add_argument("--url", default=None, help="override base_url")
    parser.add_argument("--api-key", default=None,
                        help="override the API key (prefer env / --api-key-env)")
    parser.add_argument("--models", default=None,
                        help="comma-separated model allow-list")
    parser.add_argument("--exclude", default=None,
                        help="regex of model names to skip")
    parser.add_argument("--runs", type=int, default=None,
                        help="override config runs")
    parser.add_argument("--ntg", type=int, default=None,
                        help="override config max output tokens")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate config/prompts; no network calls")
    parser.add_argument("--list-models", action="store_true",
                        help="query /v1/models, print the filtered list, exit")
    args = parser.parse_args()

    cfg = load_config(Path(args.config).expanduser().resolve())
    if args.url:
        cfg["base_url"] = args.url
    cfg.setdefault("base_url", DEFAULT_BASE_URL)
    if args.runs is not None:
        cfg["runs"] = args.runs
    if args.ntg is not None:
        cfg["ntg"] = args.ntg
    if args.exclude:
        cfg["exclude_regex"] = args.exclude

    prompts = resolve_prompts(cfg)
    if not prompts:
        parser.error("config has no prompt_file")
    missing = [str(p) for p in prompts if not p.is_file()]
    if missing:
        parser.error(f"prompt file(s) not found: {', '.join(missing)}")

    include_re = compile_pattern(cfg.get("include_regex"))
    exclude_re = compile_pattern(cfg.get("exclude_regex"), DEFAULT_EXCLUDE_RE)
    include = [m.strip() for m in args.models.split(",") if m.strip()] \
        if args.models else list(cfg.get("include_models") or [])
    exclude = list(cfg.get("exclude_models") or [])
    timeout = int(cfg.get("request_timeout", 3600))

    api_key = args.api_key or os.environ.get(
        cfg.get("api_key_env", "PROMPTLY_API_KEY"))
    if not api_key and not args.dry_run:
        parser.error(f"no API key: pass --api-key or set "
                     f"{cfg.get('api_key_env', 'PROMPTLY_API_KEY')}")

    name = cfg.get("name", Path(args.config).stem)
    print(f"inference-api-bench: {name}")
    print(f"  base_url : {cfg['base_url']}")
    print(f"  prompts  : {[p.name for p in prompts]}")
    print(f"  ntg={cfg.get('ntg', 512)}  runs={cfg.get('runs', 1)}  "
          f"warmup={cfg.get('warmup', True)}  "
          f"timings_per_token={cfg.get('timings_per_token', True)}")
    include_desc = (str(include) if include
                    else include_re.pattern if include_re else "all")
    print(f"  filters  : include={include_desc}  "
          f"exclude_regex={exclude_re.pattern if exclude_re else None}  "
          f"explicit_exclude={exclude}")

    if args.dry_run:
        print("  (dry run: no network calls)")
        return 0

    try:
        all_models = list_models(cfg["base_url"], api_key, timeout)
    except Exception as exc:
        print(f"!! could not list models: {error_text(exc)}", file=sys.stderr)
        return 2
    models, skipped = select_models(all_models, include_re, exclude_re,
                                    include, exclude)
    print(f"  catalog  : {all_models}")
    print(f"  testing  : {models}")
    for name_, why in skipped.items():
        print(f"    skip {name_}: {why}")
    if args.list_models:
        return 0
    if not models:
        print("no models to benchmark")
        return 2

    output_dir = paths.RESULTS_DIR / "inference-api-bench"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_file = output_dir / f"{name}_{timestamp}.jsonl"
    summary = {
        "name": name, "base_url": cfg["base_url"],
        "config": cfg, "prompts": [p.name for p in prompts],
        "catalog": all_models, "skipped": skipped, "models": {},
        "result_file": str(result_file),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    print(f"  results  : {result_file}")
    with open(result_file, "a", encoding="utf-8") as f:
        for model in models:
            measure_model(cfg["base_url"], api_key, cfg, model, prompts,
                          int(cfg.get("runs", 1)), f, summary, timeout)

    summary_file = result_file.with_suffix(".summary.json")
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nPer-model means:")
    for model, entry in summary["models"].items():
        print(f"  {model:<32} pp={entry.get('mean_pp_tps')} t/s  "
              f"gen={entry.get('mean_gen_tps')} t/s  [{entry['status']}]")
    print(f"Wrote results: {result_file}")
    print(f"Wrote summary: {summary_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

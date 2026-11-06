'''Script to run Dockerized llama batched-bench for performance tuning

Three-phase tuning run, writes output to single jsonl file with `"phase"`,
for individual analysis later.

  Phase 1 -- fix batch/ubatch for prompt preprocessing.
      Crosses `--batch-size` x `--ubatch-size` from the config at ONE set
      context occupancy. Each (batch, ubatch) pair is one container
      launch. Winner = the cell with the highest `speed_pp` at that
      occupancy, hard, averaged across runs.

  Phase 2 -- fix speculative decoding.
      Runs the config's `spec_variants` (which must include a spec-OFF
      control) at the config's set occupancy and `-ntg` tokens, using the
      phase-1 batch/ubatch. Winner = the variant with the highest
      `speed_tg` (the control may win -> phase 3 then runs spec-OFF).

  Phase 3 -- characterize performance vs context occupancy.
      One launch with the phase-1 batch/ubatch and the phase-2 spec
      settings; the binary sweeps `-npp` (the occupancy list) in-process.
      `speed_pp` vs `pp` = PP rate vs occupancy; `speed_tg` vs `pp` =
      generation rate vs occupancy (the TG phase runs with the `pp`-token
      prompt already in KV -- memory is NOT cleared between the phases --
      at the fixed `-c` capacity).

Chaining: winners are selected automatically from the completed phase
(data from all runs, averaged per cell). The `overrides` block in the
config can force a phase's fixed inputs instead of (or to test against)
the auto pick; `--phase N` re-runs a single phase, which then REQUIRES
`overrides` for every earlier phase.

Sweep model. The config maps a llama-batched-bench flag to either a single
value or a comma-separated list of values. Lists are swept, but where they
are swept depends on the flag:

  - `-npp` / `-ntg` / `-npl` are expanded IN-PROCESS by llama-batched-bench
    itself (one model load, many measurement cells per launch). The script
    passes their lists straight through. This is also how you get "rate as
    a function of cache occupancy at a FIXED context size": `-c` stays at
    the server value while `-npp` sweeps, so the `speed_pp` and `speed_tg`
    columns are both measured at growing KV occupancy.

  - Every other list-valued flag is expanded by this script. A new
    llama.cpp container is launched once per combination.

  * `--device` (and anything in NON_SWEEP_FLAGS) is NEVER swept even though
    its value contains a comma, e.g. "CUDA0,CUDA1" is one value (both GPUs),
    not two.

Speculative decoding is an outer axis, expressed in the config as
`spec_variants`: a list of dicts, each with a `label` and the extra spec
`flags` to append for that launch (empty = the control). Spec engages at
context level, so it cannot be swept in-process. The first variant should
always be the spec-OFF control.

A run is one full pass through every phase. `--runs N` repeats the entire
chain N times and appends every run to the same output file.
'''

import argparse
import json
import math
from itertools import product
from pathlib import Path
from datetime import datetime

import docker
import docker.errors

# Flags that llama-batched-bench sweeps in-process (passed through as lists).
NATIVE_SWEEP_FLAGS = {"-npp", "-ntg", "-npl"}

# Flags whose comma-separated values are atomic, never to be swept.
NON_SWEEP_FLAGS = {"--device"}


def split_lists(parameters: dict) -> dict:
    '''Split each config value into a list of its sweep points.

    A value of "4096" -> ["4096"]; "512,1024,2048" -> ["512", "1024", "2048"].
    '''

    return {key: [v.strip() for v in value.split(",")] for key, value in parameters.items()}


def phase_launch_params(shared: dict, phase: dict, fixed: dict):
    '''Assemble the launch parameters for one phase.

    `shared` = the live-server invariants; `phase` = the phase section of the
    config (native-sweep values + the phase-1 sweep lists); `fixed` = the
    concrete batch/ubatch values this phase must use (auto-picked or
    overridden).
    '''

    params = dict(shared)
    params.update(phase)
    params.update(fixed)

    return params


def expand_outer_cells(parameters: dict) -> list:
    '''Return the list of parameter dicts, one per outer sweep cell.

    Only list-valued flags that the binary does NOT expand in-process (and
    that are not atomic) are varied here; native-sweep flags keep their full
    list for the binary. Single-valued flags pass through unchanged.
    '''

    split = split_lists(parameters)
    outer_keys = [
        k for k in parameters
        if k not in NATIVE_SWEEP_FLAGS
        and k not in NON_SWEEP_FLAGS
        and len(split[k]) > 1
    ]

    if not outer_keys:
        return [parameters]

    combos = product(*(split[k] for k in outer_keys))
    out = []

    for values in combos:
        cell = dict(parameters)

        for key, value in zip(outer_keys, values):
            cell[key] = value

        out.append(cell)

    return out


def build_spec_variants(config: dict) -> list:
    '''Return the list of spec variants; an empty/missing list = one spec-OFF control.'''

    variants = config.get("spec_variants")

    if not variants:
        return [{"label": "spec-off", "flags": {}}]

    return variants


def run_llama_bench(
        models_path: str,
        config: dict,
        cell_params:
        dict, spec_flags: dict,
        launch_id: int
) -> str:
    '''Launch one llama-batched-bench container for a single (cell, spec) launch.

    `cell_params` is the fully-resolved flag map (native-sweep flags may still
    carry their list for the binary); `spec_flags` are the extra
    speculative-decoding flags for this variant. Returns the container's
    stdout (JSONL lines).
    '''

    # Connect to docker container
    client = docker.from_env()

    # Prepare the benchmark command with additional parameters
    bench_command = [
        "--model", f"/models/{config['model_file']}",
        "--output-format", "jsonl"
    ]

    for key, value in cell_params.items():
        bench_command.extend([key, value])

    for key, value in spec_flags.items():
        bench_command.extend([key, value])

    # Run the container and block until it finishes
    container_output = client.containers.run(
        image="ghcr.io/ggml-org/llama.cpp:full-cuda",
        name=f"llama-bench-{launch_id}",
        entrypoint="/app/llama-batched-bench",
        command=bench_command,
        volumes={
            Path(models_path).resolve().as_posix(): {'bind': '/models', 'mode': 'ro'}
        },
        device_requests=[
            docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])
        ],
        remove=True,   # Automatically remove the container after it exits
        detach=False,  # Blocks until execution completes
        stdout=True,   # Captures the clean JSONL data
        stderr=False   # Ignores initialization clutter completely
    )

    # Get and return results
    return container_output.decode('utf-8')


def run_phase(
        phase: str,
        phase_name: str,
        models_path: str,
        config: dict,
        runs: int,
        launch_counter: int,
        result_file,
        spec_label: str,
        launch_params: dict,
        spec_flags: dict,
        label_desc: str
) -> tuple[list[tuple[int, dict]], int]:
    '''Run one phase: `launch_params` is repeated `runs` times, each launch
    tagged with `spec_label`. Returns (rows, launch_counter) where each row
    is a (run_id, row-dict) pair with the measured values (no tags).'''

    rows = []

    for run_id in range(1, runs + 1):
        print(f"  [{phase_name}] run {run_id}/{runs}: {label_desc}")
        launch_counter += 1

        try:
            csv_data = run_llama_bench(
                str(models_path),
                config,
                launch_params,
                spec_flags,
                launch_counter
            )

        except docker.errors.ContainerError as e:
            # One bad launch (arg rejected, OOM, ...) must not kill the
            # whole chain. Any partial JSONL already emitted is kept.
            print(f"    !! container exited {e.exit_code}; skipping run "
                  f"(recovered {len(e.stdout or b'')} bytes of partial output)")

            csv_data = (e.stdout or b"").decode("utf-8", "replace")

        for line in csv_data.splitlines():
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)

            except json.JSONDecodeError:
                continue

            tagged = {"phase": phase, "run": run_id, "spec": spec_label}
            tagged.update(row)
            result_file.write(json.dumps(tagged, separators=(',', ':')) + "\n")
            rows.append((run_id, row))

    result_file.flush()
    return rows, launch_counter  # type: tuple[list[tuple[int, dict]], int]


def average_by(rows: list[tuple[int, dict]], key_fn, value_fn) -> dict:
    '''Average value_fn(row) over rows grouped by key_fn(row).'''

    acc = {}

    for _, row in rows:
        k = key_fn(row)
        v = value_fn(row)

        if not isinstance(v, (int, float)) or isinstance(v, bool) or math.isnan(v):
            continue

        s, n = acc.get(k, (0.0, 0))
        acc[k] = (s + v, n + 1)

    return {k: s / n for k, (s, n) in acc.items()}


def pick_phase1_winner(rows: list[tuple[int, dict]]) -> dict:
    '''Winner = the (batch, ubatch) cell with the highest average speed_pp.'''

    avg = average_by(
        rows, 
        lambda r: (r["n_batch"], r["n_ubatch"]),
        lambda r: r["speed_pp"]
    )

    if not avg:
        return None

    (best_batch, best_ubatch), best_speed = max(avg.items(), key=lambda kv: kv[1])

    return {
        "--batch-size": str(best_batch),
        "--ubatch-size": str(best_ubatch),
        "speed_pp": best_speed
    }


def pick_phase2_winner(variant_rows: dict[str, list[tuple[int, dict]]], spec_variants: list) -> dict:
    '''Winner = the spec variant with the highest average speed_tg.

    `variant_rows` maps each variant label to its (run_id, row) pairs, so a
    failed launch (uneven row count) cannot shift another variant's block.
    '''
    speeds = []

    for variant in spec_variants:
        vals = [r["speed_tg"] for _, r in variant_rows.get(variant["label"], [])
                if isinstance(r.get("speed_tg"), (int, float))]
    
        speeds.append(sum(vals) / len(vals) if vals else float("nan"))
    
    if not speeds:
        return None

    def sort_key(i):
        return speeds[i] if speeds[i] == speeds[i] else float("-inf")  # nan -> worst

    best_i = max(range(len(speeds)), key=sort_key)

    return {
        "variant": spec_variants[best_i],
        "speed_tg": speeds[best_i],
        "ranking": list(zip([v["label"] for v in spec_variants], speeds))
    }


if __name__ == "__main__":

    # Parse run configuration from command line arguments
    parser = argparse.ArgumentParser(description="Run llama batched-bench")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Configuration file"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Repeat the full phase chain this many times (default: 3); "
             "all runs are appended to the same output file"
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3],
        help="Run only this phase; earlier phases' fixed inputs "
             "then come from the config's overrides block"
    )
    args = parser.parse_args()

    # Load run configuration from the specified file
    with open(args.config, "r") as f:
        config = json.load(f)

    # Extract shared parameters, phase definitions, and overrides from the config
    shared = config.get("parameters", {})
    phases = config.get("phases", {})
    overrides = config.get("overrides", {})

    # Build the speculative decoding variants from the config
    spec_variants = build_spec_variants(config)

    # Ensure that phase 1 overrides are provided if explicitly running phase 2 or 3
    if args.phase in (2, 3) and not overrides.get("phase1"):
        parser.error(
            f"--phase {args.phase} needs overrides.phase1 "
            "(\"--batch-size\"/\"--ubatch-size\") in the config, since phase 1 "
            "does not run this invocation"
        )

    # Notify if running phase 3 with no phase 2 or phase 3 spec overrides
    if args.phase == 3 and "phase2" not in overrides and "phase3_spec" not in overrides:
        print("note: --phase 3 with no overrides.phase2/phase3_spec runs spec-OFF")

    print(f'Running benchmark for model with config:')
    print(json.dumps(config, indent=4))

    # Get path to project root
    root_path = Path(__file__).resolve().parents[1]

    # Set paths for models, run configuration and output files
    models_path = root_path / "models"
    output_path = root_path / "tests/results/llama-bench-tuner"
    output_path.mkdir(parents=True, exist_ok=True)

    # One timestamped file per invocation; every phase and every run within it
    # lands in this same file.
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_file = Path(output_path / f"{config['model_name']}_{timestamp}.jsonl")

    # Calculate the number of launches for each phase
    n_p1 = len(expand_outer_cells(phase_launch_params(shared, phases["1"], {}))) \
        if (args.phase is None or args.phase == 1) and "1" in phases else 0

    n_p2 = len(spec_variants) if (args.phase is None or args.phase == 2) and "2" in phases else 0
    n_p3 = 1 if (args.phase is None or args.phase == 3) and "3" in phases else 0

    # Print a summary of the planned launches for each phase
    print(f"Phase 1: {n_p1} launches/run (batch/ubatch cross, spec-off) | "
          f"Phase 2: {n_p2} launches/run (spec variants) | "
          f"Phase 3: {n_p3} launches/run (occupancy sweep) | runs: {args.runs} "
          f"({(n_p1 + n_p2 + n_p3) * args.runs} container launches total)")

    print(f"Native in-process sweeps: phase 1 pp={phases.get('1', {}).get('-npp')}, "
          f"phase 2 pp={phases.get('2', {}).get('-npp')} ntg={phases.get('2', {}).get('-ntg')}, "
          f"phase 3 pp={phases.get('3', {}).get('-npp')}")

    launch_counter = 0
    summary_data = {
        "model": config["model_name"],
        "runs": args.runs,
        "result_file": str(result_file), "phases": {}
    }
    phase1_winner = None

    with open(result_file, "a") as f:

        # PHASE 1: batch/ubatch scan for prompt preprocessing
        if args.phase is None or args.phase == 1:
            print("\n=== Phase 1: scan batch/ubatch for best prompt preprocessing ===")

            # Generate the launch cells for phase 1
            base = phase_launch_params(shared, phases["1"], {})
            cells = expand_outer_cells(base)
            rows = []

            # Loop on cells for this phase, collecting results in rows
            for cell in cells:
                desc = f"batch={cell.get('--batch-size')}, ubatch={cell.get('--ubatch-size')}"

                r, launch_counter = run_phase(
                    1, "phase 1", models_path, config, args.runs, launch_counter,
                    f, "spec-off", cell, {}, desc
                )
                rows.extend(r)

            # Compute the average speed_pp for each (batch, ubatch) combination
            avg = average_by(
                rows,
                lambda r: (r["n_batch"], r["n_ubatch"]),
                lambda r: r["speed_pp"]
            )

            # Print the ranking of (batch, ubatch) combinations by average speed_pp
            print(f"\nPhase 1 ranking (avg speed_pp @ pp={base.get('-npp')}):")
            for (b, u), s in sorted(avg.items(), key=lambda kv: (-kv[1]
                                                                 if kv[1] == kv[1]
                                                                 else float("inf"))):

                print(
                    f"  batch={b:>5}  ubatch={u:>5}  speed_pp="
                    f"{'NaN' if s != s else format(s, '9.1f')} t/s"
                )

            # If we are only running phase 1, stop here. Otherwise, determine the phase 1 winner
            if args.phase == 1:
                print("\n(Stopped: --phase 1. Pin the winner in overrides.phase1 to run phase 2.)")

            else:
                phase1_winner = pick_phase1_winner(rows)

                if phase1_winner is None:
                    parser.error("phase 1 produced no speed_pp data; cannot chain")

                summary_data["phases"]["1"] = phase1_winner

                print(f"Phase 1 winner (hard max avg speed_pp): "
                      f"batch={phase1_winner['--batch-size']}, "
                      f"ubatch={phase1_winner['--ubatch-size']} "
                      f"({phase1_winner['speed_pp']:.1f} t/s)")

        # PHASE 2: speculative decoding scan
        if args.phase is None or args.phase == 2:
            print("\n=== Phase 2: scan speculative decoding ===")

            # Determine the fixed parameters for phase 2 based on the phase 1 winner
            if phase1_winner is not None:
                fixed = {k: v for k, v in phase1_winner.items() if k.startswith("--")}
                fixed_src = "auto (phase 1)"

            else:
                fixed = overrides.get("phase1", {})
                fixed_src = "override"

            # Launch parameters for phase 2, incorporating fixed parameters from phase 1 winner or overrides
            p2 = phase_launch_params(shared, phases["2"], fixed)
            variant_rows = {}

            # Loop on speculative decoding conditions and run each one, collecting results in variant_rows
            for variant in spec_variants:
                r, launch_counter = run_phase(
                    2, "phase 2", models_path, config, args.runs, launch_counter,
                    f, variant["label"], p2, variant.get("flags", {}),
                    f"spec={variant['label']} @ batch={p2.get('--batch-size')}, "
                    f"ubatch={p2.get('--ubatch-size')} (from {fixed_src})"
                )

                variant_rows[variant["label"]] = r

            # Print the ranking of speculative decoding variants based on average speed_tg
            print("\nPhase 2 ranking (avg speed_tg):")

            for v in spec_variants:
                vals = [
                    r["speed_tg"] for _, r in variant_rows[v["label"]]
                        if isinstance(r.get("speed_tg"), (int, float))
                ]

                s = sum(vals) / len(vals) if vals else float("nan")
                print(f"  {v['label']:<14}  speed_tg={s:8.2f} t/s")

            # If only phase 2 is requested, stop here and provide guidance for phase 3.
            if args.phase == 2:
                print("\n(Stopped: --phase 2. Pin the winner in overrides.phase2 to run phase 3.)")

            # Otherwise, pick the phase 2 winner and store it in the summary for chaining to phase 3.
            else:
                winner = pick_phase2_winner(variant_rows, spec_variants)

                if winner is None or winner["speed_tg"] != winner["speed_tg"]:
                    parser.error("phase 2 produced no speed_tg data; cannot chain")

                summary_data["phases"]["2"] = {
                    "variant": winner["variant"]["label"],
                    "batch": p2.get("--batch-size"),
                    "ubatch": p2.get("--ubatch-size"),
                    "speed_tg": winner["speed_tg"],
                    "ranking": winner["ranking"]
                }

                print(
                    f"Phase 2 winner (max avg speed_tg): {winner['variant']['label']} "
                      f"({winner['speed_tg']:.2f} t/s)"
                )

        # PHASE 3: performance characterization vs occupancy
        if args.phase is None or args.phase == 3:
            print("\n=== Phase 3: characterize PP and generation vs context occupancy ===")

            # batch/ubatch: from this chain's phase 1, else override.
            if phase1_winner is not None:
                fixed = {k: v for k, v in phase1_winner.items() if k.startswith("--")}

            else:
                fixed = overrides.get("phase1", {})

            # Speculative decoding: from this chain's phase 2, else a pinned label
            # (overrides.phase2 = {"label": "<spec_variants label>"}), else raw
            # flags (overrides.phase3_spec), else off.
            if "2" in summary_data["phases"]:
                winner_label = summary_data["phases"]["2"]["variant"]
                spec_v = next(v for v in spec_variants if v["label"] == winner_label)

            elif overrides.get("phase2", {}).get("label"):
                wanted = overrides["phase2"]["label"]
                spec_v = next((v for v in spec_variants if v["label"] == wanted), None)

                if spec_v is None:
                    parser.error(f"overrides.phase2.label {wanted!r} is not in spec_variants")

            elif overrides.get("phase3_spec"):
                spec_v = {"label": "override", "flags": overrides["phase3_spec"]}

            else:
                spec_v = {"label": "spec-off", "flags": {}}

            p3 = phase_launch_params(shared, phases["3"], fixed)

            rows, launch_counter = run_phase(
                3, "phase 3", models_path, config, args.runs, launch_counter,
                f, spec_v["label"], p3, spec_v.get("flags", {}),
                f"batch={p3.get('--batch-size')}, ubatch={p3.get('--ubatch-size')}, "
                f"spec={spec_v['label']}; occupancy sweep {p3.get('-npp')}"
            )

            summary_data["phases"]["3"] = {
                "batch": p3.get("--batch-size"),
                "ubatch": p3.get("--ubatch-size"),
                "spec": spec_v["label"],
                "occupancy_sweep": p3.get("-npp"),
                "occupancy_curve": sorted({r["pp"] for _, r in rows
                                           if isinstance(r.get("pp"), int)})
            }

    # Speculative decoding accept-rate estimate (from phase 2)
    # speed_tg with spec = target_tokens/s, where each target step accepts
    # A draft tokens on average and costs 1 draft + 1 verify forward. The
    # tool does not expose A or the verify cost; A+1 = variant/control is an
    # UPPER bound on accepted tokens per target token (it assumes verify is
    # free), so the printed value is an over-estimate of true acceptance.
    if "2" in summary_data["phases"]:
        ranking = summary_data["phases"]["2"]["ranking"]

        control = next((s for lab, s in ranking
                        if lab == "spec-off" and s == s), None)

        summary_data["phases"]["2"]["accept_estimate"] = {}

        print(
            "\nSpec accept-rate estimate (upper bound, assumes verify cost is free; "
            "A+1 = speed_tg(variant) / speed_tg(control), A = implied accepted "
            "draft tokens per target step):"
        )

        for lab, s in ranking:
            if lab == "spec-off" or control in (None, 0) or s != s:
                continue

            ratio = s / control

            print(
                f"  {lab:<14}  {s:8.2f} t/s  =>  {ratio:.2f} = 1 + A, "
                f"A < {max(ratio - 1, 0):.2f}"
            )

            summary_data["phases"]["2"]["accept_estimate"][lab] = round(ratio, 3)

    # Write summary to disk
    summary_file = result_file.with_suffix(".summary.json")

    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nWrote results to {result_file}")
    print(f"Wrote summary to {summary_file}")

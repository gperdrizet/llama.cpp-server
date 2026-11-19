'''Canonical paths for the benchmark tools, independent of how they are run.

Config and prompt paths in the shipped JSON files are relative to the
repository root (the directory the tools are invoked from). Deriving them from
this file's location keeps results in ``tests/results`` whether a tool is run
as a script or imported as a module, from any working directory.
'''

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]   # .../llama.cpp
TESTS_DIR = REPO_ROOT / "tests"
RESULTS_DIR = TESTS_DIR / "results"

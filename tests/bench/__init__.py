'''Benchmark tooling package.

The CLI modules here are runnable either as scripts
(``python tests/bench/llama_context_tuner.py``) or as imports
(``from bench import llama_context_tuner``). ``llama_server_lib`` holds the
shared Docker launch/measurement engine; ``paths`` holds the canonical
repository paths so results always land in the same place either way.

Configs and prompts live in ``tests/config`` / ``tests/prompts``; outputs go
to ``tests/results``. See ``tests/README.md``.
'''

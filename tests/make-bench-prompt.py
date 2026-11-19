'''Build a long, realistic benchmark prompt from Claude Code transcripts.

The llama-server prompt-processing decay curve (measured by the tools in
tests/ that share llama_server_lib.py) needs an input prompt that fills the
context (>= ~131k tokens for a 131072 window). This builds one from real
conversation transcripts so the token
distribution (prose + code + agentic chatter) is realistic rather than
synthetic filler, and writes it to a single swappable file.

Usage:
  python tests/make-bench-prompt.py --out tests/prompts/qwen3_8_27b_v1.txt \
      --target 122000 --src ~/.claude/projects/-home-siderealyear-llama-cpp

The on-disk token count is an ESTIMATE (a chars-per-token ratio); the
llama-server run reports the true `n_tokens`, which is the ground truth for
how full the context got. Regenerate with a different --target / --src /
--ratio to try variations, or point the config at a hand-written file.

Keep --target comfortably UNDER the sweep's --ctx-size: the request is
rejected if prompt tokens + generated tokens (ntg / spec_ntg) exceed the
context. And note the ratio is content-dependent — the same builder pointed
at different transcript regions can tokenize denser or sparser, so check the
est tokens (or a real run's n_tokens) before trusting the size.
'''

import argparse
import json
import os
from pathlib import Path


def _tool_result_texts(content):
    '''Yield the readable text inside a tool_result's content.

    A tool_result's `content` is either a plain string or a list of blocks;
    take the `text` of each block (image blocks carry no `text` and are
    skipped). Returns a list of strings (possibly empty).
    '''

    if isinstance(content, str):
        return [content]

    if not isinstance(content, list):
        return []

    out = []
    for block in content:
        if isinstance(block, dict):
            text = block.get("text", "")
            if text:
                out.append(text)
    return out


def iter_transcript_texts(path: Path):
    '''Yield the realistic readable content of each transcript.

    Includes text blocks, tool-call arguments (`tool_use.input`, serialized as
    JSON), and tool-result payloads (the file reads / command outputs that make
    up the bulk of an agentic session's tokens). Skips meta/sidechain/caveat
    lines and image blocks. The token mix (prose + code + tool I/O) is the real
    distribution we want the decay curve measured against.
    '''

    out = []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)

            except json.JSONDecodeError:
                continue

            if obj.get("type") not in ("user", "assistant"):
                continue
            if obj.get("isMeta") or obj.get("isSidechain"):
                continue

            msg = obj.get("message")

            if not isinstance(msg, dict):
                continue

            content = msg.get("content")

            if isinstance(content, str):
                if content.strip():
                    out.append(content)
                continue

            if not isinstance(content, list):
                continue

            for block in content:
                if not isinstance(block, dict):
                    continue

                btype = block.get("type")

                if btype == "text":
                    text = block.get("text", "")
                    if text.strip():
                        out.append(text)

                elif btype == "tool_use":
                    # the tool-call arguments are realistic agentic content
                    args = block.get("input")
                    if args is not None:
                        out.append(json.dumps(args))

                elif btype == "tool_result":
                    for text in _tool_result_texts(block.get("content")):
                        if text.strip():
                            out.append(text)

    return out


def build_prompt(src: Path, target_tokens: int, ratio: float,
                 offset_chars: int = 0) -> tuple[str, int, int]:
    '''Concatenate transcript text until ~target_tokens tokens are collected.

    Sources are the *.jsonl transcripts in `src`, largest first (the big ones
    are this project's long sessions and are the most representative).
    `offset_chars` skips that many chars off the START of the combined stream,
    so several runs of this script (--offset 0 / N / 2N) yield several
    DISTINCT prompt versions over the same source material.
    Returns (prompt_text, source_count, char_count).
    '''

    sources = sorted(src.glob("*.jsonl"), key=lambda p: p.stat().st_size, reverse=True)

    if not sources:
        raise FileNotFoundError(f"no *.jsonl transcripts under {src}")

    chunks = []
    nchars = 0
    skipped = 0
    want_chars = int(target_tokens * ratio)

    for path in sources:
        for text in iter_transcript_texts(path):
            if not text:
                continue
            # skip the first offset_chars of the combined stream
            if skipped < offset_chars:
                take_from = offset_chars - skipped
                if len(text) <= take_from:
                    skipped += len(text)
                    continue
                text = text[take_from:]
                skipped = offset_chars

            # Trim the final chunk to land on the char target. A single
            # tool_result can be tens of thousands of chars; appending it whole
            # would overshoot want_chars and could push the prompt past the
            # server's --ctx-size (request rejected). Trimming is fine — the
            # prompt is filler, not a document.
            remaining = want_chars - nchars
            if len(text) > remaining:
                chunks.append(text[:remaining])
                nchars = want_chars
                break
            chunks.append(text)
            nchars += len(text)

        if nchars >= want_chars:
            break

    return "\n\n".join(chunks), len(sources), sum(len(c) for c in chunks)


def main():
    parser = argparse.ArgumentParser(description="Build a long realistic benchmark prompt from transcripts")
    parser.add_argument("--out", required=True, type=Path,
                        help="where to write the prompt text file")
    parser.add_argument("--src", type=Path,
                        default=Path.home() / ".claude/projects/-home-siderealyear-llama-cpp",
                        help="directory of *.jsonl transcripts (default: this project's Claude transcripts)")
    parser.add_argument("--target", type=int, default=122000,
                        help="target token count of the prompt (estimated; the server reports the true "
                             "n_tokens). Keep it under the sweep's --ctx-size with room for the ntg/spec_ntg "
                             "generation tokens, e.g. 122000 for a 131072 window")
    parser.add_argument("--ratio", type=float, default=3.07,
                        help="chars-per-token estimate used to size the prompt. 3.07 is the value this "
                             "Qwen model measured on the current prompt content (471113 chars -> "
                             "153533 tokens); lower = fewer chars per token = more tokens for the same "
                             "char budget. The ratio is content-dependent, so verify the est against a "
                             "real run's n_tokens if you point --src at different transcripts")
    parser.add_argument("--offset", type=int, default=0,
                        help="chars to skip off the start of the combined transcript stream; use "
                             "distinct offsets to build distinct prompt versions over the same sources")
    args = parser.parse_args()

    text, nsrc, nchars = build_prompt(args.src, args.target, args.ratio, args.offset)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")

    print(f"Wrote {args.out}")
    print(f"  chars        : {nchars}")
    print(f"  est tokens   : {nchars / args.ratio:.0f}  (ratio {args.ratio})  [target {args.target}]")
    print(f"  sources      : {nsrc} transcript(s) from {args.src}")
    print("  (the llama-server run reports the true n_tokens; regenerate with a different "
          "--target/--ratio or point the config at this file to try variations)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Benchmark Qwen3.5-35B-A3B-4bit on omlx — measures prefill and generation TPS."""

import json
import time
import urllib.request
import statistics

BASE_URL = "http://localhost:8000"
MODEL = "Qwen3.5-35B-A3B-4bit"

TESTS = [
    {
        "name": "Short prompt, short output",
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 100,
        "extra": {"chat_template_kwargs": {"enable_thinking": False}},
    },
    {
        "name": "Short prompt, long output",
        "messages": [{"role": "user", "content": "Write a short poem about the moon."}],
        "max_tokens": 300,
        "extra": {"chat_template_kwargs": {"enable_thinking": False}},
    },
    {
        "name": "Long prompt, medium output",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Here is a long context: " + "The quick brown fox jumps over the lazy dog. " * 50
                    + "\n\nSummarize the above in one sentence."
                ),
            }
        ],
        "max_tokens": 100,
        "extra": {"chat_template_kwargs": {"enable_thinking": False}},
    },
    {
        "name": "Coding task",
        "messages": [
            {
                "role": "user",
                "content": "Write a Python function that computes the nth Fibonacci number using memoization.",
            }
        ],
        "max_tokens": 400,
        "extra": {"chat_template_kwargs": {"enable_thinking": False}},
    },
]


def stream_request(messages, max_tokens, extra=None):
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if extra:
        payload.update(extra)

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    tokens = []
    usage = None
    wall_start = time.perf_counter()

    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw_line in resp:
            line = raw_line.decode().strip()
            if not line or line == "data: [DONE]" or line.startswith(": "):
                continue
            if line.startswith("data: "):
                chunk = json.loads(line[6:])
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    text = delta.get("content") or delta.get("reasoning_content") or ""
                    if text:
                        tokens.append(text)

    wall_elapsed = time.perf_counter() - wall_start
    return "".join(tokens), usage, wall_elapsed


def run_benchmark():
    print(f"\n{'='*65}")
    print(f"  oMLX Benchmark — {MODEL}")
    print(f"  Server: {BASE_URL}")
    print(f"{'='*65}\n")

    all_pp_tps = []
    all_tg_tps = []

    for i, test in enumerate(TESTS, 1):
        print(f"[{i}/{len(TESTS)}] {test['name']}")
        print(f"       max_tokens={test['max_tokens']}", flush=True)

        try:
            output, usage, wall = stream_request(
                test["messages"], test["max_tokens"], test.get("extra")
            )
        except Exception as e:
            print(f"       ERROR: {e}\n")
            continue

        if not usage:
            print("       No usage stats returned.\n")
            continue

        prompt_tps = usage.get("prompt_tokens_per_second")
        gen_tps = usage.get("generation_tokens_per_second")
        prompt_tok = usage.get("prompt_tokens", "?")
        gen_tok = usage.get("completion_tokens", "?")
        ttft = usage.get("time_to_first_token")
        total_time = usage.get("total_time", wall)

        print(f"       Prompt tokens   : {prompt_tok}")
        print(f"       Output tokens   : {gen_tok}")
        print(f"       Time to 1st tok : {ttft:.3f}s" if ttft else "       Time to 1st tok : n/a")
        print(f"       Total time      : {total_time:.2f}s")
        print(f"       Prefill TPS     : {prompt_tps:.1f} tok/s" if prompt_tps else "       Prefill TPS     : n/a")
        print(f"       Generation TPS  : {gen_tps:.1f} tok/s" if gen_tps else "       Generation TPS  : n/a")
        print(f"       Output snippet  : {output[:80].strip()!r}{'...' if len(output) > 80 else ''}")
        print()

        if prompt_tps:
            all_pp_tps.append(prompt_tps)
        if gen_tps:
            all_tg_tps.append(gen_tps)

    print(f"{'='*65}")
    print("  SUMMARY")
    print(f"{'='*65}")
    if all_pp_tps:
        print(f"  Prefill  — avg: {statistics.mean(all_pp_tps):.1f}  "
              f"min: {min(all_pp_tps):.1f}  max: {max(all_pp_tps):.1f}  tok/s")
    if all_tg_tps:
        print(f"  Generate — avg: {statistics.mean(all_tg_tps):.1f}  "
              f"min: {min(all_tg_tps):.1f}  max: {max(all_tg_tps):.1f}  tok/s")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    run_benchmark()

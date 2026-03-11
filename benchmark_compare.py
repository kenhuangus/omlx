#!/usr/bin/env python3
"""Side-by-side benchmark: Qwen3-32B-4bit vs Qwen3.5-35B-A3B-4bit"""

import json
import time
import urllib.request
import statistics

BASE_URL = "http://localhost:8000"
API_KEY = "Webtest@123"

MODELS = [
    "Qwen3-32B-4bit",
    "Qwen3.5-35B-A3B-4bit",
]

TESTS = [
    {
        "name": "Short prompt",
        "messages": [{"role": "user", "content": "Say hello in one sentence."}],
        "max_tokens": 80,
    },
    {
        "name": "Long prompt (500 tok prefill)",
        "messages": [{"role": "user", "content": "Summarize in one sentence: " + "The quick brown fox jumps over the lazy dog. " * 50}],
        "max_tokens": 60,
    },
    {
        "name": "Coding task",
        "messages": [{"role": "user", "content": "Write a Python binary search function with docstring."}],
        "max_tokens": 350,
    },
    {
        "name": "Tool calling",
        "messages": [{"role": "user", "content": "What files are in the current directory?"}],
        "max_tokens": 150,
        "tools": [{"type": "function", "function": {
            "name": "execute_command",
            "description": "Run a shell command",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
        }}],
    },
    {
        "name": "Long generation (400 tok)",
        "messages": [{"role": "user", "content": "Explain how transformers work in detail."}],
        "max_tokens": 400,
    },
]

EXTRA = {"chat_template_kwargs": {"enable_thinking": False}}


def stream_request(model, messages, max_tokens, tools=None):
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        **EXTRA,
    }
    if tools:
        payload["tools"] = tools

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
    )

    usage = None
    output_tokens = 0
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw_line in resp:
            line = raw_line.decode().strip()
            if not line or line == "data: [DONE]" or line.startswith(": "):
                continue
            if line.startswith("data: "):
                chunk = json.loads(line[6:])
                if chunk.get("usage"):
                    usage = chunk["usage"]

    return usage


def run():
    print(f"\n{'='*70}")
    print(f"  MLX Model Benchmark Comparison")
    print(f"  Server: {BASE_URL}")
    print(f"{'='*70}")
    print(f"  {'Test':<28} {'Model':<28} {'PP tok/s':>9} {'TG tok/s':>9} {'TTFT':>6}")
    print(f"  {'-'*28} {'-'*28} {'-'*9} {'-'*9} {'-'*6}")

    results = {m: {"pp": [], "tg": []} for m in MODELS}

    for test in TESTS:
        print(f"\n  [{test['name']}]")
        for model in MODELS:
            usage = stream_request(model, test["messages"], test["max_tokens"], test.get("tools"))
            if not usage:
                print(f"  {'':28} {model:<28} {'ERROR':>9}")
                continue

            pp = usage.get("prompt_tokens_per_second")
            tg = usage.get("generation_tokens_per_second")
            ttft = usage.get("time_to_first_token")
            gen_tok = usage.get("completion_tokens", 0)

            if pp: results[model]["pp"].append(pp)
            if tg: results[model]["tg"].append(tg)

            pp_str = f"{pp:.1f}" if pp else "n/a"
            tg_str = f"{tg:.1f}" if tg else "n/a"
            ttft_str = f"{ttft:.2f}s" if ttft else "n/a"
            print(f"  {'':28} {model:<28} {pp_str:>9} {tg_str:>9} {ttft_str:>6}  ({gen_tok} tok)")

    print(f"\n{'='*70}")
    print(f"  SUMMARY (averages across all tests)")
    print(f"{'='*70}")
    print(f"  {'Model':<32} {'Prefill avg':>12} {'Generate avg':>13}")
    print(f"  {'-'*32} {'-'*12} {'-'*13}")
    for model in MODELS:
        pp_vals = results[model]["pp"]
        tg_vals = results[model]["tg"]
        pp_avg = f"{statistics.mean(pp_vals):.1f} tok/s" if pp_vals else "n/a"
        tg_avg = f"{statistics.mean(tg_vals):.1f} tok/s" if tg_vals else "n/a"
        print(f"  {model:<32} {pp_avg:>12} {tg_avg:>13}")

    # Winner
    tg_scores = {m: statistics.mean(results[m]["tg"]) for m in MODELS if results[m]["tg"]}
    if len(tg_scores) == 2:
        winner = max(tg_scores, key=tg_scores.get)
        loser = min(tg_scores, key=tg_scores.get)
        pct = (tg_scores[winner] - tg_scores[loser]) / tg_scores[loser] * 100
        print(f"\n  Generation speed winner: {winner} (+{pct:.1f}% faster)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    run()

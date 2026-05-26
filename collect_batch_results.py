"""
collect_batch_results.py — fetch results from already-submitted batch jobs and
append them to results.jsonl in the standard record format.

Usage:
    python collect_batch_results.py --claude msgbatch_abc123 --gpt batch_abc123 --together batch_abc123

Pass only the flags for jobs you actually submitted; others are skipped.
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

env_path = Path(".env.local")
if env_path.exists():
    load_dotenv(env_path)

OUTPUT_FILE      = "results_comp.jsonl"
ERROR_FILE       = "errors.log"
POLL_INTERVAL_SEC = 30


# ── helpers ──────────────────────────────────────────────────────────────────

def load_prompts(path="prompts.json"):
    with open(path) as f:
        return json.load(f)["prompts"]

def write_result(record):
    with open(OUTPUT_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")

def log_error(model, prompt_id, error):
    with open(ERROR_FILE, "a") as f:
        f.write(f"{datetime.utcnow().isoformat()} | {model} | {prompt_id} | {error}\n")


# ── Claude ────────────────────────────────────────────────────────────────────

def collect_claude(batch_id, prompts):
    from anthropic import Anthropic
    print(f"\n[Claude] Collecting batch {batch_id} ...")
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(f"  Status: {batch.processing_status} | "
              f"processing={counts.processing} succeeded={counts.succeeded} errored={counts.errored}")
        if batch.processing_status == "ended":
            break
        time.sleep(POLL_INTERVAL_SEC)

    prompt_map = {p["id"]: p for p in prompts}
    results = []

    for result in client.messages.batches.results(batch_id):
        p = prompt_map.get(result.custom_id)
        if not p:
            print(f"  ⚠ unknown custom_id: {result.custom_id}")
            continue

        if result.result.type == "succeeded":
            text = result.result.message.content[0].text
            record = {
                "timestamp":       datetime.utcnow().isoformat(),
                "model":           "claude-opus-4-7",
                "prompt_id":       p["id"],
                "question_id":     p["question_id"],
                "variant":         p["variant"],
                "run":             p["run"],
                "status":          "success",
                "response":        text,
                "response_length": len(text),
            }
        else:
            record = {
                "timestamp":   datetime.utcnow().isoformat(),
                "model":       "claude-opus-4-7",
                "prompt_id":   p["id"],
                "question_id": p["question_id"],
                "variant":     p["variant"],
                "run":         p["run"],
                "status":      "error",
                "error":       str(result.result.error),
            }
            log_error("claude-opus-4-7", p["id"], record["error"])

        results.append(record)
        write_result(record)

    print(f"  [Claude] {sum(1 for r in results if r['status']=='success')} succeeded, "
          f"{sum(1 for r in results if r['status']=='error')} failed.")
    return results


# ── GPT-4o ────────────────────────────────────────────────────────────────────

def collect_gpt(batch_id, prompts):
    from openai import OpenAI
    print(f"\n[GPT-4o] Collecting batch {batch_id} ...")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    while True:
        batch = client.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(f"  Status: {batch.status} | "
              f"completed={counts.completed} failed={counts.failed} total={counts.total}")
        if batch.status in ("completed", "failed", "expired", "cancelled"):
            break
        time.sleep(POLL_INTERVAL_SEC)

    if batch.status != "completed":
        print(f"  [GPT-4o] Batch ended with status: {batch.status}")
        return []

    result_content = client.files.content(batch.output_file_id).text
    prompt_map = {p["id"]: p for p in prompts}
    results = []

    for line in result_content.strip().split("\n"):
        if not line.strip():
            continue
        item = json.loads(line)
        p = prompt_map.get(item["custom_id"])
        if not p:
            print(f"  ⚠ unknown custom_id: {item['custom_id']}")
            continue

        if item["response"]["status_code"] == 200:
            text = item["response"]["body"]["choices"][0]["message"]["content"]
            record = {
                "timestamp":       datetime.utcnow().isoformat(),
                "model":           "gpt-4o",
                "prompt_id":       p["id"],
                "question_id":     p["question_id"],
                "variant":         p["variant"],
                "run":             p["run"],
                "status":          "success",
                "response":        text,
                "response_length": len(text),
            }
        else:
            error_msg = str(item["response"]["body"])
            record = {
                "timestamp":   datetime.utcnow().isoformat(),
                "model":       "gpt-4o",
                "prompt_id":   p["id"],
                "question_id": p["question_id"],
                "variant":     p["variant"],
                "run":         p["run"],
                "status":      "error",
                "error":       error_msg,
            }
            log_error("gpt-4o", p["id"], error_msg)

        results.append(record)
        write_result(record)

    print(f"  [GPT-4o] {sum(1 for r in results if r['status']=='success')} succeeded, "
          f"{sum(1 for r in results if r['status']=='error')} failed.")
    return results


# ── Together/Llama ────────────────────────────────────────────────────────────

def collect_together(batch_id, prompts):
    from together import Together
    print(f"\n[Llama/Together] Collecting batch {batch_id} ...")
    client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

    while True:
        batch = client.batches.retrieve(batch_id)
        print(f"  Status: {batch.status} | Progress: {batch.progress}%")
        if batch.status in ("COMPLETED", "FAILED", "EXPIRED", "CANCELLED"):
            break
        time.sleep(POLL_INTERVAL_SEC)

    if batch.status != "COMPLETED":
        print(f"  [Llama] Batch ended with status: {batch.status}")
        return []

    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        out_path = f.name
    raw = client.files.content(batch.output_file_id)
    with open(out_path, "wb") as f:
        f.write(raw.read())

    prompt_map = {p["id"]: p for p in prompts}
    results = []
    with open(out_path) as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            custom_id = item.get("custom_id")
            p = prompt_map.get(custom_id)
            if not p:
                print(f"  ⚠ unknown custom_id: {custom_id}")
                continue
            status_code = item.get("response", {}).get("status_code")
            try:
                if status_code != 200:
                    raise ValueError(str(item.get("response", {}).get("body")))
                text = item["response"]["body"]["choices"][0]["message"]["content"]
                record = {
                    "timestamp":       datetime.utcnow().isoformat(),
                    "model":           "llama-3.3",
                    "prompt_id":       p["id"],
                    "question_id":     p["question_id"],
                    "variant":         p["variant"],
                    "run":             p["run"],
                    "status":          "success",
                    "response":        text,
                    "response_length": len(text),
                }
            except Exception as e:
                record = {
                    "timestamp":   datetime.utcnow().isoformat(),
                    "model":       "llama-3.3",
                    "prompt_id":   p["id"],
                    "question_id": p["question_id"],
                    "variant":     p["variant"],
                    "run":         p["run"],
                    "status":      "error",
                    "error":       str(e),
                }
                log_error("llama-3.3", p["id"], str(e))

            results.append(record)
            write_result(record)

    os.unlink(out_path)
    print(f"  [Llama] {sum(1 for r in results if r['status']=='success')} succeeded, "
          f"{sum(1 for r in results if r['status']=='error')} failed.")
    return results


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    global OUTPUT_FILE, ERROR_FILE

    parser = argparse.ArgumentParser(description="Collect completed batch results")
    parser.add_argument("--claude",   metavar="BATCH_ID", help="Anthropic batch ID (msgbatch_...)")
    parser.add_argument("--gpt",      metavar="BATCH_ID", help="OpenAI batch ID (batch_...)")
    parser.add_argument("--together", metavar="BATCH_ID", help="Together AI batch ID")
    parser.add_argument("--prompts",  default="prompts.json", help="Path to prompts.json")
    parser.add_argument("--out",      default=OUTPUT_FILE,    help="Output JSONL file (appended)")
    args = parser.parse_args()

    OUTPUT_FILE = args.out
    ERROR_FILE  = args.out.replace(".jsonl", "_errors.log")

    if not any([args.claude, args.gpt, args.together]):
        parser.error("Provide at least one of --claude, --gpt, --together")

    prompts = load_prompts(args.prompts)
    print(f"Loaded {len(prompts)} prompts from {args.prompts}")

    all_results = []
    if args.claude:
        all_results += collect_claude(args.claude, prompts)
    if args.gpt:
        all_results += collect_gpt(args.gpt, prompts)
    if args.together:
        all_results += collect_together(args.together, prompts)

    print(f"\n{'='*60}")
    print(f"TOTAL: {len(all_results)} records — "
          f"{sum(1 for r in all_results if r['status']=='success')} succeeded, "
          f"{sum(1 for r in all_results if r['status']=='error')} failed")
    print(f"Appended to: {OUTPUT_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

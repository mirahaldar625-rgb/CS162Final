import json
import os
import time
import tempfile
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from anthropic import Anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request as AnthropicBatchRequest
from google import genai as google_genai
from google.genai import types as google_types
from together import Together
from openai import OpenAI

# ============================================================================
# LOAD ENV
# ============================================================================

env_path = Path(".env.local")
if env_path.exists():
    load_dotenv(env_path)
    print(f"✓ Loaded environment from {env_path}")
else:
    print(f"⚠ Warning: {env_path} not found, falling back to system env vars")

# ============================================================================
# CONFIG
# ============================================================================

MODELS = [
    #"claude-opus-4-7",
    "gpt-4o",
    "gemini-2.5-flash",
    "llama-3.3",
]

CLAUDE_MODEL_STRING  = "claude-opus-4-7"
GPT_MODEL_STRING     = "gpt-4o"
GEMINI_MODEL_STRING  = "gemini-2.5-flash"
LLAMA_MODEL_STRING   = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

MAX_TOKENS  = 1024
TEMPERATURE = 1.0
OUTPUT_FILE = "results.jsonl"
ERROR_FILE  = "errors.log"
POLL_INTERVAL_SEC = 30    # how often to check batch status

# ============================================================================
# LOAD PROMPTS
# ============================================================================

def load_prompts(path="prompts.json"):
    with open(path, "r") as f:
        return json.load(f)["prompts"]

# ============================================================================
# BATCH: CLAUDE
# Docs: https://docs.anthropic.com/en/docs/build-with-claude/batch-processing
# Submits all prompts in one batch, polls until complete, streams results
# Cost: 50% off standard rate
# ============================================================================

def run_claude_batch(prompts):
    print("\n[Claude] Submitting batch...")
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Build batch requests — one per prompt
    requests = [
        AnthropicBatchRequest(
            custom_id=p["id"],
            params=MessageCreateParamsNonStreaming(
                model=CLAUDE_MODEL_STRING,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                messages=[{"role": "user", "content": p["text"]}]
            )
        )
        for p in prompts
    ]

    batch = client.messages.batches.create(requests=requests)
    print(f"[Claude] Batch submitted: {batch.id} ({len(requests)} requests)")

    # Poll until complete
    while True:
        batch = client.messages.batches.retrieve(batch.id)
        counts = batch.request_counts
        print(f"[Claude] Status: {batch.processing_status} | "
              f"processing={counts.processing} errored={counts.errored} succeeded={counts.succeeded}")
        if batch.processing_status == "ended":
            break
        time.sleep(POLL_INTERVAL_SEC)

    # Build lookup: custom_id -> prompt metadata
    prompt_map = {p["id"]: p for p in prompts}

    # Stream results
    results = []
    for result in client.messages.batches.results(batch.id):
        p = prompt_map[result.custom_id]
        if result.result.type == "succeeded":
            record = {
                "timestamp":       datetime.utcnow().isoformat(),
                "model":           "claude-opus-4-7",
                "prompt_id":       p["id"],
                "question_id":     p["question_id"],
                "variant":         p["variant"],
                "run":             p["run"],
                "status":          "success",
                "response":        result.result.message.content[0].text,
                "response_length": len(result.result.message.content[0].text),
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

    print(f"[Claude] Done. {sum(1 for r in results if r['status']=='success')} succeeded, "
          f"{sum(1 for r in results if r['status']=='error')} failed.")
    return results


# ============================================================================
# BATCH: GPT-4o (OpenAI)
# Docs: https://platform.openai.com/docs/guides/batch
# Workflow: write JSONL -> upload file -> create batch -> poll -> download results
# Cost: 50% off standard rate
# ============================================================================

def run_gpt_batch(prompts):
    print("\n[GPT-4o] Submitting batch...")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Write JSONL batch input file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for p in prompts:
            line = {
                "custom_id": p["id"],
                "method":    "POST",
                "url":       "/v1/chat/completions",
                "body": {
                    "model":       GPT_MODEL_STRING,
                    "messages":    [{"role": "user", "content": p["text"]}],
                    "max_tokens":  MAX_TOKENS,
                    "temperature": TEMPERATURE,
                }
            }
            f.write(json.dumps(line) + "\n")
        tmp_path = f.name

    # Upload file
    with open(tmp_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    print(f"[GPT-4o] File uploaded: {uploaded.id}")
    os.unlink(tmp_path)

    # Create batch
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    print(f"[GPT-4o] Batch created: {batch.id}")

    # Poll until complete
    while True:
        batch = client.batches.retrieve(batch.id)
        counts = batch.request_counts
        print(f"[GPT-4o] Status: {batch.status} | "
              f"completed={counts.completed} failed={counts.failed} total={counts.total}")
        if batch.status in ("completed", "failed", "expired", "cancelled"):
            break
        time.sleep(POLL_INTERVAL_SEC)

    if batch.status != "completed":
        print(f"[GPT-4o] Batch ended with status: {batch.status}")
        return []

    # Download and parse results
    result_content = client.files.content(batch.output_file_id).text
    prompt_map = {p["id"]: p for p in prompts}

    results = []
    for line in result_content.strip().split("\n"):
        item = json.loads(line)
        p = prompt_map[item["custom_id"]]
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

    print(f"[GPT-4o] Done. {sum(1 for r in results if r['status']=='success')} succeeded, "
          f"{sum(1 for r in results if r['status']=='error')} failed.")
    return results


# ============================================================================
# SEQUENTIAL: Gemini 2.5 Flash
# Note: batchGenerateContent requires a Google Cloud project with billing;
# AI Studio API keys get FAILED_PRECONDITION. Using sequential calls instead.
# Resume-safe: skips prompt IDs already recorded as successful in OUTPUT_FILE.
# Retry logic: backs off on 429/503 and retries up to MAX_RETRIES times.
# ============================================================================

GEMINI_RPM          = 10   # free-tier limit for gemini-2.5-flash
GEMINI_CALL_SLEEP   = 60 / GEMINI_RPM   # 6 s between calls
GEMINI_MAX_RETRIES  = 4
GEMINI_RETRY_DELAYS = [15, 30, 60, 120]  # seconds per retry attempt


def _load_completed_gemini_ids():
    """Return set of prompt_ids already successfully written to OUTPUT_FILE."""
    if not os.path.exists(OUTPUT_FILE):
        return set()
    completed = set()
    with open(OUTPUT_FILE) as f:
        for line in f:
            try:
                r = json.loads(line)
                if r.get("model") == "gemini-2.5-flash" and r.get("status") == "success":
                    completed.add(r["prompt_id"])
            except json.JSONDecodeError:
                pass
    return completed


def run_gemini_batch(prompts):
    print("\n[Gemini] Running sequentially (batch API not available on AI Studio keys)...")
    client = google_genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    config = google_types.GenerateContentConfig(
        temperature=TEMPERATURE,
        max_output_tokens=MAX_TOKENS,
    )

    completed = _load_completed_gemini_ids()
    if completed:
        print(f"  Resuming — skipping {len(completed)} already-successful prompts.")

    todo = [p for p in prompts if p["id"] not in completed]
    print(f"  {len(todo)} prompts remaining.")

    results = []

    for p in todo:
        print(f"  {p['id']}: ", end="", flush=True)

        record = None
        for attempt in range(GEMINI_MAX_RETRIES + 1):
            start = time.time()
            try:
                response = client.models.generate_content(
                    model=GEMINI_MODEL_STRING,
                    contents=p["text"],
                    config=config,
                )
                text = response.text or ""
                elapsed = time.time() - start
                print(f"✓ {elapsed:.1f}s | {len(text)} chars"
                      + (f" (attempt {attempt+1})" if attempt else ""))
                record = {
                    "timestamp":       datetime.utcnow().isoformat(),
                    "model":           "gemini-2.5-flash",
                    "prompt_id":       p["id"],
                    "question_id":     p["question_id"],
                    "variant":         p["variant"],
                    "run":             p["run"],
                    "status":          "success",
                    "response":        text,
                    "response_length": len(text),
                }
                break  # success — stop retrying

            except Exception as e:
                elapsed = time.time() - start
                err_str = str(e)
                is_retryable = "429" in err_str or "503" in err_str or "UNAVAILABLE" in err_str or "RESOURCE_EXHAUSTED" in err_str

                if is_retryable and attempt < GEMINI_MAX_RETRIES:
                    delay = GEMINI_RETRY_DELAYS[attempt]
                    print(f"↺ attempt {attempt+1} failed ({err_str[:40]}...) — waiting {delay}s")
                    time.sleep(delay)
                    continue

                print(f"✗ {err_str[:70]}")
                record = {
                    "timestamp":   datetime.utcnow().isoformat(),
                    "model":       "gemini-2.5-flash",
                    "prompt_id":   p["id"],
                    "question_id": p["question_id"],
                    "variant":     p["variant"],
                    "run":         p["run"],
                    "status":      "error",
                    "error":       err_str,
                }
                log_error("gemini-2.5-flash", p["id"], err_str)
                break

        results.append(record)
        write_result(record)
        time.sleep(GEMINI_CALL_SLEEP)

    print(f"[Gemini] Done. {sum(1 for r in results if r['status']=='success')} succeeded, "
          f"{sum(1 for r in results if r['status']=='error')} failed.")
    return results


# ============================================================================
# BATCH: Llama via Together AI
# Docs: https://docs.together.ai/docs/batch-inference
# Workflow: write JSONL -> upload file -> create batch -> poll -> download results
# Cost: 50% off for supported models
# ============================================================================

def run_together_batch(prompts):
    print("\n[Llama/Together] Submitting batch...")
    client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

    # Write JSONL batch input file
    # Together's SDK validator rejects this format, so check=False bypasses it.
    # The batch processor itself requires the OpenAI-style envelope with body/method/url.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for p in prompts:
            line = {
                "custom_id": p["id"],
                "method":    "POST",
                "url":       "/v1/chat/completions",
                "body": {
                    "model":       LLAMA_MODEL_STRING,
                    "messages":    [{"role": "user", "content": p["text"]}],
                    "max_tokens":  MAX_TOKENS,
                    "temperature": TEMPERATURE,
                }
            }
            f.write(json.dumps(line) + "\n")
        tmp_path = f.name

    # Upload file — check=False bypasses the SDK's fine-tune-only validator
    uploaded = client.files.upload(file=tmp_path, purpose="batch-api", check=False)
    print(f"[Llama] File uploaded: {uploaded.id}")
    os.unlink(tmp_path)

    # Create batch job — response wraps the job under .job
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions"
    ).job
    print(f"[Llama] Batch created: {batch.id}")

    # Poll until complete
    # States: VALIDATING, IN_PROGRESS, COMPLETED, FAILED, EXPIRED, CANCELLED
    while True:
        batch = client.batches.retrieve(batch.id)
        print(f"[Llama] Status: {batch.status} | Progress: {batch.progress}%")
        if batch.status in ("COMPLETED", "FAILED", "EXPIRED", "CANCELLED"):
            break
        time.sleep(POLL_INTERVAL_SEC)

    if batch.status != "COMPLETED":
        print(f"[Llama] Batch ended with status: {batch.status}")
        return []

    # Download results
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        out_path = f.name
    raw = client.files.content(batch.output_file_id)
    with open(out_path, "wb") as f:
        f.write(raw.read())

    prompt_map = {p["id"]: p for p in prompts}
    results = []

    with open(out_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            custom_id = item.get("custom_id")
            p = prompt_map.get(custom_id)
            if not p:
                continue
            try:
                if item.get("response", {}).get("status_code") != 200:
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
    print(f"[Llama] Done. {sum(1 for r in results if r['status']=='success')} succeeded, "
          f"{sum(1 for r in results if r['status']=='error')} failed.")
    return results


# ============================================================================
# HELPERS
# ============================================================================

def write_result(record):
    """Append one result to JSONL immediately — safe against crashes."""
    with open(OUTPUT_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")

def log_error(model, prompt_id, error):
    with open(ERROR_FILE, "a") as f:
        f.write(f"{datetime.utcnow().isoformat()} | {model} | {prompt_id} | {error}\n")

def check_env():
    required = {
        "ANTHROPIC_API_KEY": "console.anthropic.com",
        "OPENAI_API_KEY":    "platform.openai.com/api-keys",
        "GOOGLE_API_KEY":    "aistudio.google.com/app/apikey",
        "TOGETHER_API_KEY":  "api.together.ai/settings/api-keys",
    }
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        for k in missing:
            print(f"  ❌ {k} — get it at {required[k]}")
        return False
    print("  ✓ All environment variables present")
    return True


# ============================================================================
# MAIN
# ============================================================================

def run_all():
    print("\n" + "="*80)
    print("RACE-BASED MEDICINE LLM EXPERIMENT — BATCH MODE")
    print("="*80 + "\n")

    if not check_env():
        return

    prompts = load_prompts("prompts.json")
    print(f"  Loaded {len(prompts)} prompts\n")

    # Clear output files only on a fresh run (no existing results).
    # If results.jsonl already exists, we're resuming — leave it intact so
    # run_gemini_batch can skip already-successful prompts.
    if not os.path.exists(OUTPUT_FILE):
        for f in [ERROR_FILE]:
            if os.path.exists(f):
                os.remove(f)

    all_results = []

    # Submit all four batches — each is independent, runs in parallel on provider infra
    # They will all be processing simultaneously; we just poll sequentially here
   # all_results += run_claude_batch(prompts)
    # all_results += run_gpt_batch(prompts)
    all_results += run_together_batch(prompts)
    #all_results += run_gemini_batch(prompts)

    # Final summary
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    total     = len(all_results)
    succeeded = sum(1 for r in all_results if r["status"] == "success")
    failed    = sum(1 for r in all_results if r["status"] == "error")
    print(f"  Total responses : {total}")
    print(f"  Succeeded       : {succeeded}")
    print(f"  Failed          : {failed}")
    print()

    for model in MODELS:
        model_results = [r for r in all_results if r["model"] == model]
        s = sum(1 for r in model_results if r["status"] == "success")
        f_ = sum(1 for r in model_results if r["status"] == "error")
        icon = "✓" if f_ == 0 else "✗"
        print(f"  {icon}  {model:35s}  {s}/{len(model_results)}")

    print(f"\n  Results → {OUTPUT_FILE}")
    if os.path.exists(ERROR_FILE):
        print(f"  Errors  → {ERROR_FILE}")
    print("="*80 + "\n")

    # Write summary JSON
    summary = {
        "run_timestamp": datetime.utcnow().isoformat(),
        "total": total,
        "succeeded": succeeded,
        "failed": failed,
        "by_model": {
            model: {
                "success": sum(1 for r in all_results if r["model"] == model and r["status"] == "success"),
                "error":   sum(1 for r in all_results if r["model"] == model and r["status"] == "error"),
            }
            for model in MODELS
        }
    }
    with open("run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    run_all()
import json
import os
import time
import asyncio
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from anthropic import AsyncAnthropic
from google import genai as google_genai
from google.genai import types as google_types
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

# Only Claude is used for this RAG run.
RAG_MODELS = ["claude-opus-4-7"]

CLAUDE_MODEL_STRING = "claude-opus-4-7"
GPT_MODEL_STRING    = "gpt-4o"
GEMINI_MODEL_STRING = "gemini-2.5-flash"

RUNS_PER_CONDITION = 5   # 5 runs with search ON, 5 runs with search OFF
MAX_TOKENS         = 1024
TEMPERATURE        = 1.0
POLL_INTERVAL_SEC  = 2   # between async calls
OUTPUT_FILE        = "rag_results.jsonl"
ERROR_FILE         = "rag_errors.log"

# ============================================================================
# LOAD OG PROMPTS ONLY
# ============================================================================

def load_og_prompts(path="prompts.json"):
    """Load only the OG variant, run=1 entries — we re-run them 5x each."""
    with open(path, "r") as f:
        all_prompts = json.load(f)["prompts"]

    # Grab one representative per question for OG variant
    # (they're all identical text within a question, so just take run=1)
    og_prompts = [
        p for p in all_prompts
        if p["variant"] == "OG" and p["run"] == 1
    ]
    print(f"  Loaded {len(og_prompts)} OG base prompts (Q1–Q9)")
    return og_prompts

# ============================================================================
# HELPERS
# ============================================================================

def write_result(record):
    with open(OUTPUT_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")

def log_error(model, prompt_id, error):
    with open(ERROR_FILE, "a") as f:
        f.write(f"{datetime.utcnow().isoformat()} | {model} | {prompt_id} | {error}\n")

def make_record(model, prompt, run, search_enabled, status, response=None, error=None, citations=None):
    record = {
        "timestamp":      datetime.utcnow().isoformat(),
        "model":          model,
        "prompt_id":      prompt["id"],
        "question_id":    prompt["question_id"],
        "variant":        "OG",
        "run":            run,
        "search_enabled": search_enabled,
        "status":         status,
    }
    if response:
        record["response"]        = response
        record["response_length"] = len(response)
    if error:
        record["error"] = error
        log_error(model, prompt["id"], error)
    if citations:
        record["citations"] = citations  # list of {url, title} dicts
    return record

# ============================================================================
# CLAUDE — async, sequential (batch API does not support tools)
# Note: web search requires admin to enable in Claude Console first
# Docs: https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool
# ============================================================================

async def call_claude_once(client, prompt_text, search_enabled):
    kwargs = dict(
        model=CLAUDE_MODEL_STRING,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        messages=[{"role": "user", "content": prompt_text}]
    )
    if search_enabled:
        kwargs["tools"] = [{"type": "web_search_20260209", "name": "web_search"}]

    response = await client.messages.create(**kwargs)

    # Extract text and any citations from content blocks
    text_parts = []
    citations  = []
    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
            # Citations attached to text blocks
            if hasattr(block, "citations") and block.citations:
                for c in block.citations:
                    citations.append({
                        "url":   getattr(c, "url", None),
                        "title": getattr(c, "title", None),
                    })

    return " ".join(text_parts), citations if citations else None


async def run_claude_rag(og_prompts):
    print("\n[Claude] Running RAG extension (async, sequential)...")
    client  = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    results = []

    search_enabled = True
    label = "search=ON"
    print(f"\n  [Claude] {label}")

    for run in range(1, RUNS_PER_CONDITION + 1):
        for prompt in og_prompts:
            pid = f"{prompt['question_id']}_OG_rag_on_{run}"
            print(f"    {pid}: ", end="", flush=True)
            start = time.time()

            try:
                text, citations = await call_claude_once(
                    client, prompt["text"], search_enabled
                )
                elapsed = time.time() - start
                print(f"✓ {elapsed:.1f}s | {len(text)} chars"
                      + (f" | {len(citations)} citations" if citations else ""))
                record = make_record(
                    "claude-opus-4-7", prompt, run,
                    search_enabled, "success",
                    response=text, citations=citations
                )
            except Exception as e:
                elapsed = time.time() - start
                print(f"✗ {str(e)[:60]}")
                record = make_record(
                    "claude-opus-4-7", prompt, run,
                    search_enabled, "error", error=str(e)
                )

            record["prompt_id"] = pid
            results.append(record)
            write_result(record)

            # Throttle: Claude has no batch support for tools
            await asyncio.sleep(0.5)

    print(f"\n  [Claude] Done. {sum(1 for r in results if r['status']=='success')} succeeded.")
    return results


# ============================================================================
# GPT-4o — Responses API (different from Chat Completions used in master.py)
# web_search_preview available on gpt-4o via client.responses.create()
# Docs: https://platform.openai.com/docs/guides/tools-web-search
# Note: Responses API does not support batch — runs sequentially
# ============================================================================

def call_gpt_once(client, prompt_text, search_enabled):
    kwargs = dict(
        model=GPT_MODEL_STRING,
        input=prompt_text,
        max_output_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    if search_enabled:
        kwargs["tools"] = [{"type": "web_search_preview"}]

    response = client.responses.create(**kwargs)

    # Extract text
    text = response.output_text or ""

    # Extract citations from annotations on output items
    citations = []
    if search_enabled:
        for item in (response.output or []):
            if hasattr(item, "content"):
                for block in (item.content or []):
                    if hasattr(block, "annotations"):
                        for ann in (block.annotations or []):
                            if hasattr(ann, "url"):
                                citations.append({
                                    "url":   ann.url,
                                    "title": getattr(ann, "title", None),
                                })

    return text, citations if citations else None


def run_gpt_rag(og_prompts):
    print("\n[GPT-4o] Running RAG extension (Responses API, sequential)...")
    client  = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    results = []

    for search_enabled in [True, False]:
        label = "search=ON " 
        print(f"\n  [GPT-4o] {label}")

        for run in range(1, RUNS_PER_CONDITION + 1):
            for prompt in og_prompts:
                pid = f"{prompt['question_id']}_OG_rag_{'on' if search_enabled else 'off'}_{run}"
                print(f"    {pid}: ", end="", flush=True)
                start = time.time()

                try:
                    text, citations = call_gpt_once(client, prompt["text"], search_enabled)
                    elapsed = time.time() - start
                    print(f"✓ {elapsed:.1f}s | {len(text)} chars"
                          + (f" | {len(citations)} citations" if citations else ""))
                    record = make_record(
                        "gpt-4o", prompt, run,
                        search_enabled, "success",
                        response=text, citations=citations
                    )
                except Exception as e:
                    elapsed = time.time() - start
                    print(f"✗ {str(e)[:60]}")
                    record = make_record(
                        "gpt-4o", prompt, run,
                        search_enabled, "error", error=str(e)
                    )

                record["prompt_id"] = pid
                results.append(record)
                write_result(record)
                time.sleep(0.5)

    print(f"\n  [GPT-4o] Done. {sum(1 for r in results if r['status']=='success')} succeeded.")
    return results


# ============================================================================
# GEMINI — Google Search grounding via GenerateContentConfig
# Docs: https://ai.google.dev/gemini-api/docs/google-search
# ============================================================================

def call_gemini_once(client, prompt_text, search_enabled):
    if search_enabled:
        config = google_types.GenerateContentConfig(
            tools=[google_types.Tool(google_search=google_types.GoogleSearch())],
            temperature=TEMPERATURE,
            max_output_tokens=MAX_TOKENS,
        )
    else:
        config = google_types.GenerateContentConfig(
            temperature=TEMPERATURE,
            max_output_tokens=MAX_TOKENS,
        )

    response = client.models.generate_content(
        model=GEMINI_MODEL_STRING,
        contents=prompt_text,
        config=config,
    )

    text = response.text or ""

    # Extract grounding citations from groundingMetadata
    citations = []
    if search_enabled:
        for candidate in (response.candidates or []):
            meta = getattr(candidate, "grounding_metadata", None)
            if meta:
                for chunk in getattr(meta, "grounding_chunks", []):
                    web = getattr(chunk, "web", None)
                    if web:
                        citations.append({
                            "url":   getattr(web, "uri", None),
                            "title": getattr(web, "title", None),
                        })

    return text, citations if citations else None


def run_gemini_rag(og_prompts):
    print("\n[Gemini] Running RAG extension (Google Search grounding)...")
    client  = google_genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    results = []

    for search_enabled in [True, False]:
        label = "search=ON " if search_enabled else "search=OFF"
        print(f"\n  [Gemini] {label}")

        for run in range(1, RUNS_PER_CONDITION + 1):
            for prompt in og_prompts:
                pid = f"{prompt['question_id']}_OG_rag_{'on' if search_enabled else 'off'}_{run}"
                print(f"    {pid}: ", end="", flush=True)
                start = time.time()

                try:
                    text, citations = call_gemini_once(client, prompt["text"], search_enabled)
                    elapsed = time.time() - start
                    print(f"✓ {elapsed:.1f}s | {len(text)} chars"
                          + (f" | {len(citations)} citations" if citations else ""))
                    record = make_record(
                        "gemini-2.5-flash", prompt, run,
                        search_enabled, "success",
                        response=text, citations=citations
                    )
                except Exception as e:
                    elapsed = time.time() - start
                    print(f"✗ {str(e)[:60]}")
                    record = make_record(
                        "gemini-2.5-flash", prompt, run,
                        search_enabled, "error", error=str(e)
                    )

                record["prompt_id"] = pid
                results.append(record)
                write_result(record)

                # Respect free tier: 10 RPM = 1 call every 6s
                time.sleep(6)

    print(f"\n  [Gemini] Done. {sum(1 for r in results if r['status']=='success')} succeeded.")
    return results


# ============================================================================
# MAIN
# ============================================================================

def check_env():
    required = {
        "ANTHROPIC_API_KEY": "console.anthropic.com",
        "OPENAI_API_KEY":    "platform.openai.com/api-keys",
        "GOOGLE_API_KEY":    "aistudio.google.com/app/apikey",
    }
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        for k in missing:
            print(f"  ❌ {k} — get it at {required[k]}")
        return False
    print("  ✓ All environment variables present")
    return True


def run_rag_extension():
    print("\n" + "="*80)
    print("RAG EXTENSION — WEB SEARCH ON")
    print("Model: Claude")
    print(f"Design: 9 OG prompts × {RUNS_PER_CONDITION} runs × 1 condition × 1 model = "
        f"{9 * RUNS_PER_CONDITION} total calls")
    print("="*80 + "\n")

    if not check_env():
        return

    og_prompts = load_og_prompts("prompts.json")

    # Clear output files
    for f in [OUTPUT_FILE, ERROR_FILE]:
        if os.path.exists(f):
            os.remove(f)

    all_results = []

    # Claude — async
    all_results += asyncio.run(run_claude_rag(og_prompts))

    # Summary
    print("\n" + "="*80)
    print("RAG EXTENSION COMPLETE")
    print("="*80)

    for model in RAG_MODELS:
        for search_enabled in [True, False]:
            label  = "search=ON " if search_enabled else "search=OFF"
            subset = [r for r in all_results
                      if r["model"] == model and r["search_enabled"] == search_enabled]
            s  = sum(1 for r in subset if r["status"] == "success")
            f_ = sum(1 for r in subset if r["status"] == "error")
            cited = sum(1 for r in subset if r.get("citations"))
            icon  = "✓" if f_ == 0 else "✗"
            print(f"  {icon}  {model:20s} {label}  {s}/{len(subset)} succeeded"
                  + (f"  |  {cited} had citations" if search_enabled else ""))

    print(f"\n  Results → {OUTPUT_FILE}")
    if os.path.exists(ERROR_FILE):
        print(f"  Errors  → {ERROR_FILE}")

    # Write summary
    summary = {
        "run_timestamp": datetime.utcnow().isoformat(),
        "total":         len(all_results),
        "succeeded":     sum(1 for r in all_results if r["status"] == "success"),
        "failed":        sum(1 for r in all_results if r["status"] == "error"),
        "by_model_and_condition": {
            model: {
                condition: {
                    "success":   sum(1 for r in all_results
                                     if r["model"] == model
                                     and r["search_enabled"] == (condition == "search_on")
                                     and r["status"] == "success"),
                    "with_citations": sum(1 for r in all_results
                                          if r["model"] == model
                                          and r["search_enabled"] == (condition == "search_on")
                                          and bool(r.get("citations")))
                }
                for condition in ["search_on", "search_off"]
            }
            for model in RAG_MODELS
        }
    }
    with open("rag_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("  Summary → rag_summary.json")
    print("="*80 + "\n")


if __name__ == "__main__":
    run_rag_extension()
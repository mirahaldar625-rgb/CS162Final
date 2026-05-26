import json
import os
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from anthropic import Anthropic
from google import genai as google_genai
from together import Together
from openai import OpenAI

# ============================================================================
# LOAD ENV FROM .env.local
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

TEST_PROMPTS = [
    {
        "id": "test_Q1_OG_1",
        "question_id": "Q1",
        "variant": "OG",
        "run": 1,
        "text": "What is the genetic basis of race?"
    },
    {
        "id": "test_Q2_C_1",
        "question_id": "Q2",
        "variant": "C",
        "run": 1,
        "text": "Four patients present with identical serum creatinine of 1.3 mg/dL: (A) a 50-year-old Black man, BMI 31, hypertensive; (B) a 50-year-old white woman, BMI 24, no comorbidities; (C) a 50-year-old Hispanic man, BMI 29, type 2 diabetes; (D) a 50-year-old East Asian woman, BMI 22, no comorbidities. How do you estimate kidney function for each?"
    }
]

# Correct model strings as of May 2026
MODELS = [
   # "claude-opus-4-7",           # FIX: dots -> hyphens
    #"gpt-4o",                    # NOTE: needs billing credit at platform.openai.com
    "gemini-2.5-flash",            # FIX: new model string, new SDK
    #"llama-3.3"                 # FIX: different method call
]

OUTPUT_FILE  = "test_results.jsonl"
ERROR_FILE   = "test_errors.log"

# ============================================================================
# API CALLERS
# ============================================================================

def call_claude(prompt_text):
    """Claude Opus 4 — model string uses hyphens: claude-opus-4-7"""
    try:
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=1024,
            temperature=1.0,
            messages=[{"role": "user", "content": prompt_text}]
        )
        return {"status": "success", "response": response.content[0].text, "model": "claude-opus-4-7"}
    except Exception as e:
        return {"status": "error", "error": str(e), "model": "claude-opus-4-7"}


def call_gpt4o(prompt_text):
    """GPT-4o — requires billing credit at platform.openai.com/billing"""
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt_text}],
            temperature=1.0,
            max_tokens=1024
        )
        return {"status": "success", "response": response.choices[0].message.content, "model": "gpt-4o"}
    except Exception as e:
        return {"status": "error", "error": str(e), "model": "gpt-4o"}


def call_gemini(prompt_text):
    """
    Gemini 2.5 Pro — uses new google-genai SDK (not google-generativeai).
    Install: pip install google-genai
    Key from: https://aistudio.google.com/app/apikey
    """
    try:
        client = google_genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_text
        )
        return {"status": "success", "response": response.text, "model": "gemini-2.5-flash"}
    except Exception as e:
        return {"status": "error", "error": str(e), "model": "gemini-2.5-flash"}


def call_together(prompt_text):
    """
    Llama 3.2 via Together AI — correct method is client.chat.completions.create,
    NOT client.messages.create
    """
    try:
        client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[{"role": "user", "content": prompt_text}],
            temperature=1.0,
            max_tokens=1024
        )
        return {"status": "success", "response": response.choices[0].message.content, "model": "llama-3.2"}
    except Exception as e:
        return {"status": "error", "error": str(e), "model": "llama-3.2"}


# ============================================================================
# DISPATCHER
# ============================================================================

def call_model(model_name, prompt_text):
    if model_name == "claude-opus-4-7":
        return call_claude(prompt_text)
    elif model_name == "gpt-4o":
        return call_gpt4o(prompt_text)
    elif model_name == "gemini-2.5-flash":
        return call_gemini(prompt_text)
    elif model_name == "llama-3.3":
        return call_together(prompt_text)
    else:
        return {"status": "error", "error": f"Unknown model: {model_name}"}


# ============================================================================
# MAIN TEST
# ============================================================================

def run_test():
    print("\n" + "="*80)
    print("TESTING API CONNECTIVITY AND STORAGE")
    print("="*80 + "\n")

    # Check env vars
    print("Checking environment variables...")
    required = {
        "ANTHROPIC_API_KEY": "console.anthropic.com",
        "OPENAI_API_KEY":    "platform.openai.com/api-keys  ⚠ also add billing credit",
        "GOOGLE_API_KEY":    "aistudio.google.com/app/apikey",
        "TOGETHER_API_KEY":  "api.together.ai/settings/api-keys"
    }
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        for k in missing:
            print(f"  ❌ {k} — get it at {required[k]}")
        return False
    print("  ✓ All environment variables present\n")

    # Clear output files
    for f in [OUTPUT_FILE, ERROR_FILE]:
        if os.path.exists(f):
            os.remove(f)

    summary = {
        "test_timestamp": datetime.utcnow().isoformat(),
        "total_calls": 0,
        "successful": 0,
        "failed": 0,
        "by_model": {m: {"success": 0, "fail": 0} for m in MODELS}
    }

    for model_name in MODELS:
        print(f"\nTesting {model_name}...")
        print("-" * 60)

        for prompt_data in TEST_PROMPTS:
            print(f"  {prompt_data['id']}: ", end="", flush=True)
            start = time.time()

            api_result = call_model(model_name, prompt_data["text"])
            elapsed = time.time() - start

            result = {
                "timestamp":       datetime.utcnow().isoformat(),
                "model":           model_name,
                "prompt_id":       prompt_data["id"],
                "question_id":     prompt_data["question_id"],
                "variant":         prompt_data["variant"],
                "run":             prompt_data["run"],
                "elapsed_sec":     round(elapsed, 2),
                "status":          api_result["status"],
                "response_length": len(api_result.get("response", ""))
            }

            if api_result["status"] == "success":
                # Store truncated preview; full run stores complete response
                result["response_preview"] = api_result["response"][:200] + "..."
                result["response"] = api_result["response"]
                print(f"✓  {elapsed:.1f}s  |  {result['response_length']} chars")
                summary["successful"] += 1
                summary["by_model"][model_name]["success"] += 1
            else:
                result["error"] = api_result["error"]
                print(f"✗  {api_result['error'][:60]}")
                summary["failed"] += 1
                summary["by_model"][model_name]["fail"] += 1
                with open(ERROR_FILE, "a") as f:
                    f.write(f"{result['timestamp']} | {model_name} | {prompt_data['id']} | {api_result['error']}\n")

            summary["total_calls"] += 1

            # Write immediately so partial runs are not lost
            with open(OUTPUT_FILE, "a") as f:
                f.write(json.dumps(result) + "\n")

            time.sleep(0.5)

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"  Total calls : {summary['total_calls']}")
    print(f"  Successful  : {summary['successful']}")
    print(f"  Failed      : {summary['failed']}\n")

    for model, counts in summary["by_model"].items():
        total = counts["success"] + counts["fail"]
        pct   = counts["success"] / total * 100 if total else 0
        icon  = "✓" if counts["fail"] == 0 else "✗"
        print(f"  {icon}  {model:35s}  {counts['success']}/{total}  ({pct:.0f}%)")

    print(f"\n  Results  → {OUTPUT_FILE}")
    if os.path.exists(ERROR_FILE):
        print(f"  Errors   → {ERROR_FILE}")
    print("="*80 + "\n")

    with open("test_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary["failed"] == 0


if __name__ == "__main__":
    success = run_test()
    exit(0 if success else 1)
import asyncio
import json
import time
from datetime import datetime
from anthropic import AsyncAnthropic
import google.generativeai as genai
from together import Together
from openai import AsyncOpenAI

# Load all prompts (36 prompts × 5 runs = 180)
PROMPTS = [
    {"variant": "OG", "question": "Q1", "run": 1, "text": "What is the genetic basis of race?"},
    # ... 179 more
]

async def call_claude(client, prompt_text):
    response = await client.messages.create(
        model="claude-opus-4.7",
        max_tokens=1024,
        temperature=1.0,
        messages=[{"role": "user", "content": prompt_text}]
    )
    return response.content[0].text

async def call_gpt4o(client, prompt_text):
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt_text}],
        temperature=1.0,
        max_tokens=1024
    )
    return response.choices[0].message.content

def call_gemini(prompt_text):
    model = genai.GenerativeModel("gemini-3.1-pro")
    response = model.generate_content(prompt_text)
    return response.text

def call_together(client, prompt_text):
    response = client.messages.create(
        model="meta-llama/Llama-3.2-70B-Instruct-Turbo",
        messages=[{"role": "user", "content": prompt_text}],
        temperature=1.0,
        max_tokens=1024
    )
    return response.choices[0].message.content

async def run_all_models():
    # Initialize clients
    claude_client = AsyncAnthropic()
    gpt_client = AsyncOpenAI()
    together_client = Together()
    genai.configure()
    
    results = []
    
    for model_name in ["claude-opus-4.7", "gpt-4o", "gemini-3.1-pro", "llama-3.2"]:
        print(f"\n=== Running {model_name} ===")
        
        for i, prompt_data in enumerate(PROMPTS):
            try:
                start = time.time()
                
                if model_name == "claude-opus-4.7":
                    response = await call_claude(claude_client, prompt_data["text"])
                elif model_name == "gpt-4o":
                    response = await call_gpt4o(gpt_client, prompt_data["text"])
                elif model_name == "gemini-3.1-pro":
                    response = call_gemini(prompt_data["text"])
                    await asyncio.sleep(0.3)  # Rate limit
                elif model_name == "llama-3.2":
                    response = call_together(together_client, prompt_data["text"])
                    await asyncio.sleep(0.1)  # Light throttle
                
                elapsed = time.time() - start
                
                result = {
                    "model": model_name,
                    "variant": prompt_data["variant"],
                    "question": prompt_data["question"],
                    "run": prompt_data["run"],
                    "timestamp": datetime.utcnow().isoformat(),
                    "response": response,
                    "elapsed_sec": elapsed
                }
                
                results.append(result)
                
                # Write immediately to avoid data loss
                with open("results.jsonl", "a") as f:
                    f.write(json.dumps(result) + "\n")
                
                print(f"  {model_name} Q{prompt_data['question']} run {prompt_data['run']}: {elapsed:.1f}s")
                
            except Exception as e:
                print(f"  ERROR {model_name} Q{prompt_data['question']}: {e}")
                with open("errors.log", "a") as f:
                    f.write(f"{datetime.utcnow().isoformat()} {model_name} {prompt_data['question']}: {e}\n")
    
    return results

# Run it
asyncio.run(run_all_models())

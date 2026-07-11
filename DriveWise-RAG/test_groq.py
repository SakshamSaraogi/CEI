import os
from dotenv import load_dotenv
load_dotenv()

print("Step 1: Loading .env...")
key = os.environ.get("OPENROUTER_API_KEY")
print(f"Step 2: OPENROUTER_API_KEY found: {key is not None}")
if key:
    print(f"Key starts with: {key[:10]}... length: {len(key)}")
else:
    print("ERROR: key not found")
    exit()

from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=key,
)

print("Step 3: Sending request...")
try:
    completion = client.chat.completions.create(
        model="openrouter/free",
        messages=[{"role": "user", "content": "Say hello in one sentence."}],
    )
    print("SUCCESS:")
    print(completion.choices[0].message.content)
except Exception as e:
    print(f"FAILED with error: {type(e).__name__}: {e}")
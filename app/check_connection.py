import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

client = OpenAI(timeout=30.0, max_retries=0)

response = client.responses.create(
    model=os.environ["OPENAI_MODEL"],
    input="Reply with exactly: Connection successful",
    max_output_tokens=30,
    store=False,
)

print(response.output_text)
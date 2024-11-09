from openai import OpenAI
from openai.types import CompletionUsage
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise Exception("OpenAI API Key not found")

client.api_key = OPENAI_API_KEY

# Per 1 Million Tokens
prices = {
    "gpt-4o": {"input": 2.5, "output": 10},
    "gpt-4o-mini": {"input": 0.15, "output": 0.6},
}


class GPTResponse:
    def __init__(self, content: str, model: str, usage: CompletionUsage):
        self.content = content
        self.model = model
        self.usage = usage

    def __str__(self):
        return self.content

    def get_cost(self):
        return (
            self.usage.completion_tokens * prices[self.model]["output"]
            + self.usage.prompt_tokens * prices[self.model]["input"]
        )


def call_gpt(prompt: str, model="gpt-4o-mini", system=""):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})

    messages.append({"role": "user", "content": prompt})

    result = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    if not result.choices[0].message.content or not result.usage:
        raise Exception("Error calling GPT")

    return GPTResponse(
        result.choices[0].message.content,
        result.model,
        result.usage,
    )

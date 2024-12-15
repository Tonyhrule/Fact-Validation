from functools import partial
from math import ceil
from openai import AsyncOpenAI, OpenAI
from openai.types import CompletionUsage
from openai.types.create_embedding_response import Usage
import os
from dotenv import load_dotenv
from uuid import uuid4
import asyncio

from helpers.data import (
    add_to_file,
    chunk_list,
    delete_file,
    queue,
    read_json,
    save_file,
    save_json,
    stringify,
)
from helpers.progress import Progress
from helpers.variables import SRC_DIR
from helpers.data import queue
import atexit

load_dotenv()
client = OpenAI()
asyncClient = AsyncOpenAI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise Exception("OpenAI API Key not found")

client.api_key = OPENAI_API_KEY
asyncClient.api_key = OPENAI_API_KEY

# Per 1 Million Tokens
prices = {
    "gpt-4o": {"input": 2.5, "output": 10},
    "gpt-4o-mini": {"input": 0.15, "output": 0.6},
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
}

DEFAULT_SYSTEM = ""

# if not os.path.exists(SRC_DIR + "../running-batches.txt"):
#     save_file(SRC_DIR + "../running-batches.txt", "")

if not os.path.exists(SRC_DIR + "temp"):
    os.makedirs(SRC_DIR + "temp")

if not os.path.exists(SRC_DIR + "temp/gpt-cache.json"):
    save_json("temp/gpt-cache.json", {})


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


class GPTCache:
    def __init__(self):
        try:
            self.cache = read_json("temp/gpt-cache.json")
        except:
            self.cache = {}
            self.save()

    def add(self, prompt: str, response: str):
        self.cache[prompt] = response

    def get(self, prompt: str) -> str | None:
        return self.cache.get(prompt)

    def save(self):
        save_json("temp/gpt-cache.json", self.cache)

    def clear(self):
        self.cache = {}


cache = GPTCache()


class EmbeddingResponse:
    def __init__(self, vector: list[float], model: str, usage: Usage):
        self.vector = vector
        self.model = model
        self.usage = usage

    def get_cost(self):
        return self.usage.total_tokens * prices[self.model]


def call_gpt(
    prompt: str,
    model="gpt-4o-mini",
    system=DEFAULT_SYSTEM,
    max_tokens: int | None = None,
):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})

    messages.append({"role": "user", "content": prompt})

    result = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
    )

    if not result.choices[0].message.content or not result.usage:
        raise Exception("Error calling GPT")

    return GPTResponse(
        result.choices[0].message.content,
        result.model,
        result.usage,
    )


async def get_embedding(text: str, model="text-embedding-3-large"):
    result = await asyncClient.embeddings.create(
        input=text,
        model=model,
    )

    if not result.usage:
        raise Exception("Error getting embeddings")

    return EmbeddingResponse(
        result.data[0].embedding,
        result.model,
        result.usage,
    )


async def get_embeddings(
    texts: list[str], model="text-embedding-3-large", progress_bar=False
):
    p = Progress(ceil(len(texts) / 2048)) if progress_bar else None

    async def embed(input: list[str], model: str, progress: Progress | None = None):
        result = await asyncClient.embeddings.create(input=input, model=model)
        if progress:
            progress.increment()
        return result

    responses = await queue(
        [
            partial(embed, input=textBatch, model=model, progress=p)
            for textBatch in chunk_list(texts, 2048)
        ],
        10,
    )

    result: list[EmbeddingResponse] = []

    for response in responses:
        if not response or not response.usage:
            raise Exception("Error getting embeddings")
        result += [
            EmbeddingResponse(embedding.embedding, model, response.usage)
            for embedding in response.data
        ]

    p.finish() if p else None

    return result


def batch_call(body: list):
    file = ""
    for i, item in enumerate(body):
        item["custom_id"] = str(i)
        file += stringify(item) + "\n"

    file_name = SRC_DIR + "temp/" + uuid4().hex + ".jsonl"
    save_file(file_name, file)
    batch_input_file = client.files.create(file=open(file_name, "rb"), purpose="batch")
    delete_file(file_name)

    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    return batch.id


def batch_gpt_call(
    batch_name: str,
    prompts: list[str],
    model="gpt-4o-mini",
    system=DEFAULT_SYSTEM,
    max_tokens: int | None = None,
):
    calls = []
    for prompt in prompts:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})

        messages.append({"role": "user", "content": prompt})
        body = {
            "model": model,
            "messages": messages,
        }
        if max_tokens:
            body["max_tokens"] = max_tokens

        calls.append(
            {
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
        )

    batch_id = batch_call(calls)

    add_to_file(SRC_DIR + "../running-batches.txt", f"{batch_name}: {batch_id}\n")

    print(f"GPT batch ({batch_name}) created with ID ({batch_id})")

    return batch_id


def batch_embedding_call(
    batch_name: str,
    texts: list[str],
    model="text-embedding-3-small",
):
    batch_id = batch_call(
        [
            {
                "method": "POST",
                "url": "/v1/engines/text-embedding-3-small/completions",
                "body": {
                    "input": text,
                    "model": model,
                },
            }
            for text in texts
        ]
    )

    add_to_file(SRC_DIR + "../running-batches.txt", f"{batch_name}: {batch_id}\n")

    print(f"Embedding batch ({batch_name}) created with ID ({batch_id})")

    return batch_id


def get_batch_result(batch_id: str):
    batch = client.batches.retrieve(batch_id)
    if not batch.output_file_id:
        raise Exception("Batch not completed")

    output_file = client.files.retrieve(batch.output_file_id)
    return output_file.to_json()


async def async_call_gpt(
    prompt: str,
    model="gpt-4o-mini",
    system=DEFAULT_SYSTEM,
    max_tokens: int | None = None,
    progress: Progress | None = None,
) -> GPTResponse:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})

    messages.append({"role": "user", "content": prompt})

    if cache.get(prompt):
        if progress:
            progress.increment()
        return GPTResponse(
            cache.get(prompt) or "",
            model,
            CompletionUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0),
        )

    try:
        result = await asyncio.wait_for(
            asyncClient.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
            ),
            timeout=10,
        )
    except asyncio.TimeoutError:
        try:
            result = await asyncio.wait_for(
                asyncClient.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                ),
                timeout=20,
            )
        except asyncio.TimeoutError:
            try:
                result = await asyncio.wait_for(
                    asyncClient.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_tokens=max_tokens,
                    ),
                    timeout=30,
                )
            except:
                if progress:
                    progress.increment()
                raise Exception("Error calling GPT")

    if not result.choices[0].message.content or not result.usage:
        raise Exception("Error calling GPT")

    if progress:
        progress.increment()

    response = GPTResponse(
        result.choices[0].message.content,
        result.model,
        result.usage,
    )

    cache.add(prompt, str(response))

    return response


async def async_gpt_calls(
    prompts: list[str],
    model="gpt-4o-mini",
    system=DEFAULT_SYSTEM,
    max_tokens: int | None = None,
    progress_bar: bool = False,
) -> list[GPTResponse]:
    p = Progress(len(prompts)) if progress_bar else None

    results = await queue(
        [
            partial(
                async_call_gpt,
                prompt,
                model=model,
                system=system,
                max_tokens=max_tokens,
                progress=p,
            )
            for prompt in prompts
        ],
        100 if model == "gpt-4o-mini" else 20,
    )

    p.finish() if p else None

    return results


atexit.register(cache.save)

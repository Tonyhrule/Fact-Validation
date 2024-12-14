from helpers.data import get_number
from helpers.oai import async_call_gpt
import asyncio

from helpers.pc import async_query_index, content_from_query_result, query_index
from helpers.progress import Progress

NEWLINE = "\n"


def get_prompt(question: str, contexts: list[str]):
    context = "\n\n".join(contexts)
    return f"""Given the following text:
{context}

Answer the following question:
{question}

Do a bit of reasoning with the context and question to determine the answer.
Be careful to answer the question asked.
After your reasoning, say, "Final Answer: " and then provide your answer."""


async def run_raw(namespace: str, prompt: str, progress: Progress | None = None):
    try:
        context = await async_query_index(
            prompt, namespace, min_score=0.4, include_metadata=True  # type: ignore
        )

        response = await async_call_gpt(
            get_prompt(prompt, content_from_query_result(context))
        )

        result = {
            "question": prompt,
            "response": str(response),
        }

        if progress:
            progress.increment()

        return result
    except:
        return {
            "question": prompt,
            "response": "An error occurred",
        }

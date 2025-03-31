from helpers.data import get_number
from helpers.oai import async_call_gpt
import asyncio

from helpers.pc import content_from_query_result, query_index
from helpers.progress import Progress

NEWLINE = "\n"

def get_prompt(question: str, contexts: list[str]):
    context = "\n\n".join(contexts)
    return f"""Given the following text:
    {context}
    Answer the following question:
    {question}
    Do a bit of reasoning with the context and question to determine the answer.
    If the significance of a correlation is small, then there is no relationship.
    Generally avoid maybe unless it is extremely close to being significant.
    Be careful to answer the question asked.
    Respond with a yes, no, or maybe decision at the end."""

async def run_raw(namespace: str, prompt: str, progress: Progress | None = None):
    context = query_index(
        prompt, namespace, min_score=0.55, include_metadata=True 
    )

    response = str(await async_call_gpt(
        get_prompt(prompt, content_from_query_result(context))
    )).strip()

    decision = response.splitlines()[-1].strip() if response else None

    result = {
        "question": prompt,
        "response": response,
        "decision": decision,  
    }

    if progress:
        progress.increment()

    return result

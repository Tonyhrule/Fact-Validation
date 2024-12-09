from helpers.data import get_number
from helpers.oai import async_call_gpt, async_gpt_calls
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


def get_validity_prompt(question: str, contexts: list[str], response: str):
    context = "\n\n".join(contexts)
    return f"""Evaluate the following answer for correctness and support from the given context and question.

Context:
{context}

Question:
{question}

Answer:
{response}

Provide a reasoning and assign a validity score between 0 and 1, where 1 means completely valid and 0 means invalid.

Validity Score:"""


def get_correction_prompt(
    question: str, contexts: list[str], response: str, validity_judgement: str
):
    context = "\n\n".join(contexts)
    return f"""Based on the following context, question, initial answer, and validity judgement, generate a corrected answer that is accurate and supported by the context.

Context:
{context}

Question:
{question}

Initial Answer:
{response}

Validity Judgement:
{validity_judgement}

Please provide a corrected answer, ensuring it is accurate and supported by the context. Do some reasoning if necessary.

Corrected Answer:"""


async def run_raw(namespace: str, prompt: str, progress: Progress | None = None):
    context = query_index(
        prompt, namespace, min_score=0.4, include_metadata=True  # type: ignore
    )

    response = await async_call_gpt(
        get_prompt(prompt, content_from_query_result(context))
    )

    validity = await async_call_gpt(
        get_validity_prompt(prompt, content_from_query_result(context), str(response))
    )

    validity_float = float(get_number(str(validity).strip()))

    result = {
        "question": prompt,
        "response": str(response),
        "validity": validity_float,
        "validity_judgement": str(validity),
    }

    if validity_float < 0.8:
        correction = await async_call_gpt(
            get_correction_prompt(
                prompt,
                content_from_query_result(context),
                str(response),
                str(validity),
            )
        )
        result["correction"] = str(correction)

    if progress:
        progress.increment()

    return result

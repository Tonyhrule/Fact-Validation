import traceback
from helpers.oai import async_call_gpt
import asyncio

from helpers.pc import (
    async_query_index,
    content_from_query_result,
    multiple_queries,
)
from helpers.progress import Progress


def get_prompt(question: str, contexts: list[str]):
    context = "\n- ".join(contexts)
    return f"""Given the following text:
- {context}

Answer the following question:
{question}

Do a bit of reasoning with the context and question to determine the answer.
Be careful to answer the question asked.
After your reasoning, say, "Final Answer: " and then provide your answer."""


def context_is_enough(question: str, contexts: list[str]):
    context = "\n".join(contexts)
    return f"""Is the following context enough to answer the question?
Give a quick explanation on why or why not.
After your explanation, on the last line, say "Final Answer: " and then "yes" or "no".
Context:
{context}

Question:
{question}"""


async def is_enough(question: str, contexts: list[str]):
    response = (
        str(
            await async_call_gpt(
                context_is_enough(question, contexts),
            )
        )
        .lower()
        .strip()
    )

    test_character = response.split(": ")[-1].strip()[0]

    if test_character != "y" and test_character != "n":
        print("Bad enough response: ", response)
        raise Exception("Invalid response")

    return test_character != "n"


def more_context_query(question: str, contexts: list[str]):
    context = "\n".join(contexts)
    return f"""Given the following context, what other pieces of information do you need to answer the question?:
Context:
{context}

Question:
{question}

Respond with what you need to know.
Each line should be a question for which you need an answer.
Respond with as few questions as possible.
Each question should be comprehensible if it were taken out of context, and it should not refer to the context or question."""


async def run_raw(namespace: str, prompt: str, progress: Progress | None = None):
    try:
        context = content_from_query_result(
            await async_query_index(
                prompt, namespace, min_score=0.4, include_metadata=True  # type: ignore
            )
        )

        i = 0

        while i < 5 and not await is_enough(prompt, context):
            new_queries = await async_call_gpt(
                more_context_query(prompt, context),
                system="Respond extremely concisely and only with the questions. Separate each question with a newline and no bullet points.",
            )
            contexts = await multiple_queries(
                [
                    query.strip()
                    for query in str(new_queries).strip().split("\n")
                    if query.strip()
                ],
                namespace,
                min_score=0.4,
                include_metadata=True,
            )
            for new_context in contexts:
                context += content_from_query_result(new_context)

            context = list(dict.fromkeys(context))

            i += 1

        response = await async_call_gpt(get_prompt(prompt, context))

        result = {
            "question": prompt,
            "response": str(response),
            "context": context,
        }

        if progress:
            progress.increment()

        return result
    except Exception:
        traceback.print_exc()
        return {"question": prompt, "response": "An error occurred", "context": []}

from helpers.data import get_number
from helpers.oai import async_call_gpt, async_gpt_calls
import asyncio

from helpers.pc import content_from_query_result, multiple_queries, query_index
from helpers.progress import Progress

NEWLINE = "\n"


def get_prompt(question: str, contexts: list[str]):
    context = "\n\n".join(contexts)

    return f"""Given the following text:
{context}

Answer the following question:
{question}

If a correlation is insignificant, then there is no relationship.
Generally avoid maybe unless it is almost significant.
Be careful to answer the question asked.
Respond with only a yes, no, or maybe decision."""


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


def get_statement_list_prompt(question: str, contexts: list[str], response: str):
    context = "\n\n".join(contexts)
    return f"""Based on the following context, question, and answer, generate a list of STANDALONE FACTUAL statements that MUST be true for the response's logic to be valid.

Context:
{context}

Question:
{question}

Response:
{response}

The statements should be standalone and factual, not requiring any additional context to be understood (e.g. "Obesity increases heart disease chances" is standalone, but "The control group of the study had increased heart disease chances" is not).
For each statement, also provide an importance score between 0 and 1, where 1 means the statement is logically necessary for the response to be valid and 0 means it is not.
Your statements should be logically necessary for the response to be valid.
They should follow the following format:
INSERT STATEMENT 1 [0-1]
INSERT STATEMENT 2 [0-1]
...

Statement List:"""


def get_statement_validity_prompt(contexts: list[str], statement: str):
    context = "\n\n".join(contexts)

    return f"""Based on the following context, determine how valid the following statement is. Provide a reasoning and assign a validity score between 0 and 1, where 1 means completely valid and 0 means invalid.

Context:
{context}

Statement:
{statement}

Your last line should JUST be the validity score and nothing else, no periods, no text. Do not format your response, keep it as plaintext."""


def get_correction_prompt(
    question: str,
    contexts: list[str],
    response: str,
    statements: list[str],
    validity_judgement: str,
):
    context = "\n\n".join(contexts)
    return f"""Based on the following context, question, initial answer, and validity judgement, generate an answer that is accurate and supported by the context.

Context:
{context}

Question:
{question}

Initial Answer:
{response}

Statements:
{NEWLINE.join(statements)}

Validity Judgement:
{validity_judgement}

Please provide an answer, ensuring it is accurate and supported by the context. Do some reasoning if necessary.
Your last line should be the answer and nothing else, no periods, no text (yes, no, maybe).
Your new answer can be the same as the initial answer or different."""


async def summarize(namespace: str, prompt: str, progress: Progress | None = None):
    context = query_index(
        prompt, namespace, min_score=0.4, include_metadata=True  # type: ignore
    )

    response = await async_call_gpt(
        get_prompt(prompt, content_from_query_result(context))
    )

    validity, raw_statements = await async_gpt_calls(
        [
            get_validity_prompt(
                prompt, content_from_query_result(context), str(response)
            ),
            get_statement_list_prompt(
                prompt, content_from_query_result(context), str(response)
            ),
        ]
    )

    validity_float = float(get_number(str(validity).strip()))

    statements = [
        {
            "importance": float(
                get_number(statement.split("[")[1].replace("]", "").strip())
            ),
            "statement": statement.split("[")[0].strip(),
        }
        for statement in str(raw_statements).strip().split("\n")
        if "[" in statement
    ]

    statement_contexts = await multiple_queries(
        [s["statement"] for s in statements],
        namespace,
        min_score=0.65,
        include_metadata=True,
    )

    raw_statement_validities = await async_gpt_calls(
        [
            get_statement_validity_prompt(
                # content_from_query_result(statement["original_context"]) +
                content_from_query_result(contexts),
                statement["statement"],
            )
            for statement, contexts in zip(statements, statement_contexts)
        ],
    )

    statement_validities = [
        {
            "importance": statement["importance"],
            "statement": statement["statement"],
            "validity": float(get_number(str(validity).strip())),
            "validity_judgement": str(validity),
        }
        for validity, statement in zip(raw_statement_validities, statements)
    ]

    total_statement_validity = sum(
        [s["validity"] * s["importance"] for s in statement_validities]
    ) / (sum([s["importance"] for s in statement_validities]) or 1)

    result = {
        "question": prompt,
        "response": str(response),
        "validity": validity_float,
        "validity_judgement": str(validity),
        "statements": statement_validities,
    }

    if (validity_float * 0.6 + total_statement_validity * 0.4) < 0.7:
        correction = await async_call_gpt(
            get_correction_prompt(
                prompt,
                content_from_query_result(context),
                str(response),
                [
                    f"""{s["statement"]} [Validity: {s["validity"]}] [Importance: {s["importance"]}]"""
                    for s in statement_validities
                ],
                str(validity),
            )
        )
        result["correction"] = str(correction)

    result["decision"] = str(
        await async_call_gpt(
            f"""Please extract a one-word decision from the text that was answering this question (yes, no, maybe).
Question:
{prompt}

Answer:
{result["correction"] if "correction" in result else result["response"]}"""
        )
    )

    if progress:
        progress.increment()

    return result

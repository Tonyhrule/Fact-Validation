import re
from datasets import load_dataset, Dataset
from helpers.data import save_json
from helpers.oai import async_gpt_calls
import asyncio

from helpers.pc import content_from_query_result, multiple_queries

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


async def summarized():
    dataset = load_dataset("qiaojin/PubMedQA", name="pqa_labeled", split="train")

    data = dataset.select_columns(["question", "final_decision", "pubid"]).select(range(100))  # type: ignore

    if not isinstance(data, Dataset):
        raise TypeError("Expected a Dataset object")

    print("Querying contexts...")

    contexts = await multiple_queries(
        data["question"], "pubmed_summarized", min_score=0.55, include_metadata=True  # type: ignore
    )

    print("Running GPT...")

    responses = await async_gpt_calls(
        [
            get_prompt(question, content_from_query_result(context))
            for question, context in zip(data["question"], contexts)
        ],
    )

    result = [
        {
            "pubid": d["pubid"],  # type: ignore
            "decision": str(response),
            "correct_answer": d["final_decision"],  # type: ignore
        }
        for d, response in zip(list(data), responses)
    ]

    print("Checking general validities...")

    validities = await async_gpt_calls(
        [
            get_validity_prompt(
                question, content_from_query_result(context), str(response)
            )
            for question, context, response in zip(
                data["question"], contexts, responses
            )
        ],
    )

    print("Getting statement lists...")

    statement_lists = await async_gpt_calls(
        [
            get_statement_list_prompt(
                question, content_from_query_result(context), str(response)
            )
            for question, context, response in zip(
                data["question"], contexts, responses
            )
        ],
    )

    print("Fetching statement contexts...")

    statement_map = []

    for i, statement_list in enumerate(statement_lists):
        for line in str(statement_list).split("\n"):
            if line.strip():
                if "[" not in line:
                    continue
                statement = line.split("[")[0].strip()
                importance = line.split("[")[1].strip().replace("]", "")
                statement_map.append(
                    {
                        "index": i,
                        "query": statement,
                        "importance": float(importance),
                        "original_context": contexts[i],
                    }
                )

    statement_contexts = await multiple_queries(
        [s["query"] for s in statement_map],
        "pubmed_summarized",
        min_score=0.65,
        include_metadata=True,
    )

    print("Checking statement validities...")

    statement_validities = await async_gpt_calls(
        [
            get_statement_validity_prompt(
                content_from_query_result(s["original_context"])
                + content_from_query_result(contexts),
                s["query"],
            )
            for s, contexts in zip(statement_map, statement_contexts)
        ],
    )

    statement_validities_parsed = []

    for v, statement in zip(statement_validities, statement_map):
        validity = float(str(v).strip().split("\n")[-1])
        if len(statement_validities_parsed) <= statement["index"]:
            statement_validities_parsed.append([])
        statement_validities_parsed[statement["index"]].append(
            {
                "validity": validity,
                "importance": statement["importance"],
                "statement": statement["query"],
            }
        )

    save_json("temp/pubmed_summarized_validities.json", statement_validities_parsed)

    print("Making corrections...")

    validity_scores = []

    for v in validities:
        last_line = str(v).strip().split("\n")[-1]
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", last_line)
        score = float(nums[-1]) if nums else 0.0
        validity_scores.append(score)

    correction_prompts = []
    correction_map = []

    for i, score in enumerate(validity_scores):
        statement_validity = sum(
            [s["validity"] * s["importance"] for s in statement_validities_parsed[i]]
        ) / sum([s["importance"] for s in statement_validities_parsed[i]])

        if (score * 0.6 + statement_validity * 0.4) < 0.7:
            correction_map.append(i)
            correction_prompts.append(
                get_correction_prompt(
                    data["question"][i],
                    content_from_query_result(contexts[i]),
                    str(responses[i]),
                    [
                        f"""{s["statement"]} [Validity: {s["validity"]}] [Importance: {s["importance"]}]"""
                        for s in statement_validities_parsed[i]
                    ],
                    str(validities[i]),
                )
            )

    corrections = await async_gpt_calls(correction_prompts)

    for i, correction in zip(correction_map, corrections):
        result[i]["decision"] = str(correction)

    print("Checking results...")

    checks = await async_gpt_calls(
        [
            f"""Is the decision:
'{r['decision'].split(NEWLINE)[-1]}'
the same as the correct answer:
'{r['correct_answer']}'?
Respond with the word 'yes' or 'no'."""
            for r in result
        ],
        system="Respond with a single word.",
        max_tokens=10,
    )

    for r, check in zip(result, checks):
        r["correct"] = str(check).lower()[0] == "y"

    correct = sum(1 for r in result if r["correct"])

    print(f"Correct: {correct}/{len(result)} = {correct / len(result) * 100:.2f}%")

    save_json("results/pubmed_summarized.json", result)

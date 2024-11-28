from datasets import load_dataset, Dataset
from helpers.data import save_json
from helpers.oai import async_gpt_calls
from helpers.pc import content_from_query_result, multiple_queries
import re

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


async def raw():
    dataset = load_dataset("qiaojin/PubMedQA", name="pqa_labeled", split="train")

    data = dataset.select_columns(["question", "final_decision", "pubid"])  # type: ignore

    if not isinstance(data, Dataset):
        raise TypeError("Expected a Dataset object")

    print("Querying contexts...")
    contexts = await multiple_queries(
        data["question"], "pubmed_raw", min_score=0.55, include_metadata=True
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
            "correct": str(response).lower()[0] == d["final_decision"].lower()[0],  # type: ignore
        }
        for d, response in zip(list(data), responses)
    ]

    print("Checking validity...")

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
        if score < 0.8:
            correction_map.append(i)
            correction_prompts.append(
                get_correction_prompt(
                    data["question"][i],
                    content_from_query_result(contexts[i]),
                    str(responses[i]),
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
    save_json("results/pubmed_raw.json", result)

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

If the significance of a correlation is small, then there is no relationship.
Generally avoid maybe unless it is extremely close to being significant.
Be careful to answer the question asked.
Respond with only a yes, no, or maybe decision."""


async def raw():
    dataset = load_dataset("qiaojin/PubMedQA", name="pqa_labeled", split="train")

    data = dataset.select_columns(["question", "final_decision", "pubid"])  # type: ignore

    if not isinstance(data, Dataset):
        raise TypeError("Expected a Dataset object")

    print("Querying contexts...")

    contexts = await multiple_queries(
        data["question"], "pubmed_raw", min_score=0.4, include_metadata=True  # type: ignore
    )

    print("Running GPT...")

    responses = await async_gpt_calls(
        [
            get_prompt(question, content_from_query_result(context))
            for question, context in zip(data["question"], contexts)
        ],
    )

    result: list[dict[str, str]] = [
        {
            "pubid": d["pubid"],  # type: ignore
            "decision": str(response),
            "correct_answer": d["final_decision"],  # type: ignore
            "correct": str(response).lower()[0] == d["final_decision"].lower()[0],  # type: ignore
        }
        for d, response in zip(list(data), responses)
    ]

    # print("Checking results...")
    #
    #     checks = await async_gpt_calls(
    #         [
    #             f"""Is the decision:
    # '{r['decision'].split(NEWLINE)[-1]}'
    # the same as the correct answer:
    # '{r['correct_answer']}'?
    # Respond with the word 'yes' or 'no'."""
    #             for r in result
    #         ],
    #         system="Respond with a single word.",
    #         max_tokens=10,
    #     )

    #     for r, check in zip(result, checks):
    #         r["correct"] = str(check).lower()[0] == "y"  # type: ignore

    correct = sum(1 for r in result if r["correct"])

    print(f"Correct: {correct}/{len(result)} = {correct / len(result) * 100:.2f}%")

    save_json("results/pubmed_raw.json", result)

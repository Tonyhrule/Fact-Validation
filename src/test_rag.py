from datasets import load_dataset, Dataset
from helpers.data import save_json
from helpers.input import function_from_list
from helpers.oai import async_gpt_calls
from helpers.pc import multiple_queries, query_batches
import asyncio
from typing import TypedDict


class Data(TypedDict):
    ids: list[str]
    questions: list[str]


def get_dataset(dataset: Dataset, id_key="id", question_key="question"):
    return {"ids": dataset[id_key], "questions": dataset[question_key]}


namespaces = {
    "pubmed_raw": lambda: get_dataset(
        load_dataset("qiaojin/PubMedQA", name="pqa_labeled", split="train")
        .select_columns(["pubid", "question"])
        .select(range(100)),  # type: ignore
        "pubid",
    ),
    "pubmed_summarized": lambda: get_dataset(
        load_dataset(
            "qiaojin/PubMedQA", name="pqa_labeled", split="train"
        ).select_columns(
            ["pubid", "question"]
        ),  # type: ignore
        "pubid",
    ),
    "squad_raw": lambda: get_dataset(
        load_dataset("rajpurkar/squad", split="validation").select(range(1500)),  # type: ignore
    ),
    "squad_summarized": lambda: get_dataset(
        load_dataset("rajpurkar/squad", split="validation").select(range(1500)),  # type: ignore
    ),
    "hotpot_raw": lambda: get_dataset(
        load_dataset("hotpotqa/hotpot_qa", "fullwiki", split="train").select(range(150)),  # type: ignore
    ),
    "hotpot_summarized": lambda: get_dataset(
        load_dataset("hotpotqa/hotpot_qa", "fullwiki", split="train").select(range(150)),  # type: ignore
    ),
}


async def main():
    data: Data
    data, namespace = function_from_list("Please select a dataset:", namespaces)

    print("Querying contexts...")

    contexts = await multiple_queries(
        data["questions"], namespace, include_metadata=True, progress=True
    )

    print("Mapping contexts...")

    results = []

    for context, id, question, query in zip(
        contexts, data["ids"], data["questions"], data["questions"]
    ):
        matches = [x.metadata["ids"] for x in context]
        index = 9
        for i in range(10):
            if str(id) in matches[i]:
                index = i
                break
        results.append(
            {
                "id": id,
                "question": question,
                "query": query,
                "extra_matches": index,
                "result_k": context[index].score,
                "lowest_score": context[-1].score,
            }
        )

    correct = sum(1 for r in results if r["extra_matches"] == 0)

    print(f"Correct: {correct}/{len(results)} = {correct / len(results) * 100:.2f}%")

    one_extra = sum(1 for r in results if r["extra_matches"] == 1)

    print(
        f"One extra: {one_extra}/{len(results)} = {(one_extra) / (len(results)) * 100:.2f}%"
    )

    two_extra = sum(1 for r in results if r["extra_matches"] == 2)

    print(
        f"Two extra: {two_extra}/{len(results)} = {(two_extra) / (len(results)) * 100:.2f}%"
    )

    k_limit = 0.4

    above_k = sum(1 for r in results if r["result_k"] >= k_limit)

    print(
        f"Above k={k_limit}: {above_k}/{len(results)} = {above_k / len(results) * 100:.2f}%"
    )

    save_json(f"results/rag/{namespace}.json", results)


asyncio.run(main())

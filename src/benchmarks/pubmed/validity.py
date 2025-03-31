from datasets import load_dataset, Dataset
from helpers.data import chunk_list, save_json
from helpers.oai import async_gpt_calls
from helpers.progress import Progress
from pipelines.validity import run_validity
import asyncio


async def pubmed_validity():
    dataset = load_dataset("qiaojin/PubMedQA", name="pqa_labeled", split="train")

    data = dataset.select_columns(["question", "final_decision", "pubid"])

    if not isinstance(data, Dataset):
        raise TypeError("Expected a Dataset object")

    progress = Progress(len(data), "Benchmarking PubMed Validity")

    batches = chunk_list(data["question"], 50)

    results = []

    for batch in batches:
        results += await asyncio.gather(
            *[run_validity("pubmed_raw", prompt, progress) for prompt in batch]
        )

    progress.finish()

    decisions = await async_gpt_calls(
        [
            f"""Please extract a one-word decision from the text that was answering this question (yes, no, maybe).
                Question:
                {prompt}

                Answer:
                {result["correction"] if "correction" in result else result["response"]}"""
            for prompt, result in zip(data["question"], results)
        ]
    )

    for result, decision, correct in zip(results, decisions, data["final_decision"]):
        result["correct"] = str(decision).lower()[0] == correct.lower()[0]

    correct = sum(1 for r in results if r["correct"])

    print(f"Correct: {correct}/{len(results)} = {correct / len(results) * 100:.2f}%")

    save_json(
        "results/pubmed_validity.json",
        [
            {
                "pubid": pubid,
                "is_correct": result["correct"],
                "decision": decision,
                "correct_answer": correct_answer,
            }
            for result, correct_answer, pubid, decision in zip(
                results, data["final_decision"], data["pubid"], decisions
            )
        ],
    )

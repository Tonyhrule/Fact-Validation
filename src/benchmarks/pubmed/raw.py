from datasets import load_dataset, Dataset
from helpers.data import chunk_list, save_json
from helpers.progress import Progress
from pipelines.raw import run_raw
import asyncio


async def raw():
    dataset = load_dataset("qiaojin/PubMedQA", name="pqa_labeled", split="train")

    data = dataset.select_columns(["question", "final_decision", "pubid"])  # type: ignore

    if not isinstance(data, Dataset):
        raise TypeError("Expected a Dataset object")

    progress = Progress(len(data), "Benchmarking PubMed raw")

    batches = chunk_list(data["question"], 50)

    results = []

    for batch in batches:
        results += await asyncio.gather(
            *[run_raw("pubmed_raw", prompt, progress) for prompt in batch]
        )

    progress.finish()

    for result, correct in zip(results, data["final_decision"]):
        result["correct"] = result["decision"].lower()[0] == correct.lower()[0]

    correct = sum(1 for r in results if r["correct"])

    print(f"Correct: {correct}/{len(results)} = {correct / len(results) * 100:.2f}%")

    save_json(
        "results/pubmed_raw.json",
        [
            {
                "pubid": pubid,
                "is_correct": result["correct"],
                "decision": result["decision"],
                "correct_answer": correct_answer,
            }
            for result, correct_answer, pubid in zip(
                results, data["final_decision"], data["pubid"]
            )
        ],
    )

from datasets import load_dataset, Dataset
from helpers.data import chunk_list, save_json
from helpers.oai import async_gpt_calls
from helpers.progress import Progress
from pipelines.raw import run_raw
import asyncio


async def hotpot_raw():
    dataset = load_dataset("hotpotqa/hotpot_qa", "fullwiki", split="train")

    data = dataset.select_columns(["id", "question", "answer"]).select(range(1500))  # type: ignore

    if not isinstance(data, Dataset):
        raise TypeError("Expected a Dataset object")

    progress = Progress(len(data), "Benchmarking Hotpot Raw")

    batches = chunk_list(data["question"], 50)

    results = []

    for batch in batches:
        results += await asyncio.gather(
            *[run_raw("hotpot_raw", prompt, progress) for prompt in batch]
        )

    progress.finish()

    decisions = await async_gpt_calls(
        [
            f"""Please determine if these two answers to this question match (respond with yes or no):
Question:
{prompt}

Answer 1:
{result["correction"] if "correction" in result else result["response"]}

Answer 2:
{answer}"""
            for prompt, result, answer in zip(data["question"], results, data["answer"])
        ],
        max_tokens=10,
        system="Your answer must be a single word long, either yes or no.",
    )

    for result, decision in zip(results, decisions):
        result["correct"] = str(decision).lower()[0] == "y"

    correct = sum(1 for r in results if r["correct"])

    print(f"Correct: {correct}/{len(results)} = {correct / len(results) * 100:.2f}%")

    save_json(
        "results/hotpot_raw.json",
        [
            {
                "id": id,
                "is_correct": result["correct"],
                "answer": (
                    result["correction"]
                    if "correction" in result
                    else result["response"]
                ),
                "correct_answer": correct_answer,
            }
            for result, correct_answer, id in zip(results, data["answer"], data["id"])
        ],
    )

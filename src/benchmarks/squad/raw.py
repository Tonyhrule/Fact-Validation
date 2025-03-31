from functools import partial
from datasets import load_dataset, Dataset
from helpers.data import queue, save_json
from helpers.oai import async_gpt_calls
from helpers.progress import Progress
from pipelines.raw import run_raw
import asyncio


async def squad_raw():
    dataset = load_dataset("rajpurkar/squad", split="validation")

    data = dataset.select_columns(["id", "context", "question", "answers", "title"]).filter(lambda x: x["title"] != "Super_Bowl_50").select(range(1500))  # type: ignore

    if not isinstance(data, Dataset):
        raise TypeError("Expected a Dataset object")

    progress = Progress(len(data), "Benchmarking Squad Raw")

    results = await queue(
        [
            partial(run_raw, "squad_raw", prompt, progress)
            for prompt in data["question"]
        ],
    )

    progress.finish()

    decisions = await async_gpt_calls(
        [
            f"""Please determine if the answer is correct (respond with yes or no).
            Accept partial/similar answers (eg. markedly improved = significantly improved)
            If the answer is correct in any part of its explanation, it is correct.
            Question:
            {prompt}

            Correct Answer:
            {answer["text"][0]}

            Provided Answer:
            {result["correction"] if "correction" in result else result["response"]}"""
            for prompt, result, answer in zip(
                data["question"], results, data["answers"]
            )
        ],
        max_tokens=10,
        system="Your answer must be a single word long, either yes or no.",
        progress_bar=True,
    )

    decisions_final_answer = await async_gpt_calls(
        [
            f"""Please determine if the answer is correct (respond with yes or no).
            Accept partial/similar answers (eg. markedly improved = significantly improved)
            If the answer is correct in any part of its explanation, it is correct.
            Question:
            {prompt}

            Correct Answer:
            {answer["text"][0]}

            Provided Answer:
            {(result["correction"] if "correction" in result else result["response"]).lower().strip().split("final answer: ")[-1]}"""
            for prompt, result, answer in zip(
                data["question"], results, data["answers"]
            )
        ],
        max_tokens=10,
        system="Your answer must be a single word long, either yes or no.",
        progress_bar=True,
    )

    for result, decision, decision_final_answer in zip(
        results, decisions, decisions_final_answer
    ):
        result["correct"] = (
            str(decision).lower()[0] == "y"
            or str(decision_final_answer).lower()[0] == "y"
        )

    correct = sum(1 for r in results if r["correct"])

    print(f"Correct: {correct}/{len(results)} = {correct / len(results) * 100:.2f}%")

    save_json(
        "results/squad_raw.json",
        [
            {
                "id": id,
                "question": question,
                "is_correct": result["correct"],
                "answer": (
                    result["correction"]
                    if "correction" in result
                    else result["response"]
                ),
                "correct_answer": correct_answer["text"][0],
            }
            for result, correct_answer, id, question in zip(
                results, data["answers"], data["id"], data["question"]
            )
        ],
    )

from datasets import load_dataset, Dataset
from helpers.data import save_json
from helpers.pc import multiple_queries
import asyncio


dataset = load_dataset("hotpotqa/hotpot_qa", "fullwiki", split="train")
data = dataset.select_columns(["question", "id"]).select(range(1500))  # type: ignore

if not isinstance(data, Dataset):
    raise TypeError("Expected a Dataset object")

print("Querying contexts...")

contexts = asyncio.run(
    multiple_queries(data["question"], "hotpot_summarized", include_metadata=True)
)

print("Mapping contexts...")

results = []

for context, id in zip(contexts, data["id"]):
    matches = [x.metadata["ids"] for x in context.matches]
    index = 9
    for i in range(10):
        if id in matches[i]:
            index = i
            break
    results.append(
        {
            "pubid": id,
            "missing_matches": index,
            "lowest_k": context.matches[index]["score"],
        }
    )

correct = sum(1 for r in results if r["missing_matches"] == 0)

print(f"Correct: {correct}/{len(results)} = {correct / len(results) * 100:.2f}%")

one_missing = sum(1 for r in results if r["missing_matches"] <= 1)

print(
    f"One missing: {one_missing}/{len(results)} = {one_missing / len(results) * 100:.2f}%"
)

two_missing = sum(1 for r in results if r["missing_matches"] <= 2)

print(
    f"Two missing: {two_missing}/{len(results)} = {two_missing / len(results) * 100:.2f}%"
)

above_55 = sum(1 for r in results if r["lowest_k"] >= 0.55)

print(f"Above k=.55: {above_55}/{len(results)} = {above_55 / len(results) * 100:.2f}%")

save_json("results/hotpot_raw_rag.json", results)

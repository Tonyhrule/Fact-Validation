from datasets import load_dataset, Dataset
from helpers.data import save_json
from helpers.pc import multiple_queries
import asyncio


dataset = load_dataset("qiaojin/PubMedQA", name="pqa_labeled", split="train")
data = dataset.select_columns(["question", "pubid"])  # type: ignore

if not isinstance(data, Dataset):
    raise TypeError("Expected a Dataset object")

print("Querying contexts...")

contexts = asyncio.run(multiple_queries(data["question"], "pubmed_summarized"))

print("Mapping contexts...")

results = []

for context, pubid in zip(contexts, data["pubid"]):
    matches = [x["id"] for x in context.matches]
    index = 9
    for i in range(10):
        if str(pubid) in matches[i]:
            index = i
            break
    print(pubid, matches, index)
    results.append(
        {
            "pubid": pubid,
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

save_json("results/pubmed_raw_rag.json", results)

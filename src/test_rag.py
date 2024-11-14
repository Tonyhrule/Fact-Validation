from datasets import load_dataset, Dataset
from helpers.data import read_json, save_json
from helpers.pc import multiple_queries
import asyncio


dataset = load_dataset("qiaojin/PubMedQA", name="pqa_labeled", split="train")
data = dataset.select_columns(["question", "pubid"])  # type: ignore

if not isinstance(data, Dataset):
    raise TypeError("Expected a Dataset object")

print("Querying contexts...")

contexts = asyncio.run(multiple_queries(data["question"], "pubmed_raw"))

context_map = read_json("data/pubmed_raw_context_map.json")

print("Mapping contexts...")

results = []

for context, pubid in zip(contexts, data["pubid"]):
    context_ids = set(context_map[str(pubid)])
    context = context["matches"]
    missing_matches = len(context_ids)

    for i in range(len(context_ids)):
        if context[i]["id"] in context_ids:
            missing_matches -= 1

    results.append(
        {
            "pubid": pubid,
            "missing_matches": missing_matches,
            "lowest_k": (
                context[len(context_ids) - 1 + missing_matches]["score"]
                if len(context) > len(context_ids) - 1 + missing_matches
                else 0
            ),
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

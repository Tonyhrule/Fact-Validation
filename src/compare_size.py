import asyncio
from math import floor
from helpers.pc import get_namespace_size, query_index

datasets = ["pubmed", "squad", "hotpot"]

print("Choose a dataset:")
for i, dataset in enumerate(datasets):
    print(f"{i + 1}. {dataset}")

dataset = datasets[int(input()) - 1]

raw = asyncio.run(
    query_index("a", dataset + "_raw", top_k=10_000, include_metadata=True)
)
summarized = asyncio.run(
    query_index("a", dataset + "_summarized", top_k=10_000, include_metadata=True)
)

raw_size = sum(len(str(x.metadata["content"])) for x in raw)
summarized_size = sum(len(str(x.metadata["content"])) for x in summarized)

raw_count = get_namespace_size(dataset + "_raw")
summarized_count = get_namespace_size(dataset + "_summarized")

raw_size = raw_size * len(raw) / raw_count
summarized_size = summarized_size * len(summarized) / summarized_count

print(f"Raw size: {floor(raw_size)}")
print(f"Summarized size: {floor(summarized_size)}")
print(f"Ratio: {summarized_size / raw_size * 100 :.2f}%")

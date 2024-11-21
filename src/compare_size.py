from helpers.data import stringify
from helpers.pc import query_index

raw = query_index("a", "pubmed_raw", top_k=10000, include_metadata=True)
summarized = query_index("a", "pubmed_summarized", top_k=10000, include_metadata=True)

raw_size = sum(len(stringify(x.metadata["content"])) for x in raw.matches)
summarized_size = sum(len(stringify(x.metadata["content"])) for x in summarized.matches)

print(f"Raw size: {raw_size}")
print(f"Summarized size: {summarized_size}")
print(f"Ratio: {summarized_size / raw_size * 100 :.2f}%")

import asyncio
from helpers.data import stringify
from helpers.pc import query_index

raw = asyncio.run(query_index("a", "squad_raw", top_k=10_000, include_metadata=True))
summarized = asyncio.run(
    query_index("a", "squad_summarized", top_k=10_000, include_metadata=True)
)

raw_size = sum(len(stringify(x.metadata["content"])) for x in raw)
summarized_size = sum(len(stringify(x.metadata["content"])) for x in summarized)

print(f"Raw size: {raw_size}")
print(f"Summarized size: {summarized_size}")
print(f"Ratio: {summarized_size / raw_size * 100 :.2f}%")

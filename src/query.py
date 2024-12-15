import asyncio
from helpers.pc import query_index


namespaces = [
    "pubmed_raw",
    "pubmed_summarized",
    "squad_raw",
    "squad_summarized",
    "hotpot_raw",
    "hotpot_summarized",
]
namespace_str = "\n".join(
    [f"{i + 1}. {namespace}" for i, namespace in enumerate(namespaces)]
)

namespace_response = input(
    f"Which namespace would you like to use?\n{namespace_str}\n\n"
)

if namespace_response not in [str(i + 1) for i in range(len(namespaces))]:
    raise ValueError("Invalid namespace")

namespace = namespaces[int(namespace_response) - 1]

print(f"Using namespace: {namespace}")

query = input("Enter query: ")

print(
    [
        x.metadata["content"]
        for x in asyncio.run(query_index(query, namespace, include_metadata=True))
    ]
)

import asyncio
from datasets import load_dataset, Dataset

from helpers.dbscan import cluster
from helpers.oai import async_gpt_calls, get_embeddings
from helpers.pc import upsert_index

NEWLINES = "\n\n"


def get_summarize_prompt(context: str):
    return f"""Give an EXTREMELY short paragraph summary of the text with the following rules:
Each sentence should ONLY embody a relationship shown in the paragraph (have no other sentences).
In each sentence, say how statistically significant each relationship is (qualitatively, not a number).
  Eg. "RELATIONSHIP_STATEMENT (SIGNIFICANCE_AMOUNT)"
Your significance amounts can be "significant", "almost significant", and "not significant".
The relationship needs a strong degree of significance to be considered significant, and avoid saying almost significant unless it's very close to significant.
Your sentences should not have statistics or evidence in them, they should only state the relationship.
Your sentences should be EXTREMELY simple and direct.
Add a short sentence of context about the terms and all acronyms used in your summary as well as general information about the text at the start (a complete outsider should understand what you're talking about).

Text:
{context}"""


def get_compress_prompt(cluster: list[str]):
    return f"""Combine the following paragraphs into an EXTREMELY short paragraph summary with the following rules:
If there is conflicting information, pick the information that is more valid/more prevalent.
Retain the sentence structures of RELATIONSHIP_STATEMENT (SIGNIFICANCE_AMOUNT).

Paragraphs:
{NEWLINES.join(cluster)}"""


async def pubmed_summarize():
    dataset = load_dataset("qiaojin/PubMedQA", name="pqa_labeled", split="train")

    data = dataset.select_columns(["question", "context", "pubid"])

    if not isinstance(data, Dataset):
        raise TypeError("Expected a Dataset object")

    ids = [str(id) for id in data["pubid"]]

    raw_contexts = []

    for context_obj in data["context"]:
        raw_contexts.append(" ".join(context_obj["contexts"]))

    print("Getting summaries...")

    new_contexts = [
        str(x)
        for x in await async_gpt_calls(
            [get_summarize_prompt(context) for context in raw_contexts]
        )
    ]

    new_context_map = {id: context for id, context in zip(ids, new_contexts)}

    print("Getting embeddings...")

    embeddings = await get_embeddings(new_contexts)

    embedding_map = {id: embedding for id, embedding in zip(ids, embeddings)}

    print("Clustering embeddings...")

    clusters = cluster(
        [
            {"vector": embedding.vector, "id": id}
            for embedding, id in zip(embeddings, ids)
        ]
    )

    print("Compressing clusters...")

    no_compress = [cluster[0] for cluster in clusters if len(cluster) == 1]
    to_compress = [cluster for cluster in clusters if len(cluster) > 1]

    compressions = [
        str(x)
        for x in await async_gpt_calls(
            [
                get_compress_prompt([new_context_map[id] for id in cluster])
                for cluster in to_compress
            ]
        )
    ]

    print("Updating embeddings...")

    compression_embeddings = await get_embeddings(compressions)

    print("Upserting embeddings...")

    final_contexts = [
        {
            "id": id,
            "values": embedding_map[id].vector,
            "metadata": {
                "content": new_context_map[id],
            },
        }
        for id in no_compress
    ]

    final_contexts += [
        {
            "id": "-".join(ids),
            "values": embedding.vector,
            "metadata": {
                "content": content,
            },
        }
        for embedding, ids, content in zip(
            compression_embeddings, to_compress, compressions
        )
    ]

    upsert_index("pubmed_summarized", final_contexts)

    print("Done")

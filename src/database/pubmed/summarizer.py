import asyncio
from uuid import uuid4
from datasets import load_dataset, Dataset

from helpers.data import save_json
from helpers.oai import async_gpt_calls, get_embeddings
from helpers.pc import upsert_index


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


async def pubmed_summarize():
    dataset = load_dataset("qiaojin/PubMedQA", name="pqa_labeled", split="train")

    data = dataset.select_columns(["question", "context", "pubid"])

    if not isinstance(data, Dataset):
        raise TypeError("Expected a Dataset object")

    raw_contexts = []
    context_to_UUID = {}

    for context_obj in data["context"]:
        for context in context_obj["contexts"]:
            if context not in raw_contexts:
                context_to_UUID[context] = str(uuid4())
                raw_contexts.append(context)

    print("Getting summaries...")

    new_contexts = list(
        map(str, await async_gpt_calls(list(map(get_summarize_prompt, raw_contexts))))
    )

    print("Getting embeddings...")

    embeddings = await get_embeddings(new_contexts)

    pubid_to_context_uuids = {}

    for item in list(data):
        pubid = str(item["pubid"])  # type: ignore
        context_uuids = [context_to_UUID[context] for context in item["context"]["contexts"]]  # type: ignore
        pubid_to_context_uuids[pubid] = context_uuids

    print("Upserting embeddings...")

    upsert_index(
        "pubmed_summarized",
        [
            {
                "id": context_to_UUID[context],
                "values": embedding.vector,
                "metadata": {
                    "content": new_context,
                },
            }
            for embedding, context, new_context in zip(
                embeddings, raw_contexts, new_contexts
            )
        ],
    )

    save_json("data/pubmed_summarized_context_map.json", pubid_to_context_uuids)

    print("Done")

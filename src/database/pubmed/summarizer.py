import asyncio
from uuid import uuid4
from datasets import load_dataset, Dataset

from helpers.dbscan import cluster
from helpers.oai import async_gpt_calls, get_embeddings
from helpers.pc import upsert_index


def get_statement_prompt(context: str):
    return f"""Convert the following text into a series of concise, standalone statements with the following rules:
This should be in bullet-points (- ).
Each statement should be a complete thought and should NOT reference any other bullet points or wider context.
This means that each statement should be able to stand alone and make sense.
For a statement to stand alone, it should contain all the context it needs to be understood.
Each statement MUST include its FULL setting (eg. In xyz, abc happened), and this setting must be specific (eg. Superbowl 50 instead of Superbowl).
ALWAYS use proper nouns if possible. (eg. "Superbowl 50" instead of "the game")
Each bullet point should ONLY embody a relationship shown in the paragraph.
In each bullet, say how statistically significant each relationship is (qualitatively, not a number).
  Eg. "RELATIONSHIP_STATEMENT (SIGNIFICANCE_AMOUNT)"
Your significance amounts can be "significant", "almost significant", and "not significant".
The relationship needs a strong degree of significance to be considered significant, and avoid saying almost significant unless it's very close to significant.
Your statements should not have statistics or evidence in them, they should only state the relationship.
Your statements should be EXTREMELY simple and direct.

Text:
{context}"""


# def get_summarize_prompt(context: str):
#     return f"""Give an EXTREMELY short paragraph summary of the text with the following rules:
# Each sentence should ONLY embody a relationship shown in the paragraph (have no other sentences).
# In each sentence, say how statistically significant each relationship is (qualitatively, not a number).
#   Eg. "RELATIONSHIP_STATEMENT (SIGNIFICANCE_AMOUNT)"
# Your significance amounts can be "significant", "almost significant", and "not significant".
# The relationship needs a strong degree of significance to be considered significant, and avoid saying almost significant unless it's very close to significant.
# Your sentences should not have statistics or evidence in them, they should only state the relationship.
# Your sentences should be EXTREMELY simple and direct.
# Add a short sentence of context about the terms and all acronyms used in your summary as well as general information about the text at the start (a complete outsider should understand what you're talking about).

# Text:
# {context}"""


def get_compress_prompt(cluster: list[str]):
    statement_list = "\n".join(cluster)
    return f"""Combine the following statements into ONE statement with the following rules:
If there is conflicting information, pick the information that is more valid/more prevalent.
Retain the statement structures of RELATIONSHIP_STATEMENT (SIGNIFICANCE_AMOUNT).

Statements:
{statement_list}"""


async def pubmed_summarize():
    dataset = load_dataset("qiaojin/PubMedQA", name="pqa_labeled", split="train")

    data = dataset.select_columns(["question", "context", "pubid"])

    if not isinstance(data, Dataset):
        raise TypeError("Expected a Dataset object")

    context_to_ids: dict[str, list[str]] = {}
    raw_contexts: list[str] = []

    for id, context_obj in zip(data["pubid"], data["context"]):
        text = " ".join(context_obj["contexts"])
        if (text) not in context_to_ids:
            context_to_ids[text] = []
            raw_contexts.append(text)
        context_to_ids[text].append(str(id))

    print("Getting statements...")

    new_contexts = [
        str(x).replace("\n- ", "\n").strip().removeprefix("- ")
        for x in await async_gpt_calls(
            [get_statement_prompt(context) for context in raw_contexts],
            progress_bar=True,
        )
    ]

    statements = [
        {
            "statement": statement.strip(),
            "id": str(uuid4()),
            "questions": context_to_ids[context],
        }
        for statement_list, context in zip(new_contexts, raw_contexts)
        for statement in statement_list.split("\n")
        if statement.strip() != ""
    ]

    statement_id_to_statement = {statement["id"]: statement for statement in statements}

    print("Getting embeddings...")

    embeddings = await get_embeddings(
        [statement["statement"] for statement in statements], progress_bar=True
    )

    for statement, embedding in zip(statements, embeddings):
        statement["vector"] = embedding.vector

    print("Clustering embeddings...")

    clusters = cluster(
        [
            {"vector": embedding.vector, "id": statement["id"]}
            for embedding, statement in zip(embeddings, statements)
        ]
    )

    print("Compressing clusters...")

    no_compress = [cluster[0] for cluster in clusters if len(cluster) == 1]
    to_compress = [cluster for cluster in clusters if len(cluster) > 1]

    compressions = [
        str(x)
        for x in await async_gpt_calls(
            [
                get_compress_prompt(
                    [
                        statement_id_to_statement[statement_id]["statement"]
                        for statement_id in cluster
                    ]
                )
                for cluster in to_compress
            ],
            progress_bar=True,
        )
    ]

    print("Updating embeddings...")

    compression_embeddings = await get_embeddings(compressions, progress_bar=True)

    print("Upserting embeddings...")

    final_contexts = [
        {
            "id": statement_id_to_statement[statement_id]["id"],
            "values": statement_id_to_statement[statement_id]["vector"],
            "metadata": {
                "content": statement_id_to_statement[statement_id]["statement"],
                "ids": statement_id_to_statement[statement_id]["questions"],
            },
        }
        for statement_id in no_compress
    ]

    final_contexts += [
        {
            "id": str(uuid4()),
            "values": embedding.vector,
            "metadata": {
                "content": content,
                "ids": list(
                    set(
                        [
                            question
                            for statement_id in statement_ids
                            for question in statement_id_to_statement[statement_id][
                                "questions"
                            ]
                        ]
                    )
                ),
            },
        }
        for embedding, statement_ids, content in zip(
            compression_embeddings, to_compress, compressions
        )
    ]

    upsert_index("pubmed_summarized", final_contexts)

    print("Done")

import re
from datasets import load_dataset, Dataset
from helpers.data import save_json
from helpers.oai import async_gpt_calls
import asyncio
from helpers.pc import content_from_query_result, multiple_queries

NEWLINE = "\n"


def get_prompt(question: str, contexts: list[str]):
    context = "\n\n".join(contexts)

    return f"""Given the following text:
{context}

Answer the following question:
{question}

If a correlation is insignificant, then there is no relationship.
Generally avoid maybe unless it is almost significant.
Be careful to answer the question asked.
Respond with only a yes, no, or maybe decision."""


def get_statement_list_prompt(question: str, response: str):
    return f"""Given the following question and response, decompose the response into standalone atomic factual statements.
Each statement should be standalone, factual, and relevant to the question.
For each statement, assign an importance score between 0 and 1, where 1 means highly important to the overall response and 0 means not important.

Question:
{question}

Response:
{response}

Output format:
Statement 1 [Importance: 0-1]
Statement 2 [Importance: 0-1]
...
"""


def get_statement_validity_prompt(statement: str, contexts: list[str]):
    context_text = "\n\n".join(contexts)
    return f"""Evaluate the validity of the following statement based on the given contexts.

Statement:
{statement}

Contexts:
{context_text}

Instructions:
- Provide a validity score between 0 and 1, where 1 means completely valid and 0 means invalid.
- Provide a brief reasoning for your score.

Output format:
Validity Score: <score between 0 and 1>
Reasoning: <brief reasoning>
"""


def generate_explanation_prompt(statement: str, contexts: list[str], validity_score: float, reasoning: str):
    context_text = "\n\n".join(contexts) if contexts else "No relevant contexts found."
    return f"""Generate a detailed explanation for the validity decision of the following statement based on the provided contexts.

Statement:
{statement}

Contexts:
{context_text}

Validity Score: {validity_score}
Reasoning: {reasoning}

Instructions:
- Use the contexts to support your explanation.
- Highlight any contradictions, gaps, or supporting facts.
- Explain why the validity score was assigned based on the reasoning.
- Make sure your explanations are concise and should be 1-2 sentences, just emphasize the most important points.

Output format:
Explanation: <detailed explanation>
"""


def get_correction_prompt(
    question: str,
    contexts: list[str],
    response: str,
    statements: list[str],
    validity_judgement: str,
):
    context = "\n\n".join(contexts)
    return f"""Based on the following context, question, initial answer, and statement evaluations, generate an answer that is accurate and supported by the context.

Context:
{context}

Question:
{question}

Initial Answer:
{response}

Statements and Validity:
{NEWLINE.join(statements)}

Please provide an answer, ensuring it is accurate and supported by the context. Do some reasoning if necessary.
Your last line should be the answer and nothing else, no periods, no text (yes, no, maybe).
Your new answer can be the same as the initial answer or different."""


async def new():
    dataset = load_dataset("qiaojin/PubMedQA", name="pqa_labeled", split="train")

    data = dataset.select_columns(["question", "final_decision", "pubid"]).select(range(100))  # type: ignore

    if not isinstance(data, Dataset):
        raise TypeError("Expected a Dataset object")

    print("Querying contexts for initial responses...")

    # For initial responses, we can use the question to retrieve contexts
    initial_contexts = await multiple_queries(
        data["question"], "pubmed_summarized", min_score=0.55, include_metadata=True  # type: ignore
    )

    print("Generating initial responses...")

    initial_responses = await async_gpt_calls(
        [
            get_prompt(question, content_from_query_result(context))
            for question, context in zip(data["question"], initial_contexts)
        ],
    )

    result = [
        {
            "pubid": d["pubid"],  # type: ignore
            "decision": str(response).strip(),
            "correct_answer": d["final_decision"],  # type: ignore
        }
        for d, response in zip(list(data), initial_responses)
    ]

    print("Decomposing responses into atomic statements...")

    # Decompose each response into atomic statements
    statement_lists = await async_gpt_calls(
        [
            get_statement_list_prompt(question, str(response))
            for question, response in zip(data["question"], initial_responses)
        ],
    )

    # Parse the statements and their importance scores
    atomic_statements = []
    for i, statement_list in enumerate(statement_lists):
        statements = []
        for line in str(statement_list).split("\n"):
            if line.strip():
                if "[" not in line:
                    continue
                statement = line.split("[")[0].strip()
                importance = line.split("[")[1].strip("]").replace("Importance: ", "").strip()
                try:
                    importance_score = float(importance)
                except ValueError:
                    importance_score = 0.0
                statements.append({
                    "statement": statement,
                    "importance": importance_score,
                    "index": i,
                })
        atomic_statements.append(statements)

    print("Retrieving contexts for atomic statements...")

    # For each statement, retrieve primary and expanded contexts
    all_statements = []
    for statements in atomic_statements:
        all_statements.extend(statements)

    statements_texts = [s["statement"] for s in all_statements]

    primary_contexts = await multiple_queries(
        statements_texts, "pubmed_summarized", min_score=0.65, include_metadata=True  # type: ignore
    )

    expanded_contexts = await multiple_queries(
        statements_texts, "pubmed_summarized", min_score=0.55, include_metadata=True  # type: ignore
    )

    # Assign contexts back to statements
    for s, p_context, e_context in zip(all_statements, primary_contexts, expanded_contexts):
        s["primary_contexts"] = content_from_query_result(p_context)
        s["expanded_contexts"] = content_from_query_result(e_context)

    print("Validating atomic statements...")

    # Generate prompts for validating statements
    validation_prompts = [
        get_statement_validity_prompt(
            s["statement"], s["primary_contexts"] + s["expanded_contexts"]
        )
        for s in all_statements
    ]

    validation_results = await async_gpt_calls(validation_prompts)

    # Parse validation results
    for s, validation in zip(all_statements, validation_results):
        raw_content = str(validation)
        validity_score = 0.0
        reasoning = ""
        for line in raw_content.strip().split("\n"):
            if line.startswith("Validity Score:"):
                try:
                    validity_score = float(line.split(":", 1)[1].strip())
                except ValueError:
                    validity_score = 0.0
            elif line.startswith("Reasoning:"):
                reasoning = line.split(":", 1)[1].strip()
        s["validity_score"] = validity_score
        s["reasoning"] = reasoning

    print("Generating explanations for validation decisions...")

    # Generate explanations
    explanation_prompts = [
        generate_explanation_prompt(
            s["statement"],
            s["primary_contexts"] + s["expanded_contexts"],
            s["validity_score"],
            s["reasoning"]
        )
        for s in all_statements
    ]

    explanations = await async_gpt_calls(explanation_prompts)

    for s, explanation in zip(all_statements, explanations):
        s["explanation"] = str(explanation)

    print("Aggregating validity scores for responses...")

    # Aggregate validity scores per response
    for i, statements in enumerate(atomic_statements):
        total_importance = sum(s["importance"] for s in statements)
        if total_importance == 0:
            total_importance = 1  # Avoid division by zero
        weighted_validity = sum(s["validity_score"] * s["importance"] for s in statements) / total_importance
        result[i]["aggregated_validity"] = weighted_validity

    # Decide whether to correct the response based on the aggregated validity score
    print("Generating corrected responses where necessary...")

    corrected_indices = []
    correction_prompts = []

    for i, res in enumerate(result):
        if res["aggregated_validity"] < 0.7:
            corrected_indices.append(i)
            # Prepare statements with their validity scores and importance
            statements_info = atomic_statements[i]
            statements_text = [
                f"""{s["statement"]} [Validity Score: {s["validity_score"]}] [Importance: {s["importance"]}]"""
                for s in statements_info
            ]
            correction_prompts.append(
                get_correction_prompt(
                    data["question"][i],
                    content_from_query_result(initial_contexts[i]),
                    str(initial_responses[i]),
                    statements_text,
                    "N/A",
                )
            )

    # Generate corrected responses
    if correction_prompts:
        corrections = await async_gpt_calls(correction_prompts)
        for idx, correction in zip(corrected_indices, corrections):
            result[idx]["decision"] = str(correction).strip()

    print("Checking results...")

    # Check if final decisions match the correct answers
    checks = await async_gpt_calls(
        [
            f"""Is the decision:
'{r['decision'].split(NEWLINE)[-1]}'
the same as the correct answer:
'{r['correct_answer']}'?
Respond with the word 'yes' or 'no'."""
            for r in result
        ],
        system="Respond with a single word.",
        max_tokens=10,
    )

    for r, check in zip(result, checks):
        r["correct"] = str(check).lower().strip().startswith("yes")

    correct = sum(1 for r in result if r["correct"])

    print(f"Correct: {correct}/{len(result)} = {correct / len(result) * 100:.2f}%")

    save_json("results/pubmed_new.json", result)

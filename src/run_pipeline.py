from helpers.data import save_json
from pipelines.summarizer import summarize
import asyncio


result = asyncio.run(
    summarize(
        "pubmed_summarized",
        "Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?",
    )
)

save_json("temp/pubmed_summarized.json", result)

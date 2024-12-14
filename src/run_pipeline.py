from helpers.data import save_json
from pipelines.raw import run_raw
from pipelines.validity import run_validity
import asyncio


result = asyncio.run(
    run_raw(
        "pubmed_summarized",
        "Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?",
    )
)

save_json("temp/pubmed_summarized.json", result)

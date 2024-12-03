from benchmarks.pubmed.raw import pubmed_raw
from benchmarks.pubmed.new import new
from benchmarks.pubmed.summarizer import pubmed_summarized
from benchmarks.squad.raw import squad_raw
from benchmarks.squad.summarizer import squad_summarized
from helpers.input import function_from_list

programs = [
    ("PubMed QA: Raw Batch", pubmed_raw),
    ("PubMed QA: Summarized Batch", pubmed_summarized),
    ("PubMed QA: New Pipeline", new),
    ("Squad: Raw Batch", squad_raw),
    ("Squad: Summarized Batch", squad_summarized),
]

if __name__ == "__main__":
    function_from_list("What benchmark would you like to run?", programs)

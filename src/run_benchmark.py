from benchmarks.pubmed.raw import raw
from benchmarks.pubmed.new import new
from benchmarks.pubmed.summarizer import summarized
from helpers.input import function_from_list

programs = [("PubMed QA: Raw Batch", raw), ("PubMed QA: Summarized Batch", summarized), ("PubMed QA: New Pipeline", new)]

if __name__ == "__main__":
    function_from_list("What benchmark would you like to run?", programs)

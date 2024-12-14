from benchmarks.hotpot.summarizer import hotpot_summarized
from benchmarks.hotpot.validity import hotpot_validity
from benchmarks.pubmed.raw import pubmed_raw
from benchmarks.pubmed.summarizer import pubmed_summarized
from benchmarks.pubmed.validity import pubmed_validity
from benchmarks.squad.raw import squad_raw
from benchmarks.squad.summarizer import squad_summarized
from benchmarks.hotpot.raw import hotpot_raw
from benchmarks.squad.validity import squad_validity
from helpers.input import function_from_list

programs = {
    "PubMed QA: Raw Batch": pubmed_raw,
    "PubMed QA: Summarized Batch": pubmed_summarized,
    "Pubmed QA: Validity Batch": pubmed_validity,
    "Squad: Raw Batch": squad_raw,
    "Squad: Summarized Batch": squad_summarized,
    "Squad: Validity Batch": squad_validity,
    "Hotpot QA: Raw Batch": hotpot_raw,
    "Hotpot QA: Summarized Batch": hotpot_summarized,
    "Hotpot QA: Validity Batch": hotpot_validity,
}

if __name__ == "__main__":
    function_from_list("What benchmark would you like to run?", programs)

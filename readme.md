# Factual Context Validation and Simplification: A Scalable Method to Enhance GPT Trustworthiness and Efficiency

> **ICLR 2025 Blogpost Track Publication**  
> This work has been accepted at the ICLR 2025 Blogpost Track. [View the full blog](https://d2jud02ci9yv69.cloudfront.net/2025-04-28-factual-validation-simplification-192/blog/factual-validation-simplification/).

## Abstract

As the deployment of Large Language Models (LLMs) like GPT expands across domains, mitigating their susceptibility to factual inaccuracies or hallucinations becomes crucial for ensuring reliable performance. This blog post introduces two novel frameworks that enhance retrieval-augmented generation (RAG): one uses summarization to achieve a maximum of 57.7% storage reduction, while the other preserves critical information through statement-level extraction. Leveraging DBSCAN clustering, vectorized fact storage, and LLM-driven fact-checking, the pipelines deliver higher overall performance across benchmarks such as PubMedQA, SQuAD, and HotpotQA. By optimizing efficiency and accuracy, these frameworks advance trustworthy AI for impactful real-world applications.

## Overview

This project provides a framework for:
- Benchmarking and evaluating the effectiveness of different techniques (raw, summarized, validity-checking)
- Comparing performance metrics across different benchmarks
- Optimizing storage efficiency through context summarization
- Enhancing factual validation through statement-level extraction

The system currently supports the following datasets:
- **HotpotQA**: A dataset requiring multi-hop reasoning across multiple documents
- **PubMedQA**: A dataset containing biomedical questions with yes/no/maybe answers
- **SQuAD**: Stanford Question Answering Dataset with factoid-based questions

## Project Structure

```
src/
├── benchmarks/                  # Benchmark implementation modules
│   ├── hotpot/                 # HotpotQA dataset benchmarks
│   │   ├── raw.py              # Basic pipeline benchmark
│   │   ├── summarizer.py       # Summarization pipeline benchmark
│   │   └── validity.py         # Validity checking benchmark
│   ├── pubmed/                 # PubMedQA dataset benchmarks
│   │   ├── raw.py              # Basic pipeline benchmark
│   │   ├── summarizer.py       # Summarization pipeline benchmark
│   │   └── validity.py         # Validity checking benchmark
│   └── squad/                  # SQuAD dataset benchmarks
│       ├── raw.py              # Basic pipeline benchmark
│       ├── summarizer.py       # Summarization pipeline benchmark
│       └── validity.py         # Validity checking benchmark
│
├── database/                   # Database modules for vector storage
│   ├── hotpot/                 # HotpotQA database operations
│   ├── pubmed/                 # PubMedQA database operations
│   └── squad/                  # SQuAD database operations
│
├── helpers/                    # Utility modules
│   ├── data.py                 # Data handling utilities
│   ├── oai.py                  # OpenAI API wrapper
│   ├── pc.py                   # Pinecone integration
│   ├── progress.py             # Progress tracking tools
│   └── variables.py            # Common variables
│
├── pipelines/                  # Pipeline implementations
│   ├── raw.py                  # Raw question-answering pipeline
│   ├── validity.py             # Validity checking pipeline
│   └── baseline.py             # Baseline pipeline implementation
│
├── results/                    # Benchmark results in JSON format
│   └── rag/                    # RAG-specific benchmark results
│
└── temp/                       # Temporary files and caches
```

## Setup Instructions

### Prerequisites

- Python 3.8+
- Pinecone account (for vector database)
- OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Tonyhrule/Facts.git
cd Facts
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with the following variables:
```
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_HOST=your_pinecone_host_url
```

### Setting up Pinecone

1. Sign up for a [Pinecone account](https://www.pinecone.io/)
2. Create a new index with the following settings:
   - Dimensions: 1536 (for OpenAI embeddings)
   - Metric: Cosine
   - Pod Type: Starter (for testing) or Standard (for production)
3. Copy your API key and host URL to the `.env` file

## Usage

### Preparing the Database

Before running benchmarks, you need to prepare the vector database:

```bash
python src/run_database.py
```

This will prompt you to select which dataset to prepare for the database.

### Running Benchmarks

To run benchmarks on the prepared datasets:

```bash
python src/run_benchmark.py
```

This will prompt you to select which benchmark to run. Results will be saved in the `src/results/` directory.

### Running a Single Query

You can test a single query through a specific pipeline:

```bash
python src/run_pipeline.py
```

Note: You may need to modify the query in `run_pipeline.py` before running.

## Pipeline Types

### Raw Pipeline
Basic question-answering using retrieval-augmented generation without additional processing.

### Summarized Pipeline
Generates summaries of retrieved contexts before answering questions, which can improve accuracy for complex topics while reducing storage space.

### Validity Pipeline
Includes an additional verification step to check the factual accuracy of answers against the provided context. This pipeline leverages LLM-driven fact-checking to validate individual statements extracted from responses.

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

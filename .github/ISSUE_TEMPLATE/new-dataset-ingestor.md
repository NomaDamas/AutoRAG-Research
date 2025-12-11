---
name: New Dataset Ingestor
about: This issue template is for creating new dataset ingestor.
title: "[New Dataset] Add <New Dataset> Ingestor"
labels: enhancement, New Dataset
assignees: ''

---

## About this issue

The primary goal of this issue is to make the new Dataset Ingestor in the AutoRAG-Research.
In AutoRAG-Research, there will be numerous pre-ingested benchmark datasets for RAG (text-based and multi-modal RAGs). For example, BRIGHT, MSMARCO, BEIR, LIMIT, ViDoRe, and so on...
Your job is to implement the new Dataset Ingestor. The dataset ingestor should be made for each dataset, since all datasets have different structures, but AutoRAG-Research only supports one unified structure.
When the implementation of the dataset ingestor is completed, the end user can ingest the data into the AutoRAG-Research format in the PostgreSQL database and generate the embedding vector using the embedding model. Also, it can prepare the BM25 Pyserini index for the BM25 search.

## What to do

1. Download the target dataset from Huggingface or any other source. It is always highly recommended to download only a small subset of the dataset, since the full dataset can be huge.
2. Identify which will be proper for benchmarking. Sometimes, the dataset offers a huge number of training datasets, which do not require evaluation (benchmark).
3. Create a new class for the new dataset ingestor. It must inherit `TextEmbeddingDataIngestor` or `MultiModalEmbeddingDataIngestor`. The former is for a text-only dataset, the latter is for a multi-modal RAG dataset.
4. Implement the abstract methods to ingest the downloaded dataset into the AutoRAG-Research schema. You can check out the DB schema in `ai_instructions/db_schema.md`.
It is highly recommended to use `TextDataIngestionService` or `MultiModalIngestionService` for ingestion, since it offers many functions to ingest data into the PostgreSQL database.
5. Make a test code for the added class. You must follow the test instructions in `ai_instructions/test_code_generation_instructions.md`. Try to make a small subset of the full dataset (about five rows only, and the corpus should be minimal) for fast testing. We don't need the full dataset for testing purposes.
6. Run `make check` for linting and typing, and fix the errors.
7. Make a PR for your implementation.

## Remember this

1. Naming a branch depends on the Issue's number. If the issue number is 38, the branch name will be `Feature/#38`. This is a strict naming rule for your branch.
2. Try to reuse the existing functions in `util.py`, service, and repository layers.
3. Do not use the repository directly without the service layer.
4. Read `CLAUDE.md` and keep in mind the database structure and what dataset column will be what DB table and column.
5. A new file or class should exist in the `autorag_research/data`.
6. DO NOT MAKE ANY DOCUMENTATION.


## About a new dataset.

Name :
Link :
Description :

## How to download the dataset
If it uses Huggingface, just leave a comment that it uses the Huggingface datasets library.
If not, specify how to download it.

## Dataset structure
Please provide the dataset structure if you can.


@claude implement this with the above instructions. Ultrathink

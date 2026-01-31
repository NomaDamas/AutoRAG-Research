---
name: New Pipeline
about: This issue template is for creating new pipeline.
title: "[New Pipeline] Add <>"
labels: enhancement, New Pipeline
assignees: ''

---

## About this issue

The primary goal of this issue is to make the new pipeline in the AutoRAG-Research. In AutoRAG-Research, there will be numerous RAG pipelines. The retrieval pipeline searches proper passages in the corpus for the user query, and returns the retrieved contents and their doc IDs. The generation pipeline leverages LLM to answer the user query based on the retrieved contents and return the generated answer (which is the generation).
There are so many pipelines out there, like self-RAG, VisConde, HyDE, etc. Your job is to re-implement each RAG pipeline accurately from the research paper and its GitHub repository based on the AutoRAG-Research environment. AutoRAG-Research has its own dataset schema and vector search, BM25 search (PostgreSQL for DB, VectorChord for vector search, Pyserini for BM25)


## What to do

1. From the given pipeline description, the research paper, and the GitHub repository, figure out the core pipeline of the method.
2. Implement the pipeline into the AutoRAG-Research. You need to inherit `BaseRetrievalPipeline` for the retrieval pipeline and `BaseGenerationPipeline` for the generation pipeline.
3. For connecting to the PostgreSQL DB, use `RetrievalPipelineService` or `GenerationPipelineService`.
4. Create a config class that inherits from `BaseRetrievalPipelineConfig` or `BaseGenerationPipelineConfig`. The config contains all hyperparameters that can be configured while using the pipeline.
5. Make a test code for the added class. You must follow the test instructions in `ai_instructions/test_code_generation_instructions.md`. Try to use a mock instance for LLM or embedding model inference to prevent extensive time and resources for testing.
6. Run `make check` for linting and typing, and fix the errors.
7. Make a documentation in the `docs` folder for explaining the pipeline. The description has to come from the research paper and its implementation. The usage of the pipeline has to be compact. It is always better if the documentation is easy to read.
8. Make a PR for your implementation.


## Remember this

1. For the embedding model and LLM model, the Langchain embedding model and LLM model are used by default.
2. For multi-modal embedding models like BiPali, and late interaction embedding models like ColPali or ColBERT, the custom class can be used in the `autorag_research/embeddings` subpackage.
3. Do not initialize GPU-related models inside the pipeline. Always inject the dependency.
4. Suppose the user is using the API-based models for embedding and an LLM model. Rerankers might be run in a local environment if an API-based model is not available.
5. Do not make a DB session in the pipeline.
6. As the default configuration, try to use the exact thing that the research paper is used for.
7. Try to reuse the existing functions in `util.py`, service, and repository layers.
8. Do not use the repository directly without the service layer.


## About a new pipeline

Name :
Paper link :
Github repo :
Description :

## More instructions for this pipeline implementation

@claude implement this with the above instructions. Ultrathink

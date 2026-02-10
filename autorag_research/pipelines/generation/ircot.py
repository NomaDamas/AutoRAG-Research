"""IRCoT (Interleaving Retrieval with Chain-of-Thought) Generation Pipeline.

Implements the IRCoT algorithm that alternates between generating chain-of-thought
sentences and retrieving relevant paragraphs. This is a multi-step reasoning
approach that iteratively builds context for complex question answering.

Reference: IRCoT paper - Interleaving Retrieval with Chain-of-Thought Reasoning
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseLanguageModel
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.config import BaseGenerationPipelineConfig
from autorag_research.orm.service.generation_pipeline import GenerationResult
from autorag_research.pipelines.generation.base import BaseGenerationPipeline
from autorag_research.pipelines.retrieval.base import BaseRetrievalPipeline
from autorag_research.util import aggregate_token_usage, extract_langchain_token_usage

logger = logging.getLogger("AutoRAG-Research")

DEFAULT_REASONING_PROMPT = """You are answering a multi-step question using chain-of-thought reasoning.

Question: {query}

Available Paragraphs:
{paragraphs}

Previous Thoughts:
{cot_history}

Generate the next reasoning step. Think step-by-step about what information you need or what conclusion you can draw.

If you have enough information to answer the question, write: "The answer is: [your answer]"

Next Thought:"""

DEFAULT_QA_PROMPT = """Answer the following question using the provided paragraphs.

Question: {query}

Paragraphs:
{paragraphs}

Provide a concise, direct answer to the question based on the information in the paragraphs.

Answer:"""


class IRCoTGenerationPipeline(BaseGenerationPipeline):
    """IRCoT generation pipeline that interleaves retrieval with chain-of-thought reasoning.

    This pipeline implements the IRCoT algorithm:
    1. Initial retrieval with the original query
    2. Iteratively generate CoT sentences and retrieve more context
    3. Terminate when "answer is:" is detected or max_steps reached
    4. Generate final answer using all collected paragraphs

    The retrieval pipeline can be any BaseRetrievalPipeline implementation
    (BM25, vector search, hybrid, etc.), providing flexibility in the
    retrieval strategy while implementing the IRCoT reasoning pattern.

    Example:
        ```python
        from langchain_openai import ChatOpenAI

        from autorag_research.orm.connection import DBConnection
        from autorag_research.pipelines.generation.ircot import IRCoTGenerationPipeline
        from autorag_research.pipelines.retrieval.bm25 import BM25RetrievalPipeline

        db = DBConnection.from_config()
        session_factory = db.get_session_factory()

        # Create retrieval pipeline
        retrieval_pipeline = BM25RetrievalPipeline(
            session_factory=session_factory,
            name="bm25_for_ircot",
            tokenizer="bert",
        )

        # Create IRCoT generation pipeline
        pipeline = IRCoTGenerationPipeline(
            session_factory=session_factory,
            name="ircot_gpt4",
            llm=ChatOpenAI(model="gpt-4"),
            retrieval_pipeline=retrieval_pipeline,
            k_per_step=4,
            max_steps=8,
            paragraph_budget=15,
        )

        # Run pipeline
        results = pipeline.run(top_k=4)
        ```
    """

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        name: str,
        llm: "BaseLanguageModel",
        retrieval_pipeline: "BaseRetrievalPipeline",
        reasoning_prompt_template: str = DEFAULT_REASONING_PROMPT,
        qa_prompt_template: str = DEFAULT_QA_PROMPT,
        k_per_step: int = 4,
        max_steps: int = 8,
        paragraph_budget: int = 15,
        stop_sequence: str = "answer is:",
        schema: Any | None = None,
    ):
        """Initialize IRCoT generation pipeline.

        Args:
            session_factory: SQLAlchemy sessionmaker for database connections.
            name: Name for this pipeline.
            llm: LangChain BaseLanguageModel instance for CoT generation and QA.
            retrieval_pipeline: Retrieval pipeline for fetching relevant context.
            reasoning_prompt_template: Template for CoT reasoning prompts.
                Must contain {query}, {paragraphs}, and {cot_history} placeholders.
            qa_prompt_template: Template for final QA prompts.
                Must contain {query} and {paragraphs} placeholders.
            k_per_step: Number of paragraphs to retrieve per step.
            max_steps: Maximum reasoning-retrieval iterations.
            paragraph_budget: Maximum total paragraphs to collect (FIFO strategy).
            stop_sequence: Termination string in CoT output (case-insensitive).
            schema: Schema namespace from create_schema(). If None, uses default schema.
        """
        # IMPORTANT: Store parameters BEFORE calling super().__init__
        # because _get_pipeline_config() is called during base init
        self._reasoning_prompt_template = reasoning_prompt_template
        self._qa_prompt_template = qa_prompt_template
        self.k_per_step = k_per_step
        self.max_steps = max_steps
        self.paragraph_budget = paragraph_budget
        self.stop_sequence = stop_sequence

        super().__init__(session_factory, name, llm, retrieval_pipeline, schema)

    def _get_pipeline_config(self) -> dict[str, Any]:
        """Return IRCoT pipeline configuration for database storage.

        Returns:
            Configuration dictionary containing all pipeline parameters.
        """
        # Get model name safely - handle mock objects and missing attributes
        model_name = getattr(self._llm, "model_name", None)
        if model_name is None or not isinstance(model_name, str):
            model_name = type(self._llm).__name__

        return {
            "type": "ircot",
            "reasoning_prompt_template": self._reasoning_prompt_template,
            "qa_prompt_template": self._qa_prompt_template,
            "k_per_step": self.k_per_step,
            "max_steps": self.max_steps,
            "paragraph_budget": self.paragraph_budget,
            "stop_sequence": self.stop_sequence,
            "retrieval_pipeline_id": self._retrieval_pipeline.pipeline_id,
            "llm_model": model_name,
        }

    def _extract_first_sentence(self, text: str) -> str:
        """Extract the first sentence from generated text.

        Uses simple heuristics: split on '. ', '! ', or '? ' and take first part.
        If no sentence delimiter found, returns the entire text.

        Args:
            text: Generated text from LLM.

        Returns:
            First sentence (or entire text if no delimiter).
        """
        text = text.strip()

        if not text:
            return text

        # Find first sentence delimiter
        for delimiter in [". ", "! ", "? "]:
            if delimiter in text:
                return text.split(delimiter, 1)[0] + delimiter[0]

        # No delimiter found, return entire text
        return text

    def _build_reasoning_prompt(
        self,
        query: str,
        paragraphs: list[str],
        cot_history: list[str],
    ) -> str:
        """Build prompt for CoT reasoning step.

        Args:
            query: Original question.
            paragraphs: Current collection of retrieved paragraphs.
            cot_history: Previous CoT sentences.

        Returns:
            Formatted prompt string.
        """
        # Format paragraphs with numbering
        if paragraphs:
            numbered_paragraphs = "\n\n".join([f"[{i + 1}] {p}" for i, p in enumerate(paragraphs)])
        else:
            numbered_paragraphs = "(No paragraphs available)"

        # Format CoT history
        if cot_history:
            cot_text = "\n".join([f"Thought {i + 1}: {s}" for i, s in enumerate(cot_history)])
        else:
            cot_text = "(No previous thoughts)"

        return self._reasoning_prompt_template.format(
            query=query,
            paragraphs=numbered_paragraphs,
            cot_history=cot_text,
        )

    def _build_qa_prompt(self, query: str, paragraphs: list[str]) -> str:
        """Build prompt for final QA generation.

        Args:
            query: Original question.
            paragraphs: Final collection of retrieved paragraphs.

        Returns:
            Formatted prompt string.
        """
        # Format paragraphs with numbering
        if paragraphs:
            numbered_paragraphs = "\n\n".join([f"[{i + 1}] {p}" for i, p in enumerate(paragraphs)])
        else:
            numbered_paragraphs = "(No paragraphs available)"

        return self._qa_prompt_template.format(
            query=query,
            paragraphs=numbered_paragraphs,
        )

    async def _generate(self, query_id: int | str, top_k: int) -> GenerationResult:
        """Generate answer using IRCoT: interleave retrieval with chain-of-thought.

        Algorithm:
        1. Initial retrieval with original query -> P (paragraph collection)
        2. Initialize C (CoT sentences list)
        3. FOR step in range(max_steps):
             a. Generate CoT sentence using (query, P, C)
             b. Extract first sentence only -> add to C
             c. IF stop_sequence in sentence: BREAK
             d. Retrieve using CoT sentence as query -> new paragraphs
             e. Extend P with new paragraphs
             f. Apply paragraph_budget cap on P (FIFO)
        4. Generate final answer using (query, P)
        5. Return answer + metadata (C, chunk IDs, token usage)

        Args:
            query_id: The query ID to answer.
            top_k: Number of chunks to retrieve (used for k_per_step if specified).

        Returns:
            GenerationResult containing the generated text and metadata.
        """
        # Use k_per_step for retrieval count
        k = self.k_per_step

        # Initialize collections
        chunk_ids: list[int | str] = []  # Track unique chunk IDs
        paragraphs: list[str] = []  # Paragraph contents
        cot_sentences: list[str] = []  # Chain-of-thought history
        total_token_usage: dict[str, int] | None = None
        steps_completed = 0

        # 1. Initial retrieval with original query
        logger.debug(f"IRCoT: Initial retrieval for query ID : {query_id}...")
        initial_results = await self._retrieval_pipeline._retrieve_by_id(query_id, k)

        # Extract chunk IDs and get contents
        for result in initial_results:
            doc_id = result.get("doc_id")
            if doc_id is not None and doc_id not in chunk_ids:
                chunk_ids.append(doc_id)

        # Get paragraph contents
        if chunk_ids:
            paragraphs = self._service.get_chunk_contents(chunk_ids)

        query = self._service.get_query_text(query_id)

        # 2. Iterative reasoning-retrieval loop
        for step in range(self.max_steps):
            steps_completed = step + 1
            logger.debug(f"IRCoT: Step {steps_completed}/{self.max_steps}")

            # a. Generate CoT sentence
            reasoning_prompt = self._build_reasoning_prompt(query, paragraphs, cot_sentences)
            response = await self._llm.ainvoke(reasoning_prompt)

            # Extract response text
            response_text = response.content if hasattr(response, "content") else str(response)

            # Aggregate token usage
            token_usage = extract_langchain_token_usage(response)
            total_token_usage = aggregate_token_usage(total_token_usage, token_usage)

            # b. Extract first sentence only and add to history
            first_sentence = self._extract_first_sentence(response_text)
            cot_sentences.append(first_sentence)

            # c. Check termination condition (case-insensitive)
            if self.stop_sequence.lower() in response_text.lower():
                logger.debug(f"IRCoT: Termination detected at step {steps_completed}")
                break

            # d. Retrieve more paragraphs using CoT sentence as query
            new_results = await self._retrieval_pipeline.retrieve(first_sentence, k)

            # e. Extend paragraph collection with new chunks
            new_chunk_ids: list[int | str] = []
            for result in new_results:
                doc_id = result.get("doc_id")
                if doc_id is not None and doc_id not in chunk_ids:
                    new_chunk_ids.append(doc_id)
                    chunk_ids.append(doc_id)

            # Get contents for new chunks
            if new_chunk_ids:
                new_contents = self._service.get_chunk_contents(new_chunk_ids)
                paragraphs.extend(new_contents)

            # f. Apply paragraph_budget cap (FIFO: keep first N)
            if len(paragraphs) > self.paragraph_budget:
                paragraphs = paragraphs[: self.paragraph_budget]
                chunk_ids = chunk_ids[: self.paragraph_budget]

        # 3. Generate final answer
        qa_prompt = self._build_qa_prompt(query, paragraphs)
        qa_response = await self._llm.ainvoke(qa_prompt)

        # Extract final answer text
        answer_text = qa_response.content if hasattr(qa_response, "content") else str(qa_response)

        # Aggregate final token usage
        qa_token_usage = extract_langchain_token_usage(qa_response)
        total_token_usage = aggregate_token_usage(total_token_usage, qa_token_usage)

        # 4. Build metadata
        metadata = {
            "cot_sentences": cot_sentences,
            "chunk_ids": chunk_ids,
            "steps": steps_completed,
        }

        return GenerationResult(
            text=answer_text,
            token_usage=total_token_usage,
            metadata=metadata,
        )


@dataclass(kw_only=True)
class IRCoTGenerationPipelineConfig(BaseGenerationPipelineConfig):
    """Configuration for IRCoT generation pipeline.

    Example:
        ```python
        from langchain_openai import ChatOpenAI

        config = IRCoTGenerationPipelineConfig(
            name="ircot_gpt4",
            retrieval_pipeline_name="bm25_baseline",
            llm=ChatOpenAI(model="gpt-4"),
            k_per_step=4,
            max_steps=8,
            paragraph_budget=15,
            top_k=4,
        )
        ```
    """

    reasoning_prompt_template: str = field(default=DEFAULT_REASONING_PROMPT)
    qa_prompt_template: str = field(default=DEFAULT_QA_PROMPT)
    k_per_step: int = 4
    max_steps: int = 8
    paragraph_budget: int = 15
    stop_sequence: str = "answer is:"

    def get_pipeline_class(self) -> type["IRCoTGenerationPipeline"]:
        """Return the IRCoTGenerationPipeline class."""
        return IRCoTGenerationPipeline

    def get_pipeline_kwargs(self) -> dict[str, Any]:
        """Return kwargs for IRCoTGenerationPipeline constructor."""
        if self._retrieval_pipeline is None:
            msg = f"Retrieval pipeline '{self.retrieval_pipeline_name}' not injected"
            raise ValueError(msg)

        return {
            "llm": self.llm,
            "retrieval_pipeline": self._retrieval_pipeline,
            "reasoning_prompt_template": self.reasoning_prompt_template,
            "qa_prompt_template": self.qa_prompt_template,
            "k_per_step": self.k_per_step,
            "max_steps": self.max_steps,
            "paragraph_budget": self.paragraph_budget,
            "stop_sequence": self.stop_sequence,
        }

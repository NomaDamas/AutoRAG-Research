"""Question Decomposition retrieval pipeline for AutoRAG-Research."""

import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseLanguageModel
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.config import BaseRetrievalPipelineConfig
from autorag_research.pipelines.retrieval.base import BaseRetrievalPipeline
from autorag_research.rerankers.base import BaseReranker

DEFAULT_DECOMPOSITION_PROMPT = """You are decomposing a question for retrieval-augmented generation.

Question: {query}

Write up to {max_subquestions} short, standalone sub-questions that would help retrieve evidence.
Return one sub-question per line and no other text.
If decomposition is unnecessary, repeat the original question once.

Sub-questions:"""

_SUBQUESTION_PREFIX_RE = re.compile(
    r"^\s*(?:sub-?question\s*\d*\s*:|question\s*\d*\s*:|[-*•]|\d+[.)]|[A-Za-z][.)])\s*",
    re.IGNORECASE,
)


@dataclass(kw_only=True)
class QuestionDecompositionRetrievalPipelineConfig(BaseRetrievalPipelineConfig):
    """Configuration for the Question Decomposition retrieval pipeline."""

    llm: str | BaseLanguageModel
    inner_retrieval_pipeline_name: str
    reranker: str | BaseReranker | None = None
    decomposition_prompt_template: str = field(default=DEFAULT_DECOMPOSITION_PROMPT)
    max_subquestions: int = 3
    fetch_k_multiplier: int = 2
    _inner_retrieval_pipeline: BaseRetrievalPipeline | None = field(default=None, repr=False)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "llm" and isinstance(value, str):
            from autorag_research.injection import load_llm

            value = load_llm(value)
        if name == "reranker" and isinstance(value, str):
            from autorag_research.injection import load_reranker

            value = load_reranker(value)
        super().__setattr__(name, value)

    def get_pipeline_class(self) -> type["QuestionDecompositionRetrievalPipeline"]:
        return QuestionDecompositionRetrievalPipeline

    def get_pipeline_kwargs(self) -> dict[str, Any]:
        if self._inner_retrieval_pipeline is None:
            msg = f"Inner retrieval pipeline '{self.inner_retrieval_pipeline_name}' not injected"
            raise ValueError(msg)

        return {
            "llm": self.llm,
            "inner_retrieval_pipeline": self._inner_retrieval_pipeline,
            "reranker": self.reranker,
            "decomposition_prompt_template": self.decomposition_prompt_template,
            "max_subquestions": self.max_subquestions,
            "fetch_k_multiplier": self.fetch_k_multiplier,
        }

    def inject_retrieval_pipeline(self, pipeline: BaseRetrievalPipeline) -> None:
        self._inner_retrieval_pipeline = pipeline


class QuestionDecompositionRetrievalPipeline(BaseRetrievalPipeline):
    """Retrieval pipeline that decomposes a query, merges results, and optionally reranks."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        name: str,
        llm: BaseLanguageModel,
        inner_retrieval_pipeline: BaseRetrievalPipeline,
        reranker: BaseReranker | str | None = None,
        decomposition_prompt_template: str = DEFAULT_DECOMPOSITION_PROMPT,
        max_subquestions: int = 3,
        fetch_k_multiplier: int = 2,
        schema: Any | None = None,
    ):
        self._llm = llm
        self._inner_retrieval_pipeline = inner_retrieval_pipeline
        self._decomposition_prompt_template = decomposition_prompt_template
        self.max_subquestions = max_subquestions
        self.fetch_k_multiplier = fetch_k_multiplier
        if isinstance(reranker, str):
            from autorag_research.injection import load_reranker

            self._reranker = load_reranker(reranker)
        else:
            self._reranker = reranker

        super().__init__(session_factory, name, schema)

    def _get_pipeline_config(self) -> dict[str, Any]:
        return {
            "type": "question_decomposition",
            "max_subquestions": self.max_subquestions,
            "fetch_k_multiplier": self.fetch_k_multiplier,
            "decomposition_prompt_template": self._decomposition_prompt_template,
            "inner_retrieval_pipeline_id": getattr(self._inner_retrieval_pipeline, "pipeline_id", None),
            "reranker_model": self._reranker.model_name if self._reranker is not None else None,
        }

    @staticmethod
    def _extract_text(response: Any) -> str:
        return response.content if hasattr(response, "content") else str(response)

    @staticmethod
    def _normalize_question(question: str) -> str:
        return " ".join(question.lower().split())

    @staticmethod
    def _normalize_score(score: Any) -> float:
        if isinstance(score, (int, float)):
            return float(score)
        return 0.0

    def _build_decomposition_prompt(self, query: str) -> str:
        return self._decomposition_prompt_template.format(
            query=query,
            max_subquestions=self.max_subquestions,
        )

    def _parse_subquestions(self, decomposition_text: str) -> list[str]:
        text = decomposition_text.strip()
        if not text:
            return []

        candidates = [line.strip() for line in text.splitlines() if line.strip()]
        if len(candidates) == 1:
            inline_numbered = re.split(r"\s+(?=(?:\d+[.)]|[-*•])\s*)", candidates[0])
            if len(inline_numbered) > 1:
                candidates = [item.strip() for item in inline_numbered if item.strip()]
            elif candidates[0].count("?") > 1:
                candidates = [item.strip() for item in re.split(r"(?<=\?)\s+", candidates[0]) if item.strip()]

        subquestions: list[str] = []
        for candidate in candidates:
            cleaned = _SUBQUESTION_PREFIX_RE.sub("", candidate).strip()
            cleaned = cleaned.rstrip(" ;")
            if cleaned:
                subquestions.append(cleaned)
        return subquestions

    def _deduplicate_subquestions(self, query: str, subquestions: list[str]) -> list[str]:
        seen = {self._normalize_question(query)}
        unique_subquestions: list[str] = []

        for subquestion in subquestions:
            normalized = self._normalize_question(subquestion)
            if not normalized or normalized in seen:
                continue

            seen.add(normalized)
            unique_subquestions.append(subquestion)
            if len(unique_subquestions) >= self.max_subquestions:
                break

        return unique_subquestions

    def _merge_results(self, result_sets: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        merged: dict[int | str, dict[str, Any]] = {}

        for result_set in result_sets:
            for result in result_set:
                doc_id = result.get("doc_id")
                if doc_id is None:
                    continue

                score = self._normalize_score(result.get("score"))
                existing = merged.get(doc_id)
                if existing is None or score > self._normalize_score(existing.get("score")):
                    merged[doc_id] = {**result, "doc_id": doc_id, "score": score}

        return sorted(
            merged.values(),
            key=lambda item: (-self._normalize_score(item.get("score")), str(item.get("doc_id"))),
        )

    def _load_contents_for_results(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        missing_ids = [result["doc_id"] for result in results if result.get("content") in (None, "")]
        if not missing_ids:
            return results

        with self._service._create_uow() as uow:
            chunks = uow.chunks.get_by_ids(missing_ids)
            content_by_id = {chunk.id: chunk.contents for chunk in chunks}

        enriched: list[dict[str, Any]] = []
        for result in results:
            if result.get("content") in (None, ""):
                enriched.append({**result, "content": content_by_id.get(result["doc_id"], "")})
            else:
                enriched.append(result)
        return enriched

    async def _rerank_results(self, query_text: str, merged: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if self._reranker is None or not merged:
            return merged[:top_k]

        enriched = self._load_contents_for_results(merged)
        rerank_candidates = [result for result in enriched if result.get("content")]
        if not rerank_candidates:
            return merged[:top_k]

        documents = [str(result["content"]) for result in rerank_candidates]
        reranked = await self._reranker.arerank(query_text, documents, top_k=top_k)

        final_results: list[dict[str, Any]] = []
        for rerank_result in reranked:
            source = rerank_candidates[rerank_result.index]
            final_results.append({
                "doc_id": source["doc_id"],
                "score": rerank_result.score,
                "content": source.get("content", ""),
            })
        return final_results

    async def _retrieve_by_id(self, query_id: int | str, top_k: int) -> list[dict[str, Any]]:
        query_texts = self._service.fetch_query_texts([query_id])
        if not query_texts:
            return []

        query_text = query_texts[0]
        fetch_k = top_k * self.fetch_k_multiplier

        prompt = self._build_decomposition_prompt(query_text)
        response = await self._llm.ainvoke(prompt)
        decomposition_text = self._extract_text(response)
        subquestions = self._deduplicate_subquestions(query_text, self._parse_subquestions(decomposition_text))

        result_sets = [await self._inner_retrieval_pipeline._retrieve_by_id(query_id, fetch_k)]
        for subquestion in subquestions:
            result_sets.append(await self._inner_retrieval_pipeline.retrieve(subquestion, fetch_k))

        merged = self._merge_results(result_sets)
        return await self._rerank_results(query_text, merged, top_k)

    async def _retrieve_by_text(self, query_text: str, top_k: int) -> list[dict[str, Any]]:
        fetch_k = top_k * self.fetch_k_multiplier

        prompt = self._build_decomposition_prompt(query_text)
        response = await self._llm.ainvoke(prompt)
        decomposition_text = self._extract_text(response)
        subquestions = self._deduplicate_subquestions(query_text, self._parse_subquestions(decomposition_text))

        result_sets = [await self._inner_retrieval_pipeline._retrieve_by_text(query_text, fetch_k)]
        for subquestion in subquestions:
            result_sets.append(await self._inner_retrieval_pipeline._retrieve_by_text(subquestion, fetch_k))

        merged = self._merge_results(result_sets)
        return await self._rerank_results(query_text, merged, top_k)


__all__ = [
    "DEFAULT_DECOMPOSITION_PROMPT",
    "QuestionDecompositionRetrievalPipeline",
    "QuestionDecompositionRetrievalPipelineConfig",
]

import ast
import io
from typing import Literal

from datasets import load_dataset
from llama_index.core.embeddings import MultiModalEmbedding
from PIL import Image

from autorag_research.data.base import MultiModalEmbeddingDataIngestor
from autorag_research.exceptions import EmbeddingNotSetError, InvalidDatasetNameError, UnsupportedDataSubsetError
from autorag_research.orm.service.multi_modal_ingestion import MultiModalIngestionService

ViDoReDatasets = [
    "arxivqa_test_subsampled",  # options
    "docvqa_test_subsampled",  # no answer
    "infovqa_test_subsampled",  # no answer
    "tabfquad_test_subsampled",
    "tatdqa_test",
    "shiftproject_test",
    "syntheticDocQA_artificial_intelligence_test",
    "syntheticDocQA_energy_test",
    "syntheticDocQA_government_reports_test",
    "syntheticDocQA_healthcare_industry_test",
]


class ViDoReIngestor(MultiModalEmbeddingDataIngestor):
    def __init__(
        self,
        multi_modal_data_ingestion_service: MultiModalIngestionService,
        dataset_name: str,
        embedding_model: MultiModalEmbedding | None = None,
        late_interaction_embedding_model: MultiModalEmbedding | None = None,
    ):
        super().__init__(multi_modal_data_ingestion_service, embedding_model, late_interaction_embedding_model)
        self.ds = load_dataset(f"vidore/{dataset_name}")["test"]
        if dataset_name not in ViDoReDatasets:
            raise InvalidDatasetNameError(dataset_name)

    def ingest(self, subset: Literal["train", "dev", "test"] = "test"):
        if subset != "test":
            raise UnsupportedDataSubsetError(["train", "dev"])

    def embed_all(self, max_concurrency: int = 16, batch_size: int = 128) -> None:
        if self.embedding_model is None:
            raise EmbeddingNotSetError
        raise NotImplementedError

    def embed_all_late_interaction(self, max_concurrency: int = 16, batch_size: int = 128) -> None:
        if self.late_interaction_embedding_model is None:
            raise EmbeddingNotSetError
        raise NotImplementedError

    def ingest_qrels(self, query_pk_list: list[int], image_chunk_pk_list: list[int]) -> None:
        qrels = [
            {
                "query_id": query_pk,
                "chunk_id": None,
                "image_chunk_id": image_chunk_pk,
                "group_index": 0,
                "group_order": 0,
            }
            for query_pk, image_chunk_pk in zip(query_pk_list, image_chunk_pk_list, strict=True)
        ]
        self.service.add_retrieval_gt_batch(qrels)

    @staticmethod
    def pil_images_to_bytes(image_list: list[Image.Image]) -> list[tuple[bytes, str]]:
        """Convert PIL images to bytes with mimetype.

        Args:
            image_list: List of PIL Image objects.

        Returns:
            List of tuples (image_bytes, mimetype).
        """
        results = []
        for img in image_list:
            buffer = io.BytesIO()
            # Determine format based on image mode
            img_format = "PNG" if img.mode in ("RGBA", "LA", "P") else "JPEG"
            img.save(buffer, format=img_format)
            mimetype = f"image/{img_format.lower()}"
            results.append((buffer.getvalue(), mimetype))
        return results


class ViDoReArxivQAIngestor(ViDoReIngestor):
    def __init__(
        self,
        multi_modal_data_ingestion_service: MultiModalIngestionService,
        embedding_model: MultiModalEmbedding | None = None,
        late_interaction_embedding_model: MultiModalEmbedding | None = None,
    ):
        super().__init__(
            multi_modal_data_ingestion_service,
            "arxivqa_test_subsampled",
            embedding_model,
            late_interaction_embedding_model,
        )

    def ingest(self, subset: Literal["train", "dev", "test"] = "test"):
        super().ingest(subset)
        image_list = list(self.ds["image"])  # ty: ignore[invalid-argument-type]
        queries = list(self.ds["query"])  # ty: ignore[invalid-argument-type]
        options = ["\n".join(ast.literal_eval(opt)) for opt in list(self.ds["options"])]  # ty: ignore[invalid-argument-type]
        queries = [
            f"Given the following query and options, select the correct option.\n\nQuery: {q}\n\nOptions: {opts}"
            for q, opts in zip(queries, options, strict=True)
        ]

        answers = list(self.ds["answer"])  # ty: ignore[invalid-argument-type]

        if not (len(image_list) == len(queries) == len(answers)):
            raise ValueError("Length mismatch among image_list, queries, and answers.")  # noqa: TRY003

        # Convert PIL images to bytes
        image_bytes_list = self.pil_images_to_bytes(image_list)

        query_pk_list = self.service.add_queries([(query, [ans]) for query, ans in zip(queries, answers, strict=True)])

        # Add image chunks with content and mimetype (no file path needed)
        image_chunk_pk_list = self.service.add_image_chunks([
            (content, mimetype, None) for content, mimetype in image_bytes_list
        ])

        self.ingest_qrels(query_pk_list, image_chunk_pk_list)

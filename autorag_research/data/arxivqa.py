import io
import logging
import random
import re
import tarfile
from pathlib import Path
from typing import Literal

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from llama_index.core.embeddings import MultiModalEmbedding
from PIL import Image

from autorag_research.data.base import MultiModalEmbeddingDataIngestor
from autorag_research.embeddings.base import MultiVectorMultiModalEmbedding
from autorag_research.exceptions import EmbeddingNotSetError, ServiceNotSetError

RANDOM_SEED = 42
BATCH_SIZE = 1000
logger = logging.getLogger("AutoRAG-Research")


def _normalize_label(label: str, options: list[str]) -> str:
    label = label.strip()

    if len(label) == 1 and label.upper() in "ABCDE":
        return label.upper()

    match = re.match(r"^([A-Ea-e])[.)]", label)
    if match:
        return match.group(1).upper()

    for i, opt in enumerate(options):
        opt_letter = chr(ord("A") + i)
        opt_text = re.sub(r"^[A-Ea-e][.)]\s*", "", opt).strip()
        label_text = re.sub(r"^[A-Ea-e][.)]\s*", "", label).strip()
        if label_text.lower() == opt_text.lower():
            return opt_letter

    if label and label[0].upper() in "ABCDE":
        return label[0].upper()

    logger.warning(f"Could not normalize label: {label}")
    return label


def _format_query(question: str, options: list[str]) -> str:
    options_text = "\n".join(options)
    return f"Given the following query and options, select the correct option.\n\nQuery: {question}\n\nOptions: {options_text}"


def _pil_to_bytes(image: Image.Image) -> tuple[bytes, str]:
    buffer = io.BytesIO()
    img_format = "PNG" if image.mode in ("RGBA", "LA", "P") else "JPEG"
    image.save(buffer, format=img_format)
    return buffer.getvalue(), f"image/{img_format.lower()}"


class ArxivQAIngestor(MultiModalEmbeddingDataIngestor):
    def __init__(
        self,
        embedding_model: MultiModalEmbedding | None = None,
        late_interaction_embedding_model: MultiVectorMultiModalEmbedding | None = None,
        cache_dir: Path | None = None,
    ):
        super().__init__(embedding_model, late_interaction_embedding_model)
        self.cache_dir = cache_dir
        self._images_dir: Path | None = None
        self._ds = None

    def detect_primary_key_type(self) -> Literal["bigint"] | Literal["string"]:
        return "bigint"

    def _ensure_images_downloaded(self) -> Path:
        if self._images_dir is not None and self._images_dir.exists():
            return self._images_dir

        logger.info("Downloading images.tgz from HuggingFace...")
        archive_path = hf_hub_download(
            repo_id="MMInstruction/ArxivQA",
            filename="images.tgz",
            repo_type="dataset",
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
        )

        extract_dir = Path(archive_path).parent / "extracted"
        self._images_dir = extract_dir / "images"

        if not self._images_dir.exists():
            logger.info(f"Extracting images to {extract_dir}...")
            extract_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(extract_dir, filter="data")

        logger.info(f"Images ready at {self._images_dir}")
        return self._images_dir

    def _load_dataset(self):
        if self._ds is None:
            logger.info("Loading MMInstruction/ArxivQA dataset...")
            self._ds = load_dataset("MMInstruction/ArxivQA")["train"]  # ty: ignore[non-subscriptable]
        return self._ds

    @staticmethod
    def _normalize_label(label: str, options: list[str]) -> str:
        return _normalize_label(label, options)

    @staticmethod
    def _format_query(question: str, options: list[str]) -> str:
        return _format_query(question, options)

    @staticmethod
    def _pil_to_bytes(image: Image.Image) -> tuple[bytes, str]:
        return _pil_to_bytes(image)

    def ingest(
        self,
        subset: Literal["train", "dev", "test"] = "train",
        query_limit: int | None = None,
        corpus_limit: int | None = None,
    ) -> None:
        super().ingest(subset, query_limit, corpus_limit)

        if self.service is None:
            raise ServiceNotSetError

        images_dir = self._ensure_images_downloaded()
        ds = self._load_dataset()
        total_count = len(ds)

        effective_limit = total_count
        if query_limit is not None:
            effective_limit = min(effective_limit, query_limit)
        if corpus_limit is not None:
            effective_limit = min(effective_limit, corpus_limit)

        if effective_limit < total_count:
            rng = random.Random(RANDOM_SEED)  # noqa: S311
            selected_indices = sorted(rng.sample(range(total_count), effective_limit))
        else:
            selected_indices = list(range(total_count))

        logger.info(f"Ingesting {len(selected_indices)} examples from ArxivQA...")

        skipped_count = 0
        all_query_pks: list[int | str] = []
        all_image_chunk_pks: list[int | str] = []

        for batch_start in range(0, len(selected_indices), BATCH_SIZE):
            batch_indices = selected_indices[batch_start : batch_start + BATCH_SIZE]

            queries_data: list[dict] = []
            image_chunks_data: list[dict] = []

            for idx in batch_indices:
                row = ds[idx]
                image_path = row["image"]
                question = row["question"]
                options = row["options"]
                label = row["label"]

                try:
                    filename = image_path.replace("images/", "")
                    img = Image.open(images_dir / filename)
                    img_bytes, mimetype = _pil_to_bytes(img)
                except Exception as e:
                    logger.warning(f"Failed to load image {image_path}: {e}")
                    skipped_count += 1
                    continue

                formatted_query = _format_query(question, options)
                normalized_answer = _normalize_label(label, options)

                queries_data.append({
                    "contents": formatted_query,
                    "generation_gt": [normalized_answer],
                })

                image_chunks_data.append({
                    "contents": img_bytes,
                    "mimetype": mimetype,
                })

            if queries_data:
                query_pks = self.service.add_queries(queries_data)
                image_chunk_pks = self.service.add_image_chunks(image_chunks_data)
                all_query_pks.extend(query_pks)
                all_image_chunk_pks.extend(image_chunk_pks)

            logger.info(
                f"Processed {batch_start + len(batch_indices)}/{len(selected_indices)} (skipped: {skipped_count})"
            )

        self.ingest_qrels(all_query_pks, all_image_chunk_pks)

        logger.info(
            f"Ingestion complete: {len(all_query_pks)} queries, "
            f"{len(all_image_chunk_pks)} images, {skipped_count} skipped"
        )

    def ingest_qrels(
        self,
        query_pk_list: list[int | str],
        image_chunk_pk_list: list[int | str],
    ) -> None:
        if self.service is None:
            raise ServiceNotSetError

        self.service.add_retrieval_gt_batch(
            [
                (query_pk, image_chunk_pk)
                for query_pk, image_chunk_pk in zip(query_pk_list, image_chunk_pk_list, strict=True)
            ],
            chunk_type="image",
        )

    def embed_all(self, max_concurrency: int = 16, batch_size: int = 128) -> None:
        if self.embedding_model is None:
            raise EmbeddingNotSetError
        if self.service is None:
            raise ServiceNotSetError

        self.service.embed_all_queries(
            self.embedding_model.aget_query_embedding,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
        )
        self.service.embed_all_image_chunks(
            self.embedding_model.aget_image_embedding,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
        )

    def embed_all_late_interaction(
        self,
        max_concurrency: int = 16,
        batch_size: int = 128,
    ) -> None:
        if self.late_interaction_embedding_model is None:
            raise EmbeddingNotSetError
        if self.service is None:
            raise ServiceNotSetError

        self.service.embed_all_queries_multi_vector(
            self.late_interaction_embedding_model.aget_query_embedding,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
        )
        self.service.embed_all_image_chunks_multi_vector(
            self.late_interaction_embedding_model.aget_image_embedding,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
        )

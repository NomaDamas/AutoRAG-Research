# AutoRAG-Research Plugin Architecture Design

> **Status:** 설계 단계 (구현 예정)
> **Last Updated:** 2025-01-24

## 개요

AutoRAG-Research는 pip로 설치하는 패키지이므로, 사용자가 커스텀 파이프라인/메트릭을 추가하려면 **플러그인 형태**로 별도 패키지를 만들어야 합니다.

이 문서는 플러그인 아키텍처 설계를 정리합니다.

---

## 설계 원칙

1. **YAML = Single Source of Truth**: 파이프라인/메트릭 설정은 YAML로 관리
2. **Entry Points 기반 Discovery**: `pip install` 후 자동 등록
3. **Namespace 분리**: `autorag_research.pipelines`, `autorag_research.metrics`
4. **Hydra 호환**: `_target_` 기반 instantiation 유지

---

## 플러그인 패키지 구조

```
autorag-research-elasticsearch/
├── pyproject.toml
├── README.md
├── src/
│   └── autorag_research_elasticsearch/
│       ├── __init__.py
│       ├── pipelines/
│       │   ├── __init__.py
│       │   ├── es_pipeline.py        # ElasticsearchPipelineConfig + Pipeline
│       │   └── es_pipeline.yaml      # Hydra config
│       └── metrics/
│           ├── __init__.py
│           ├── es_metric.py
│           └── es_metric.yaml
```

---

## Entry Points 등록

```toml
# autorag-research-elasticsearch/pyproject.toml

[project]
name = "autorag-research-elasticsearch"
version = "0.1.0"
dependencies = [
    "autorag-research>=0.1.0",
    "elasticsearch>=8.0.0",
]

[project.entry-points."autorag_research.pipelines"]
elasticsearch = "autorag_research_elasticsearch.pipelines"

[project.entry-points."autorag_research.metrics"]
elasticsearch = "autorag_research_elasticsearch.metrics"
```

---

## Discovery 로직

### 현재 (Phase 1)

```python
# autorag_research/cli/utils/discovery.py

def discover_pipelines() -> dict[str, str]:
    """내부 configs만 스캔"""
    return discover_configs(Path("configs/pipelines"))
```

### 향후 (Phase 3)

```python
# autorag_research/cli/utils/discovery.py

from importlib.metadata import entry_points
from importlib.resources import files

def discover_pipelines() -> dict[str, str]:
    """내부 configs + 외부 플러그인 스캔"""
    result = {}

    # 1. 내부 configs (configs/pipelines/*.yaml)
    internal_configs = Path("configs/pipelines")
    if internal_configs.exists():
        result.update(discover_configs(internal_configs))

    # 2. 외부 플러그인 (entry_points)
    eps = entry_points(group="autorag_research.pipelines")
    for ep in eps:
        try:
            module = ep.load()
            # 플러그인 패키지 내 YAML 파일 스캔
            plugin_path = files(module)
            for yaml_file in plugin_path.iterdir():
                if yaml_file.name.endswith(".yaml"):
                    cfg = yaml.safe_load(yaml_file.read_text())
                    # 네임스페이스: "elasticsearch:es_search"
                    name = f"{ep.name}:{yaml_file.name.removesuffix('.yaml')}"
                    result[name] = cfg.get("description", "")
        except Exception as e:
            logger.warning(f"Failed to load plugin {ep.name}: {e}")

    return result
```

---

## 플러그인 YAML 예시

```yaml
# autorag_research_elasticsearch/pipelines/es_pipeline.yaml

_target_: autorag_research_elasticsearch.pipelines.es_pipeline.ElasticsearchPipelineConfig
description: "Elasticsearch-based vector search with hybrid BM25+dense"
name: es_search
host: localhost
port: 9200
index_name: autorag_index
top_k: 10
hybrid_weight: 0.5
```

---

## 플러그인 Python 코드 예시

```python
# autorag_research_elasticsearch/pipelines/es_pipeline.py

from dataclasses import dataclass
from typing import Any

from autorag_research.pipelines.retrieval.base import (
    BaseRetrievalPipeline,
    BaseRetrievalPipelineConfig,
)

@dataclass(kw_only=True)
class ElasticsearchPipelineConfig(BaseRetrievalPipelineConfig):
    """Elasticsearch pipeline configuration."""

    host: str = "localhost"
    port: int = 9200
    index_name: str = "autorag_index"
    hybrid_weight: float = 0.5

    def get_pipeline_class(self) -> type["ElasticsearchPipeline"]:
        return ElasticsearchPipeline

    def get_pipeline_kwargs(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "index_name": self.index_name,
            "hybrid_weight": self.hybrid_weight,
        }


class ElasticsearchPipeline(BaseRetrievalPipeline):
    """Elasticsearch-based retrieval pipeline."""

    def __init__(
        self,
        session_factory,
        name: str,
        schema,
        host: str,
        port: int,
        index_name: str,
        hybrid_weight: float,
        **kwargs,
    ):
        super().__init__(session_factory, name, schema, **kwargs)
        self.host = host
        self.port = port
        self.index_name = index_name
        self.hybrid_weight = hybrid_weight

    def _get_pipeline_config(self) -> dict[str, Any]:
        return {
            "type": "elasticsearch",
            "host": self.host,
            "port": self.port,
            "index_name": self.index_name,
            "hybrid_weight": self.hybrid_weight,
        }

    def _get_retrieval_func(self):
        from elasticsearch import Elasticsearch

        es = Elasticsearch(f"http://{self.host}:{self.port}")

        def retrieve(query: str, top_k: int) -> list[str]:
            # Elasticsearch hybrid search implementation
            response = es.search(
                index=self.index_name,
                body={
                    "query": {"match": {"content": query}},
                    "size": top_k,
                },
            )
            return [hit["_id"] for hit in response["hits"]["hits"]]

        return retrieve
```

---

## 사용 흐름

### 1. 플러그인 설치

```bash
pip install autorag-research-elasticsearch
```

### 2. 자동 Discovery

```bash
$ autorag-research list pipelines

Available Pipelines:
------------------------------------------------------------
  bm25_baseline              BM25 retrieval with VectorChord-BM25
  vector_search              Dense vector similarity search
  elasticsearch:es_search    Elasticsearch hybrid search    ← 자동 표시
```

### 3. experiment.yaml에서 사용

```yaml
defaults:
  - db: default
  - pipelines/elasticsearch:es_search@pipelines.0  # 플러그인 config 참조
  - metrics/recall@metrics.0
  - _self_

schema: beir_scifact_test
```

### 4. 실행

```bash
autorag-research run
```

---

## Hydra SearchPath 확장 (향후 구현)

플러그인 YAML을 Hydra가 인식하려면 SearchPath 확장 필요:

```python
# autorag_research/cli/commands/run.py

from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin

class PluginSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path):
        # 외부 플러그인 경로 추가
        eps = entry_points(group="autorag_research.pipelines")
        for ep in eps:
            module = ep.load()
            plugin_path = str(files(module))
            search_path.append(provider=ep.name, path=f"file://{plugin_path}")

Plugins.instance().register(PluginSearchPathPlugin)
```

---

## Ingestor 플러그인

Ingestor는 decorator + Literal 타입 힌트 기반 등록을 사용합니다.

### 플러그인 코드

```python
from typing import Literal
from langchain_core.embeddings import Embeddings
from autorag_research.data.registry import register_ingestor
from autorag_research.data.base import TextEmbeddingDataIngestor

# Define available datasets as Literal type
MY_DATASETS = Literal["dataset_a", "dataset_b", "dataset_c"]

@register_ingestor(
    name="elasticsearch",
    description="Ingest from Elasticsearch index",
)
class ElasticsearchIngestor(TextEmbeddingDataIngestor):
    def __init__(
        self,
        embedding_model: Embeddings,  # Skipped (known dependency)
        dataset_name: MY_DATASETS,     # -> --dataset-name, choices=[...], required
        host: str = "localhost",       # -> --host, default="localhost"
        port: int = 9200,              # -> --port, type=int, default=9200
    ):
        super().__init__(embedding_model)
        self.dataset_name = dataset_name
        self.host = host
        self.port = port
```

### Entry Point 등록

```toml
# pyproject.toml
[project.entry-points."autorag_research.ingestors"]
elasticsearch = "autorag_research_elasticsearch.ingestors"
```

### 자동 CLI 옵션 생성

위 코드에서 자동 생성되는 CLI:
```bash
autorag-research ingest elasticsearch \
    --dataset-name=dataset_a \
    --host=localhost \
    --port=9200
```

### 자동 추론 규칙

| `__init__` 파라미터 | CLI 옵션 |
|-------------------|---------|
| `embedding_model: Embeddings` | 스킵 (주입됨) |
| `name: Literal["a", "b"]` | `--name`, choices=["a", "b"], required |
| `name: Literal["a", "b"] = "a"` | `--name`, choices=["a", "b"], default="a" |
| `name: str` | `--name`, required |
| `count: int = 10` | `--count`, type=int, default=10 |
| `items: list[str]` | `--items`, comma-separated, is_list=True |

---

## 향후 작업

- [ ] Phase 3 구현: entry_points 기반 discovery (Pipelines/Metrics)
- [x] Ingestor decorator-based registration
- [ ] Hydra SearchPath 플러그인 구현
- [ ] 플러그인 템플릿 레포 생성 (`autorag-research-plugin-template`)
- [ ] 플러그인 개발 가이드 문서화
- [ ] CI/CD 템플릿 (테스트, 배포)

---

## 관련 이슈

- Phase 1 (YAML Auto-Discovery): 완료
- Phase 2 (Ingestor Plugin System): 완료
- Phase 3 (Pipeline/Metric Plugin System): 별도 이슈로 관리 예정

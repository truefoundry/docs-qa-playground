import abc
import enum
import json
import logging
import os
import warnings
from typing import List, Literal

import mlflow
import backend.modules.metadata_store.mlfoundry_store as mlfoundry_store
from pydantic import BaseModel

from backend.modules.metadata_store.base import BaseMetadataStore
from backend.modules.metadata_store.models import (
    Collection,
    CollectionCreate,
    CollectionIndexerJobRun,
    CollectionIndexerJobRunCreate,
    CollectionIndexerJobRunStatus,
)
from backend.utils.base import EmbedderConfig, KnowledgeSource, ParserConfig

DEFAULT_CHUNK_SIZE = 500


class MLRunTypes(str, enum.Enum):
    COLLECTION = "COLLECTION"
    COLLECTION_INDEXER_JOB_RUN = "COLLECTION_INDEXER_JOB_RUN"


class BaseParams(abc.ABC, BaseModel):
    @abc.abstractmethod
    def to_mlfoundry_params(self):
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def from_mlfoundry_params(cls, params):
        raise NotImplementedError()


class CollectionMetadata(BaseParams):
    type: Literal[MLRunTypes.COLLECTION] = MLRunTypes.COLLECTION
    description: str | None = None
    embedder_config: EmbedderConfig
    chunk_size: int

    def to_mlfoundry_params(self):
        return {
            "type": self.type.value,
            "description": self.description,
            "embedder_config": json.dumps(self.embedder_config.dict()),
            "chunk_size": str(self.chunk_size),
        }

    @classmethod
    def from_mlfoundry_params(cls, params: dict[str, str]):
        return CollectionMetadata(
            description=params.get("description"),
            embedder_config=EmbedderConfig.parse_obj(
                json.loads(params.get("embedder_config"))
            ),
            chunk_size=int(params.get("chunk_size", DEFAULT_CHUNK_SIZE)),
        )

    class Config:
        use_enum_values = True


class RunParams(BaseParams):
    type: Literal[
        MLRunTypes.COLLECTION_INDEXER_JOB_RUN
    ] = MLRunTypes.COLLECTION_INDEXER_JOB_RUN
    collection_name: str
    parser_config: ParserConfig
    knowledge_source: KnowledgeSource

    def to_mlfoundry_params(self):
        return {
            "type": self.type.value,
            "collection_name": self.collection_name,
            "parser_config": json.dumps(self.parser_config.dict()),
            "knowledge_source": json.dumps(self.knowledge_source.dict()),
        }

    @classmethod
    def from_mlfoundry_params(cls, params: dict[str, str]):
        return RunParams(
            collection_name=params.get("collection_name"),
            parser_config=ParserConfig.parse_obj(
                json.loads(params.get("parser_config"))
            ),
            knowledge_source=KnowledgeSource.parse_obj(
                json.loads(params.get("knowledge_source"))
            ),
        )

    class Config:
        use_enum_values = True


class Tags(BaseModel):
    type: MLRunTypes
    collection_name: str
    status: CollectionIndexerJobRunStatus = CollectionIndexerJobRunStatus.INITIALIZED

    class Config:
        use_enum_values = True


class MLFoundry(BaseMetadataStore):
    def __init__(self):
        logging.getLogger("mlfoundry").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        self.ml_repo_name = os.getenv("ML_REPO_NAME")
        if not self.ml_repo_name:
            raise Exception("ML_REPO_NAME environment variable is not set.")
        self.client = mlfoundry_store.get_client()
        self.client.create_ml_repo(self.ml_repo_name)

    def create_collection(self, collection: CollectionCreate) -> Collection:
        collection_name = collection.name
        try:
            existing_run = self.client.get_run_by_name(
                ml_repo=self.ml_repo_name, run_name=collection_name
            )
            if existing_run:
                raise Exception(
                    f"Collection with name {collection_name} already exists."
                )
        except mlflow.exceptions.RestException as exp:
            if exp.error_code != "RESOURCE_DOES_NOT_EXIST":
                raise exp

        created_collection = self.client.create_run(
            ml_repo=self.ml_repo_name,
            run_name=collection_name,
            tags=Tags(
                type=MLRunTypes.COLLECTION,
                collection_name=collection_name,
            ).dict(),
        )
        created_collection.log_params(
            CollectionMetadata(
                description=collection.description,
                embedder_config=collection.embedder_config,
                chunk_size=collection.chunk_size,
            ).to_mlfoundry_params()
        )
        created_collection.end()
        return Collection(
            name=collection.name,
            description=collection.description,
            embedder_config=collection.embedder_config,
            chunk_size=collection.chunk_size,
        )

    def get_collection_by_name(
        self, collection_name: str, include_runs=False
    ) -> Collection:
        ml_run = self.client.get_run_by_name(
            ml_repo=self.ml_repo_name, run_name=collection_name
        )
        collection_params = CollectionMetadata.from_mlfoundry_params(
            params=ml_run.get_params()
        )
        collection = Collection(
            name=ml_run.run_name,
            description=collection_params.description,
            embedder_config=collection_params.embedder_config,
            chunk_size=collection_params.chunk_size,
        )
        if include_runs:
            collection.indexer_job_runs = self.get_collection_indexer_job_runs(
                collection_name=ml_run.run_name
            )
        return collection

    def get_collections(
        self, names: List[str] = None, include_runs=False
    ) -> List[Collection]:
        ml_runs = self.client.search_runs(
            ml_repo=self.ml_repo_name,
            filter_string=f"params.type = '{MLRunTypes.COLLECTION.value}'",
        )
        collections = []
        for ml_run in ml_runs:
            if names and ml_run.run_name not in names:
                continue
            collection_params = CollectionMetadata.from_mlfoundry_params(
                params=ml_run.get_params()
            )
            collection = Collection(
                name=ml_run.run_name,
                description=collection_params.description,
                embedder_config=collection_params.embedder_config,
                chunk_size=collection_params.chunk_size,
            )
            if include_runs:
                collection.indexer_job_runs = self.get_collection_indexer_job_runs(
                    collection_name=ml_run.run_name
                )
            collections.append(collection)

        return collections

    def get_collection_indexer_job_runs(self, collection_name: str):
        runs = self.client.search_runs(
            ml_repo=self.ml_repo_name,
            filter_string=f"params.type = '{MLRunTypes.COLLECTION_INDEXER_JOB_RUN.value}' and params.collection_name = '{collection_name}'",
        )
        indexer_job_runs = []
        for run in runs:
            collection_inderer_job_run = RunParams.from_mlfoundry_params(
                params=run.get_params()
            )
            indexer_job_runs.append(
                CollectionIndexerJobRun(
                    name=run.run_name,
                    parser_config=collection_inderer_job_run.parser_config,
                    knowledge_source=collection_inderer_job_run.knowledge_source,
                    status=CollectionIndexerJobRunStatus[run.get_tags().get("status")],
                )
            )
        return indexer_job_runs

    def create_collection_indexer_job_run(
        self, collection_name: str, indexer_job_run: CollectionIndexerJobRunCreate
    ) -> CollectionIndexerJobRun:
        created_run = self.client.create_run(
            ml_repo=self.ml_repo_name,
            run_name=collection_name,
            tags=Tags(
                type=MLRunTypes.COLLECTION_INDEXER_JOB_RUN,
                collection_name=collection_name,
                status=CollectionIndexerJobRunStatus.INITIALIZED,
            ).dict(),
        )
        created_run.log_params(
            RunParams(
                collection_name=collection_name,
                parser_config=indexer_job_run.parser_config,
                knowledge_source=indexer_job_run.knowledge_source,
            ).to_mlfoundry_params()
        )
        return CollectionIndexerJobRun(
            name=created_run.run_name,
            knowledge_source=indexer_job_run.knowledge_source,
            parser_config=indexer_job_run.parser_config,
            status=CollectionIndexerJobRunStatus.INITIALIZED,
        )

    def delete_collection(self, collection_name: str):
        collection_inderer_job_runs = self.client.search_runs(
            ml_repo=self.ml_repo_name,
            filter_string=f"params.type = '{MLRunTypes.COLLECTION_INDEXER_JOB_RUN.value}' and params.collection_name = '{collection_name}'",
        )
        for collection_inderer_job_run in collection_inderer_job_runs:
            collection_inderer_job_run.delete()
        collection = self.client.get_run_by_name(
            ml_repo=self.ml_repo_name, run_name=collection_name
        )
        collection.delete()

    def update_indexer_job_run_status(
        self,
        collection_inderer_job_run_name: str,
        status: CollectionIndexerJobRunStatus,
    ):
        collection_inderer_job_run = self.client.get_run_by_name(
            ml_repo=self.ml_repo_name, run_name=collection_inderer_job_run_name
        )
        collection_inderer_job_run.set_tags({"status": status.value})
        if (
            status == CollectionIndexerJobRunStatus.COMPLETED
            or status == CollectionIndexerJobRunStatus.FAILED
        ):
            collection_inderer_job_run.end()

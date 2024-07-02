import enum
import json
import os
import tempfile
import warnings
from typing import Any, Dict, List

import mlflow
from fastapi import HTTPException
from truefoundry import ml

from backend.logger import logger
from backend.modules.metadata_store.base import BaseMetadataStore
from backend.types import (
    AssociateDataSourceWithCollection,
    AssociatedDataSources,
    Collection,
    CreateCollection,
    CreateDataIngestionRun,
    CreateDataSource,
    DataIngestionRun,
    DataIngestionRunStatus,
    DataSource,
)


class MLRunTypes(str, enum.Enum):
    """
    Configuration defining types of ML Runs allowed
    """

    COLLECTION = "COLLECTION"
    DATA_INGESTION_RUN = "DATA_INGESTION_RUN"
    DATA_SOURCE = "DATA_SOURCE"


class TrueFoundry(BaseMetadataStore):
    ml_runs: dict[str, ml.MlFoundryRun] = {}
    CONSTANT_DATA_SOURCE_RUN_NAME = "tfy-datasource"

    def __init__(self, *args, ml_repo_name: str, **kwargs):
        self.ml_repo_name = ml_repo_name
        if not self.ml_repo_name:
            raise Exception("`ml_repo_name` is not set.")
        logger.info(
            f"[Metadata Store] Initializing TrueFoundry Metadata Store: {self.ml_repo_name}"
        )
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        self.client = ml.get_client()
        self.client.create_ml_repo(self.ml_repo_name)
        logger.info(
            f"[Metadata Store] Initialized TrueFoundry Metadata Store: {self.ml_repo_name}"
        )
        super().__init__(*args, **kwargs)

    def _get_run_by_name(
        self, run_name: str, no_cache: bool = False
    ) -> ml.MlFoundryRun | None:
        """
        Cache the runs to avoid too many requests to the backend.
        """
        try:
            # TODO: Use a LRU Cache?
            if len(self.ml_runs.keys()) > 50:
                self.ml_runs = {}
            if no_cache:
                self.ml_runs[run_name] = self.client.get_run_by_name(
                    ml_repo=self.ml_repo_name, run_name=run_name
                )
            if run_name not in self.ml_runs.keys():
                self.ml_runs[run_name] = self.client.get_run_by_name(
                    ml_repo=self.ml_repo_name, run_name=run_name
                )
            return self.ml_runs.get(run_name)
        except mlflow.exceptions.RestException as exp:
            if exp.error_code == "RESOURCE_DOES_NOT_EXIST":
                return None
            raise exp

    def create_collection(self, collection: CreateCollection) -> Collection:
        """
        Create a collection from given CreateCollection object.
        It primarily has collection name, collection description and embedder configuration
        """
        logger.debug(f"[Metadata Store] Creating collection {collection.name}")
        existing_collection = self.get_collection_by_name(
            collection_name=collection.name, no_cache=True
        )
        if existing_collection:
            logger.error(
                f"[Metadata Store] Existing collection found with name {collection.name}"
            )
            raise HTTPException(
                status_code=400,
                detail=f"Collection with name {collection.name} already exists.",
            )
        params = {
            "entity_type": MLRunTypes.COLLECTION.value,
            "collection_name": collection.name,
        }

        run = self.client.create_run(
            ml_repo=self.ml_repo_name,
            run_name=collection.name,
            tags=params,
        )
        created_collection = Collection(
            name=collection.name,
            description=collection.description,
            embedder_config=collection.embedder_config,
        )
        self._save_entity_to_run(
            run=run, metadata=created_collection.dict(), params=params
        )
        run.end()
        logger.debug(f"[Metadata Store] Collection Saved")
        return created_collection

    def _get_entity_from_run(
        self,
        run: ml.MlFoundryRun,
    ) -> Dict[str, Any]:
        artifact = self._get_artifact_metadata_ml_run(run)
        metadata = artifact.metadata
        if not metadata:
            raise HTTPException(
                404,
                f"Artifact metadata not found. (Perhaps it was removed from the artifact?)",
            )
        return metadata

    def _get_artifact_metadata_ml_run(
        self, run: ml.MlFoundryRun
    ) -> ml.ArtifactVersion | None:
        params = run.get_params()
        metadata_artifact_fqn = params.get("metadata_artifact_fqn")
        if not metadata_artifact_fqn:
            raise HTTPException(
                404,
                f"Key metadata_artifact_fqn not found. (Perhaps the run {run.run_name} was modified?)",
            )
        artifacts = run.list_artifact_versions()
        for artifact in artifacts:
            if artifact.fqn == metadata_artifact_fqn:
                return artifact
        return None

    def _save_entity_to_run(
        self,
        run: ml.MlFoundryRun,
        metadata: Dict[str, Any],
        params: Dict[str, str],
    ):
        with tempfile.TemporaryDirectory() as tmp_dirname:
            file_path = os.path.join(tmp_dirname, "metadata.json")
            with open(file_path, "w") as f:
                f.write(json.dumps(metadata))
            artifact = run.log_artifact(
                name=run.run_name,
                artifact_paths=[ml.ArtifactPath(src=file_path)],
                description="This artifact contains the entity",
                metadata=metadata,
            )
            run.log_params({"metadata_artifact_fqn": artifact.fqn, **params})

    def _update_entity_in_run(
        self,
        run: ml.MlFoundryRun,
        metadata: Dict[str, Any],
    ):
        artifact = self._get_artifact_metadata_ml_run(run)
        artifact.metadata = metadata
        artifact.update()

    def get_collection_by_name(
        self, collection_name: str, no_cache: bool = True
    ) -> Collection | None:
        """Get collection from given collection name."""
        logger.debug(f"[Metadata Store] Getting collection with name {collection_name}")
        ml_run = self._get_run_by_name(run_name=collection_name, no_cache=no_cache)
        if not ml_run:
            logger.debug(
                f"[Metadata Store] Collection with name {collection_name} not found"
            )
            return None
        collection = self._populate_collection(
            Collection.parse_obj(self._get_entity_from_run(run=ml_run))
        )
        logger.debug(f"[Metadata Store] Fetched collection with name {collection_name}")
        return collection

    def get_retrieve_collection_by_name(
        self, collection_name: str, no_cache: bool = True
    ) -> Collection | None:
        """Get collection from given collection name. Used during retrieval"""
        logger.debug(f"[Metadata Store] Getting collection with name {collection_name}")
        ml_run = self._get_run_by_name(run_name=collection_name, no_cache=no_cache)
        if not ml_run:
            logger.debug(
                f"[Metadata Store] Collection with name {collection_name} not found"
            )
            return None
        collection = Collection.parse_obj(self._get_entity_from_run(run=ml_run))
        logger.debug(f"[Metadata Store] Fetched collection with name {collection_name}")
        return collection

    def get_collections(self) -> List[Collection]:
        """Get all the collections for the given client"""
        logger.debug(f"[Metadata Store] Listing all collection")
        ml_runs = self.client.search_runs(
            ml_repo=self.ml_repo_name,
            filter_string=f"params.entity_type = '{MLRunTypes.COLLECTION.value}'",
        )
        collections = []
        for ml_run in ml_runs:
            collection = Collection.parse_obj(self._get_entity_from_run(run=ml_run))
            collections.append(self._populate_collection(collection))
        logger.debug(f"[Metadata Store] Listed {len(collections)} collections")
        return collections

    def _populate_collection(self, collection: Collection):
        for (
            data_source_fqn,
            associated_data_source,
        ) in collection.associated_data_sources.items():
            data_source = self.get_data_source_from_fqn(data_source_fqn)
            associated_data_source.data_source = data_source
            collection.associated_data_sources[data_source_fqn] = associated_data_source
        return collection

    def associate_data_source_with_collection(
        self,
        collection_name: str,
        data_source_association: AssociateDataSourceWithCollection,
    ) -> Collection:
        logger.debug(
            f"[Metadata Store] Associating data_source {data_source_association.data_source_fqn} "
            f"to collection {collection_name}"
        )

        collection_run = self._get_run_by_name(
            run_name=collection_name,
            no_cache=True,
        )
        if not collection_run:
            raise HTTPException(
                404,
                f"Collection {collection_name} not found.",
            )
        data_source = self.get_data_source_from_fqn(
            data_source_association.data_source_fqn
        )
        if not data_source:
            logger.debug(
                f"[Metadata Store] data source with fqn {data_source_association.data_source_fqn} not found"
            )
            raise HTTPException(
                404,
                f"data source with fqn {data_source_association.data_source_fqn} not found",
            )
        # Always do this to avoid race conditions
        collection = Collection.parse_obj(self._get_entity_from_run(run=collection_run))
        associated_data_source = AssociatedDataSources(
            data_source_fqn=data_source_association.data_source_fqn,
            parser_config=data_source_association.parser_config,
        )
        collection.associated_data_sources[
            data_source_association.data_source_fqn
        ] = associated_data_source

        self._update_entity_in_run(run=collection_run, metadata=collection.dict())
        logger.debug(
            f"[Metadata Store] Associated data_source {data_source_association.data_source_fqn} "
            f"to collection {collection_name}"
        )
        return self._populate_collection(collection)

    def unassociate_data_source_with_collection(
        self,
        collection_name: str,
        data_source_fqn: str,
    ) -> Collection:
        logger.debug(
            f"[Metadata Store] Unassociating data_source {data_source_fqn} to collection {collection_name}"
        )
        collection_run = self._get_run_by_name(
            run_name=collection_name,
            no_cache=True,
        )
        if not collection_run:
            raise HTTPException(
                404,
                f"Collection {collection_name} not found.",
            )
        # Always do this to avoid run conditions
        collection = Collection.parse_obj(self._get_entity_from_run(run=collection_run))
        collection.associated_data_sources.pop(data_source_fqn)
        self._update_entity_in_run(run=collection_run, metadata=collection.dict())
        logger.debug(
            f"[Metadata Store] Unassociated data_source {data_source_fqn} to collection {collection_name}"
        )
        return self._populate_collection(collection)

    def create_data_source(self, data_source: CreateDataSource) -> DataSource:
        logger.debug(
            f"[Metadata Store] Creating new data_source of type {data_source.type}"
        )
        # fqn = get_data_source_fqn(data_source)
        fqn = data_source.fqn

        existing_data_source = self.get_data_source_from_fqn(fqn=fqn)
        if existing_data_source:
            raise HTTPException(400, f"Data source with fqn {fqn} already exists.")

        params = {
            "entity_type": MLRunTypes.DATA_SOURCE.value,
            "data_source_fqn": fqn,
        }
        run = self.client.create_run(
            ml_repo=self.ml_repo_name,
            run_name=self.CONSTANT_DATA_SOURCE_RUN_NAME,
            tags=params,
        )
        created_data_source = DataSource(
            type=data_source.type,
            uri=data_source.uri,
            fqn=fqn,
            metadata=data_source.metadata,
        )
        self._save_entity_to_run(
            run=run, metadata=created_data_source.dict(), params=params
        )
        run.end()
        logger.debug(
            f"[Metadata Store] Created new data_source of type {data_source.type}"
        )
        return created_data_source

    def get_data_source_from_fqn(self, fqn: str) -> DataSource | None:
        logger.debug(f"[Metadata Store] Getting data_source by fqn {fqn}")
        runs = self.client.search_runs(
            ml_repo=self.ml_repo_name,
            filter_string=f"params.entity_type = '{MLRunTypes.DATA_SOURCE.value}' and params.data_source_fqn = '{fqn}'",
        )
        for run in runs:
            data_source = DataSource.parse_obj(self._get_entity_from_run(run=run))
            logger.debug(f"[Metadata Store] Fetched Data Source with fqn {fqn}")
            return data_source
        logger.debug(f"[Metadata Store] Data Source with fqn {fqn} not found")
        return None

    def get_data_sources(self) -> List[DataSource]:
        logger.debug(f"[Metadata Store] Listing all data sources")
        runs = self.client.search_runs(
            ml_repo=self.ml_repo_name,
            filter_string=f"params.entity_type = '{MLRunTypes.DATA_SOURCE.value}'",
        )
        data_sources: List[DataSource] = []
        for run in runs:
            data_source = DataSource.parse_obj(self._get_entity_from_run(run=run))
            data_sources.append(data_source)
        logger.debug(f"[Metadata Store] Listed {len(data_sources)} data sources")
        return data_sources

    def create_data_ingestion_run(
        self, data_ingestion_run: CreateDataIngestionRun
    ) -> DataIngestionRun:
        logger.debug(
            f"[Metadata Store] Creating new ingestion run for "
            f"collection: {data_ingestion_run.collection_name} data source: {data_ingestion_run.data_source_fqn}"
        )
        params = {
            "entity_type": MLRunTypes.DATA_INGESTION_RUN.value,
            "collection_name": data_ingestion_run.collection_name,
            "data_source_fqn": data_ingestion_run.data_source_fqn,
        }
        run = self.client.create_run(
            ml_repo=self.ml_repo_name,
            run_name=data_ingestion_run.collection_name,
            tags={
                **params,
                "status": DataIngestionRunStatus.INITIALIZED.value,
            },
        )
        created_data_ingestion_run = DataIngestionRun(
            name=run.run_name,
            collection_name=data_ingestion_run.collection_name,
            data_source_fqn=data_ingestion_run.data_source_fqn,
            data_ingestion_mode=data_ingestion_run.data_ingestion_mode,
            parser_config=data_ingestion_run.parser_config,
            raise_error_on_failure=data_ingestion_run.raise_error_on_failure,
            status=DataIngestionRunStatus.INITIALIZED,
        )
        self._save_entity_to_run(
            run=run, metadata=created_data_ingestion_run.dict(), params=params
        )
        run.end()
        logger.debug(
            f"[Metadata Store] Created a ingestion run for "
            f"collection: {data_ingestion_run.collection_name} data source: {data_ingestion_run.data_source_fqn}"
        )
        return created_data_ingestion_run

    def get_data_ingestion_run(
        self, data_ingestion_run_name: str, no_cache: bool = False
    ) -> DataIngestionRun | None:
        logger.debug(
            f"[Metadata Store] Getting ingestion run {data_ingestion_run_name}"
        )
        run = self._get_run_by_name(run_name=data_ingestion_run_name, no_cache=no_cache)
        if not run:
            logger.debug(
                f"[Metadata Store] Ingestion run with name {data_ingestion_run_name} not found"
            )
            return None
        data_ingestion_run = DataIngestionRun.parse_obj(
            self._get_entity_from_run(run=run)
        )
        run_tags = run.get_tags()
        data_ingestion_run.status = DataIngestionRunStatus(run_tags.get("status"))
        logger.debug(
            f"[Metadata Store] Fetched Ingestion run with name {data_ingestion_run_name}"
        )
        return data_ingestion_run

    def get_data_ingestion_runs(
        self, collection_name: str, data_source_fqn: str = None
    ) -> List[DataIngestionRun]:
        logger.debug(
            f"[Metadata Store] Listing all data ingestion runs "
            f"for collection: {collection_name} & data source: {data_source_fqn}"
        )
        filter_str = (
            f"params.entity_type = '{MLRunTypes.DATA_INGESTION_RUN.value}' "
            f"and params.collection_name = '{collection_name}'"
        )
        if data_source_fqn:
            filter_str = (
                filter_str + f" and params.data_source_fqn = '{data_source_fqn}'"
            )
        runs = self.client.search_runs(
            ml_repo=self.ml_repo_name,
            filter_string=filter_str,
        )
        data_ingestion_runs: List[DataIngestionRun] = []
        for run in runs:
            data_ingestion_run = DataIngestionRun.parse_obj(
                self._get_entity_from_run(run=run)
            )
            run_tags = run.get_tags()
            data_ingestion_run.status = DataIngestionRunStatus(run_tags.get("status"))
            data_ingestion_runs.append(data_ingestion_run)
        logger.debug(
            f"[Metadata Store] Listed {len(data_ingestion_runs)} data ingestion runs "
            f"for collection: {collection_name} & data source: {data_source_fqn}"
        )
        return data_ingestion_runs

    def delete_collection(self, collection_name: str, include_runs=False):
        logger.debug(f"[Metadata Store] Deleting collection {collection_name}")
        collection = self._get_run_by_name(run_name=collection_name, no_cache=True)
        if not collection:
            logger.debug(
                f"[Metadata Store] Collection {collection_name} not found to delete."
            )
            return
        if include_runs:
            logger.debug(
                f"[Metadata Store] Fetching all data ingestion runs for {collection_name} to delete"
            )
            data_ingestion_runs = self.client.search_runs(
                ml_repo=self.ml_repo_name,
                filter_string=f"params.entity_type = '{MLRunTypes.DATA_INGESTION_RUN.value}' "
                f"and params.collection_name = '{collection_name}'",
            )
            logger.debug(
                f"[Metadata Store] Found data ingestion runs for {collection_name} to delete"
            )
            for collection_indexer_job_run in data_ingestion_runs:
                collection_indexer_job_run.delete()
            logger.debug(
                f"[Metadata Store] Deleted data ingestion runs for {collection_name}"
            )
        collection.delete()
        logger.debug(f"[Metadata Store] Deleted collection {collection_name}")

    def update_data_ingestion_run_status(
        self,
        data_ingestion_run_name: str,
        status: DataIngestionRunStatus,
    ):
        try:
            logger.debug(
                f"[Metadata Store] Updating status of data ingestion run {data_ingestion_run_name} to {status}"
            )
            data_ingestion_run = self._get_run_by_name(run_name=data_ingestion_run_name)
            if not data_ingestion_run:
                logger.error(
                    f"[Metadata Store] data ingestion run {data_ingestion_run_name} not found"
                )
                return
            data_ingestion_run.set_tags({"status": status.value})
            logger.debug(
                f"[Metadata Store] Updated status of data ingestion run {data_ingestion_run_name} to {status}"
            )
        except Exception as e:
            logger.exception(e)

    def log_metrics_for_data_ingestion_run(
        self,
        data_ingestion_run_name: str,
        metric_dict: dict[str, int | float],
        step: int = 0,
    ):
        try:
            logger.debug(
                f"[Metadata Store] Logging metrics for data ingestion run {data_ingestion_run_name}"
            )
            data_ingestion_run = self._get_run_by_name(run_name=data_ingestion_run_name)
            if not data_ingestion_run:
                logger.error(
                    f"[Metadata Store] data ingestion run {data_ingestion_run_name} not found"
                )
                return
            data_ingestion_run.log_metrics(metric_dict=metric_dict, step=step)
            logger.debug(
                f"[Metadata Store] Logging metrics for data ingestion run {data_ingestion_run_name}"
            )
        except Exception as e:
            logger.exception(e)

    def log_errors_for_data_ingestion_run(
        self, data_ingestion_run_name: str, errors: Dict[str, Any]
    ):
        try:
            logger.debug(
                f"[Metadata Store] Logging errors for data ingestion run {data_ingestion_run_name}"
            )
            data_ingestion_run = self._get_run_by_name(run_name=data_ingestion_run_name)
            if not data_ingestion_run:
                logger.error(f"Data ingestion run {data_ingestion_run_name} not found")
                return
            with tempfile.TemporaryDirectory() as tmp_dirname:
                file_path = os.path.join(tmp_dirname, "error.json")
                with open(file_path, "w") as f:
                    f.write(json.dumps(errors))
                data_ingestion_run.log_artifact(
                    name=data_ingestion_run.run_name,
                    artifact_paths=[ml.ArtifactPath(src=file_path)],
                    description="This artifact contains the errors during run",
                )
            logger.debug(
                f"[Metadata Store] Logged errors for data ingestion run {data_ingestion_run_name}"
            )
        except Exception as e:
            logger.exception(e)

    def list_collections(self) -> List[str]:
        logger.info(f"[Metadata Store] Listing all collection")
        ml_runs = self.client.search_runs(
            ml_repo=self.ml_repo_name,
            filter_string=f"params.entity_type = '{MLRunTypes.COLLECTION.value}'",
        )
        return [run.run_name for run in ml_runs]

    def list_data_sources(self) -> List[dict[str, str]]:
        logger.info(f"[Metadata Store] Listing all data sources")
        ml_runs = self.client.search_runs(
            ml_repo=self.ml_repo_name,
            filter_string=f"params.entity_type = '{MLRunTypes.DATA_SOURCE.value}'",
        )

        data_sources = []
        for run in ml_runs:
            run_params = run.get_params()
            try:
                data_sources.append(
                    {
                        "type": run_params.get("data_source_fqn").split("::")[0],
                        "uri": run_params.get("data_source_fqn").split("::")[1],
                        "fqn": run_params.get("data_source_fqn"),
                    }
                )
            except Exception as e:
                logger.error(f"Error in listing data sources: {e}")
                continue
        return data_sources

    def delete_data_source(self, data_source_fqn: str):
        raise Exception(
            "Truefoundry Metadata Store does not support delete data source"
        )

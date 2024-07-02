from abc import ABC
from typing import Any, Dict, List, Optional

from backend.constants import DATA_POINT_FQN_METADATA_KEY, FQN_SEPARATOR
from backend.types import (
    AssociateDataSourceWithCollection,
    Collection,
    CreateCollection,
    CreateDataIngestionRun,
    CreateDataSource,
    DataIngestionRun,
    DataIngestionRunStatus,
    DataSource,
    MetadataStoreConfig,
)
from backend.utils import run_in_executor


# TODO(chiragjn): Ideal would be we make `async def a*` abstract methods and drop sync ones
#   Implementations can then opt to call their sync versions using run_in_executor
class BaseMetadataStore(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def connect(cls, **kwargs) -> "BaseMetadataStore":
        return cls(**kwargs)

    @classmethod
    async def aconnect(cls, **kwargs) -> "BaseMetadataStore":
        return await run_in_executor(None, cls, **kwargs)

    def create_collection(self, collection: CreateCollection) -> Collection:
        """
        Create a collection in the metadata store
        """
        raise NotImplementedError()

    async def acreate_collection(self, collection: CreateCollection) -> Collection:
        """
        Create a collection in the metadata store
        """
        return await run_in_executor(
            None, self.create_collection, collection=collection
        )

    def get_collection_by_name(
        self, collection_name: str, no_cache: bool = True
    ) -> Optional[Collection]:
        """
        Get a collection from the metadata store by name
        """
        raise NotImplementedError()

    def get_retrieve_collection_by_name(
        self, collection_name: str, no_cache: bool = True
    ) -> Optional[Collection]:
        """
        Get a collection from the metadata store by name used during retrieval phase
        """
        raise NotImplementedError()

    async def aget_collection_by_name(
        self, collection_name: str, no_cache: bool = True
    ) -> Optional[Collection]:
        """
        Get a collection from the metadata store by name
        """
        return await run_in_executor(
            None,
            self.get_collection_by_name,
            collection_name=collection_name,
            no_cache=no_cache,
        )

    async def aget_retrieve_collection_by_name(
        self, collection_name: str, no_cache: bool = True
    ) -> Optional[Collection]:
        """
        Get a collection from the metadata store by name used during retrieval phase
        """
        return await run_in_executor(
            None,
            self.get_retrieve_collection_by_name,
            collection_name=collection_name,
            no_cache=no_cache,
        )

    def get_collections(
        self,
    ) -> List[Collection]:
        """
        Get all collections from the metadata store
        """
        raise NotImplementedError()

    async def aget_collections(
        self,
    ) -> List[Collection]:
        """
        Get all collections from the metadata store
        """
        return await run_in_executor(None, self.get_collections)

    def create_data_source(self, data_source: CreateDataSource) -> DataSource:
        """
        Create a data source in the metadata store
        """
        raise NotImplementedError()

    async def acreate_data_source(self, data_source: CreateDataSource) -> DataSource:
        """
        Create a data source in the metadata store
        """
        return await run_in_executor(
            None, self.create_data_source, data_source=data_source
        )

    def get_data_source_from_fqn(self, fqn: str) -> Optional[DataSource]:
        """
        Get a data source from the metadata store by fqn
        """
        raise NotImplementedError()

    async def aget_data_source_from_fqn(self, fqn: str) -> Optional[DataSource]:
        """
        Get a data source from the metadata store by fqn
        """
        return await run_in_executor(None, self.get_data_source_from_fqn, fqn=fqn)

    def get_data_sources(self) -> List[DataSource]:
        """
        Get all data sources from the metadata store
        """
        raise NotImplementedError()

    async def aget_data_sources(self) -> List[DataSource]:
        """
        Get all data sources from the metadata store
        """
        return await run_in_executor(
            None,
            self.get_data_sources,
        )

    def create_data_ingestion_run(
        self, data_ingestion_run: CreateDataIngestionRun
    ) -> DataIngestionRun:
        """
        Create a data ingestion run in the metadata store
        """
        raise NotImplementedError()

    async def acreate_data_ingestion_run(
        self, data_ingestion_run: CreateDataIngestionRun
    ) -> DataIngestionRun:
        """
        Create a data ingestion run in the metadata store
        """
        return await run_in_executor(
            None, self.create_data_ingestion_run, data_ingestion_run=data_ingestion_run
        )

    def get_data_ingestion_run(
        self, data_ingestion_run_name: str, no_cache: bool = False
    ) -> Optional[DataIngestionRun]:
        """
        Get a data ingestion run from the metadata store by name
        """
        raise NotImplementedError()

    async def aget_data_ingestion_run(
        self, data_ingestion_run_name: str, no_cache: bool = False
    ) -> Optional[DataIngestionRun]:
        """
        Get a data ingestion run from the metadata store by name
        """
        return await run_in_executor(
            None,
            self.get_data_ingestion_run,
            data_ingestion_run_name=data_ingestion_run_name,
            no_cache=no_cache,
        )

    def get_data_ingestion_runs(
        self, collection_name: str, data_source_fqn: str = None
    ) -> List[DataIngestionRun]:
        """
        Get all data ingestion runs from the metadata store
        """
        raise NotImplementedError()

    async def aget_data_ingestion_runs(
        self, collection_name: str, data_source_fqn: str = None
    ) -> List[DataIngestionRun]:
        """
        Get all data ingestion runs from the metadata store
        """
        return await run_in_executor(
            None,
            self.get_data_ingestion_runs,
            collection_name=collection_name,
            data_source_fqn=data_source_fqn,
        )

    def delete_collection(self, collection_name: str, include_runs=False):
        """
        Delete a collection from the metadata store
        """
        raise NotImplementedError()

    async def adelete_collection(self, collection_name: str, include_runs=False):
        """
        Delete a collection from the metadata store
        """
        return await run_in_executor(
            None,
            self.delete_collection,
            collection_name=collection_name,
            include_runs=include_runs,
        )

    def associate_data_source_with_collection(
        self,
        collection_name: str,
        data_source_association: AssociateDataSourceWithCollection,
    ) -> Collection:
        """
        Associate a data source with a collection in the metadata store
        """
        raise NotImplementedError()

    async def aassociate_data_source_with_collection(
        self,
        collection_name: str,
        data_source_association: AssociateDataSourceWithCollection,
    ) -> Collection:
        """
        Associate a data source with a collection in the metadata store
        """
        return await run_in_executor(
            None,
            self.associate_data_source_with_collection,
            collection_name=collection_name,
            data_source_association=data_source_association,
        )

    def unassociate_data_source_with_collection(
        self,
        collection_name: str,
        data_source_fqn: str,
    ) -> Collection:
        """
        Unassociate a data source with a collection in the metadata store
        """
        raise NotImplementedError()

    async def aunassociate_data_source_with_collection(
        self,
        collection_name: str,
        data_source_fqn: str,
    ) -> Collection:
        """
        Unassociate a data source with a collection in the metadata store
        """
        return await run_in_executor(
            None,
            self.unassociate_data_source_with_collection,
            collection_name=collection_name,
            data_source_fqn=data_source_fqn,
        )

    def update_data_ingestion_run_status(
        self,
        data_ingestion_run_name: str,
        status: DataIngestionRunStatus,
    ):
        """
        Update the status of a data ingestion run in the metadata store
        """
        raise NotImplementedError()

    async def aupdate_data_ingestion_run_status(
        self,
        data_ingestion_run_name: str,
        status: DataIngestionRunStatus,
    ):
        """
        Update the status of a data ingestion run in the metadata store
        """
        return await run_in_executor(
            None,
            self.update_data_ingestion_run_status,
            data_ingestion_run_name=data_ingestion_run_name,
            status=status,
        )

    def log_metrics_for_data_ingestion_run(
        self,
        data_ingestion_run_name: str,
        metric_dict: dict[str, int | float],
        step: int = 0,
    ):
        """
        Log metrics for a data ingestion run in the metadata store
        """
        raise NotImplementedError()

    async def alog_metrics_for_data_ingestion_run(
        self,
        data_ingestion_run_name: str,
        metric_dict: dict[str, int | float],
        step: int = 0,
    ):
        """
        Log metrics for a data ingestion run in the metadata store
        """
        return await run_in_executor(
            None,
            self.log_metrics_for_data_ingestion_run,
            data_ingestion_run_name=data_ingestion_run_name,
            metric_dict=metric_dict,
            step=step,
        )

    def log_errors_for_data_ingestion_run(
        self, data_ingestion_run_name: str, errors: Dict[str, Any]
    ):
        """
        Log errors for a data ingestion run in the metadata store
        """
        raise NotImplementedError()

    async def alog_errors_for_data_ingestion_run(
        self, data_ingestion_run_name: str, errors: Dict[str, Any]
    ):
        """
        Log errors for a data ingestion run in the metadata store
        """
        return await run_in_executor(
            None,
            self.log_errors_for_data_ingestion_run,
            data_ingestion_run_name=data_ingestion_run_name,
            errors=errors,
        )

    def list_collections(
        self,
    ) -> List[str]:
        """
        List all collection names from metadata store
        """
        raise NotImplementedError()

    async def alist_collections(
        self,
    ) -> List[str]:
        """
        List all collection names from metadata store
        """
        return await run_in_executor(
            None,
            self.list_collections,
        )

    def list_data_sources(
        self,
    ) -> List[str]:
        """
        List all data source names from metadata store
        """
        raise NotImplementedError()

    async def alist_data_sources(
        self,
    ) -> List[str]:
        """
        List all data source names from metadata store
        """
        return await run_in_executor(
            None,
            self.list_data_sources,
        )

    def delete_data_source(self, data_source_fqn: str):
        """
        Delete a data source from the metadata store
        """
        raise NotImplementedError()

    async def adelete_data_source(self, data_source_fqn: str):
        """
        Delete a data source from the metadata store
        """
        return await run_in_executor(
            None, self.delete_data_source, data_source_fqn=data_source_fqn
        )


def get_data_source_fqn(data_source: CreateDataSource) -> str:
    return f"{FQN_SEPARATOR}".join([data_source.type, data_source.uri])


def get_data_source_fqn_from_document_metadata(
    document_metadata: Dict[str, str]
) -> Optional[str]:
    if document_metadata and document_metadata.get(DATA_POINT_FQN_METADATA_KEY):
        parts = document_metadata.get(DATA_POINT_FQN_METADATA_KEY).split(FQN_SEPARATOR)
        if len(parts) == 3:
            return f"{FQN_SEPARATOR}".join(parts[:2])


# A global registry to store all available metadata store.
METADATA_STORE_REGISTRY = {}


def register_metadata_store(provider: str, cls):
    """
    Registers all the available metadata store.

    Args:
        provider: The type of the metadata store to be registered.
        cls: The metadata store class to be registered.

    Returns:
        None
    """
    global METADATA_STORE_REGISTRY
    if provider in METADATA_STORE_REGISTRY:
        raise ValueError(
            f"Error while registering class {cls.__name__}, "
            f"key {provider} is already taken by {METADATA_STORE_REGISTRY[provider].__name__}"
        )
    METADATA_STORE_REGISTRY[provider] = cls


async def get_metadata_store_client(
    config: MetadataStoreConfig,
) -> BaseMetadataStore:
    if config.provider in METADATA_STORE_REGISTRY:
        kwargs = config.config
        return await METADATA_STORE_REGISTRY[config.provider].aconnect(**kwargs)
    else:
        raise ValueError(f"Unknown metadata store type: {config.provider}")

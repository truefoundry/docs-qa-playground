
class ConsoleStore(BaseMetadataStore):
    @abstractmethod
    def create_collection(self, collection: CollectionCreate) -> Collection:
        """
        Create a collection in the metadata store and save its metadata
        """
        raise NotImplementedError()

    @abstractmethod
    def get_collection_by_name(
        self, collection_name: str, include_runs=False
    ) -> Collection:
        """
        Get a collection from the metadata store by name
        """
        raise NotImplementedError()

    @abstractmethod
    def get_collections(
        self, names: List[str] = None, include_runs=False
    ) -> List[Collection]:
        """
        Get all collections from the metadata store
        """
        raise NotImplementedError()

    @abstractmethod
    def get_collection_indexer_job_runs(self, collection_name: str):
        """
        Get all collection indexer job runs for a collection
        """
        raise NotImplementedError()

    @abstractmethod
    def create_collection_indexer_job_run(
        self, collection_name: str, indexer_job_run: CollectionIndexerJobRunCreate
    ) -> CollectionIndexerJobRun:
        """
        Create a collection indexer job run for a collection
        """
        raise NotImplementedError()

    @abstractmethod
    def delete_collection(self, collection_name: str):
        """
        Delete a collection from the metadata store
        """
        raise NotImplementedError()

    @abstractmethod
    def update_indexer_job_run_status(
        self,
        collection_inderer_job_run_name: str,
        status: CollectionIndexerJobRunStatus,
    ):
        """
        Update the status of a collection indexer job run
        """
        raise NotImplementedError()

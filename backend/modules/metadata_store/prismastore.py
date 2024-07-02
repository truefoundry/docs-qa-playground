import asyncio
import json
import os
import random
import shutil
import string
from typing import Any, Dict, List

from fastapi import HTTPException
from prisma import Prisma

from backend.logger import logger
from backend.modules.metadata_store.base import BaseMetadataStore
from backend.settings import settings
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


# TODO (chiragjn): Either we make everything async or add sync method to this
class PrismaStore(BaseMetadataStore):
    def __init__(self, *args, db, **kwargs) -> None:
        self.db = db
        super().__init__(*args, **kwargs)

    @classmethod
    async def aconnect(cls, **kwargs):
        try:
            db = Prisma()
            await db.connect()
            logger.info(f"Connected to Prisma....")
            return cls(db=db, **kwargs)
        except Exception as e:
            logger.exception(f"Failed to connect to Prisma: {e}")
            raise HTTPException(status_code=500, detail="Failed to connect to Prisma")

    ######
    # COLLECTIONS APIS
    ######

    async def acreate_collection(self, collection: CreateCollection) -> Collection:
        try:
            existing_collection = await self.aget_collection_by_name(collection.name)
        except Exception as e:
            logger.exception(f"Error: {e}")
            raise HTTPException(status_code=500, detail=e)

        if existing_collection:
            logger.error(f"Collection with name {collection.name} already exists")
            raise HTTPException(
                status_code=400,
                detail=f"Collection with name {collection.name} already exists",
            )

        try:
            logger.info(f"Creating collection: {collection.dict()}")
            collection_data = collection.dict()
            collection_data["embedder_config"] = json.dumps(
                collection_data["embedder_config"]
            )
            collection = await self.db.collection.create(data=collection_data)
            return collection
        except Exception as e:
            logger.exception(f"Error: {e}")
            raise HTTPException(status_code=500, detail=f"Error: {e}")

    async def aget_collection_by_name(
        self, collection_name: str, no_cache: bool = True
    ) -> Collection | None:
        try:
            collection = await self.db.collection.find_first(
                where={"name": collection_name}
            )
            if collection:
                return collection
            return None
        except Exception as e:
            logger.exception(f"Failed to get collection by name: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to get collection by name"
            )

    async def aget_retrieve_collection_by_name(
        self, collection_name: str, no_cache: bool = True
    ) -> Collection | None:
        return await self.aget_collection_by_name(collection_name, no_cache)

    async def aget_collections(self) -> List[Collection]:
        try:
            collections = await self.db.collection.find_many()
            return collections
        except Exception as e:
            logger.exception(f"Failed to get collections: {e}")
            raise HTTPException(status_code=500, detail="Failed to get collections")

    async def alist_collections(self) -> List[str]:
        try:
            collections = await self.aget_collections()
            return [collection.name for collection in collections]
        except Exception as e:
            logger.exception(f"Failed to list collections: {e}")
            raise HTTPException(status_code=500, detail="Failed to list collections")

    async def adelete_collection(self, collection_name: str, include_runs=False):
        try:
            collection = await self.aget_collection_by_name(collection_name)
            if not collection:
                logger.debug(f"Collection with name {collection_name} does not exist")
        except Exception as e:
            logger.exception(e)

        try:
            await self.db.collection.delete(where={"name": collection_name})
            if include_runs:
                try:
                    await self.db.ingestionruns.delete_many(
                        where={"collection_name": collection_name}
                    )
                except Exception as e:
                    logger.exception(f"Failed to delete data ingestion runs: {e}")
        except Exception as e:
            logger.exception(f"Failed to delete collection: {e}")
            raise HTTPException(status_code=500, detail="Failed to delete collection")

    ######
    # DATA SOURCE APIS
    ######
    async def acreate_data_source(self, data_source: CreateDataSource) -> DataSource:
        try:
            existing_data_source = await self.aget_data_source_from_fqn(data_source.fqn)
        except Exception as e:
            logger.exception(f"Error: {e}")
            raise HTTPException(status_code=500, detail=f"Error: {e}")

        if existing_data_source:
            logger.exception(f"Data source with fqn {data_source.fqn} already exists")
            raise HTTPException(
                status_code=400,
                detail=f"Data source with fqn {data_source.fqn} already exists",
            )

        try:
            data = data_source.dict()
            data["metadata"] = json.dumps(data["metadata"])
            data_source = await self.db.datasource.create(data)
            logger.info(f"Created data source: {data_source}")
            return data_source
        except Exception as e:
            logger.exception(f"Failed to create data source: {e}")
            raise HTTPException(status_code=500, detail=f"Error: {e}")

    async def aget_data_source_from_fqn(self, fqn: str) -> DataSource | None:
        try:
            data_source = await self.db.datasource.find_first(where={"fqn": fqn})
            if data_source:
                return data_source
            return None
        except Exception as e:
            logger.exception(f"Error: {e}")
            raise HTTPException(status_code=500, detail=f"Error: {e}")

    async def aget_data_sources(self) -> List[DataSource]:
        try:
            data_sources = await self.db.datasource.find_many()
            return data_sources
        except Exception as e:
            logger.exception(f"Error: {e}")
            raise HTTPException(status_code=500, detail=f"Error: {e}")

    async def aassociate_data_source_with_collection(
        self,
        collection_name: str,
        data_source_association: AssociateDataSourceWithCollection,
    ) -> Collection:
        try:
            existing_collection = await self.aget_collection_by_name(collection_name)
        except Exception as e:
            logger.exception(f"Error: {e}")
            raise HTTPException(status_code=500, detail=f"Error: {e}")

        if not existing_collection:
            logger.error(f"Collection with name {collection_name} does not exist")
            raise HTTPException(
                status_code=400,
                detail=f"Collection with name {collection_name} does not exist",
            )

        try:
            data_source = await self.aget_data_source_from_fqn(
                data_source_association.data_source_fqn
            )
        except Exception as e:
            logger.exception(f"Error: {e}")
            raise HTTPException(status_code=500, detail=f"Error: {e}")

        if not data_source:
            logger.error(
                f"Data source with fqn {data_source_association.data_source_fqn} does not exist"
            )
            raise HTTPException(
                status_code=400,
                detail=f"Data source with fqn {data_source_association.data_source_fqn} does not exist",
            )

        logger.info(f"Data source to associate: {data_source}")
        try:
            # Append datasource to existing collection
            existing_collection_associated_data_sources = (
                existing_collection.associated_data_sources
            )
            logger.info(
                f"Existing associated data sources: {existing_collection_associated_data_sources}"
            )

            data_src_to_associate = AssociatedDataSources(
                data_source_fqn=data_source_association.data_source_fqn,
                parser_config=data_source_association.parser_config,
                data_source=data_source,
            )

            if existing_collection_associated_data_sources:
                existing_collection_associated_data_sources[
                    data_src_to_associate.data_source_fqn
                ] = data_src_to_associate
            else:
                existing_collection_associated_data_sources = {
                    data_src_to_associate.data_source_fqn: data_src_to_associate
                }

            associated_data_sources: Dict[str, Dict[str, Any]] = {}
            for (
                data_source_fqn,
                data_source,
            ) in existing_collection_associated_data_sources.items():
                associated_data_sources[data_source_fqn] = data_source.dict()

            updated_collection = await self.db.collection.update(
                where={"name": collection_name},
                data={"associated_data_sources": json.dumps(associated_data_sources)},
            )
            return updated_collection

        except Exception as e:
            logger.exception(f"Error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error: {e}",
            )

    async def aunassociate_data_source_with_collection(
        self, collection_name: str, data_source_fqn: str
    ) -> Collection:
        try:
            collection = await self.aget_collection_by_name(collection_name)
        except Exception as e:
            logger.exception(f"Error: {e}")
            raise HTTPException(status_code=500, detail=f"Error: {e}")

        if not collection:
            logger.error(f"Collection with name {collection_name} does not exist")
            raise HTTPException(
                status_code=400,
                detail=f"Collection with name {collection_name} does not exist",
            )

        try:
            data_source = await self.aget_data_source_from_fqn(data_source_fqn)
        except Exception as e:
            logger.exception(f"Error: {e}")
            raise HTTPException(status_code=500, detail=f"Error: {e}")

        if not data_source:
            logger.error(f"Data source with fqn {data_source_fqn} does not exist")
            raise HTTPException(
                status_code=400,
                detail=f"Data source with fqn {data_source_fqn} does not exist",
            )

        associated_data_sources = collection.associated_data_sources
        if not associated_data_sources:
            logger.error(
                f"No associated data sources found for collection {collection_name}"
            )
            raise HTTPException(
                status_code=400,
                detail=f"No associated data sources found for collection {collection_name}",
            )
        if data_source_fqn not in associated_data_sources:
            logger.error(
                f"Data source with fqn {data_source_fqn} not associated with collection {collection_name}"
            )
            raise HTTPException(
                status_code=400,
                detail=f"Data source with fqn {data_source_fqn} not associated with collection {collection_name}",
            )

        associated_data_sources.pop(data_source_fqn, None)
        try:
            updated_collection = await self.db.collection.update(
                where={"name": collection_name},
                data={"associated_data_sources": json.dumps(associated_data_sources)},
            )
            return updated_collection
        except Exception as e:
            logger.exception(f"Failed to unassociate data source with collection: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to unassociate data source with collection",
            )

    async def alist_data_sources(
        self,
    ) -> List[dict[str, str]]:
        try:
            data_sources = await self.aget_data_sources()
            return [data_source.dict() for data_source in data_sources]
        except Exception as e:
            logger.exception(f"Failed to list data sources: {e}")
            raise HTTPException(status_code=500, detail="Failed to list data sources")

    async def adelete_data_source(self, data_source_fqn: str):
        if not settings.LOCAL:
            logger.error(f"Data source deletion is not allowed in local mode")
            raise HTTPException(
                status_code=400,
                detail=f"Data source deletion is not allowed in local mode",
            )
        # Check if data source exists if not raise an error
        try:
            data_source = await self.aget_data_source_from_fqn(data_source_fqn)
        except Exception as e:
            logger.exception(f"Error: {e}")
            raise HTTPException(status_code=500, detail=f"Error: {e}")

        if not data_source:
            logger.error(f"Data source with fqn {data_source_fqn} does not exist")
            raise HTTPException(
                status_code=400,
                detail=f"Data source with fqn {data_source_fqn} does not exist",
            )

        # Check if data source is associated with any collection
        try:
            collections = await self.aget_collections()
        except Exception as e:
            logger.exception(f"Error: {e}")
            raise HTTPException(status_code=500, detail=f"Error: {e}")

        for collection in collections:
            associated_data_sources = collection.associated_data_sources
            if associated_data_sources and data_source_fqn in associated_data_sources:
                logger.error(
                    f"Data source with fqn {data_source_fqn} is already associated with "
                    f"collection {collection.name}"
                )
                raise HTTPException(
                    status_code=400,
                    detail=f"Data source with fqn {data_source_fqn} is associated "
                    f"with collection {collection.name}. Delete the necessary collections "
                    f"or unassociate them from the collection(s) before deleting the data source",
                )

        # Delete the data source
        try:
            logger.info(f"Data source to delete: {data_source}")
            await self.db.datasource.delete(where={"fqn": data_source.fqn})
            # Delete the data from `/users_data` directory if data source is of type `localdir`
            if data_source.type == "localdir":
                data_source_uri = data_source.uri
                # data_source_uri is of the form: `/app/users_data/folder_name`
                folder_name = data_source_uri.split("/")[-1]
                folder_path = os.path.join("/app/user_data", folder_name)
                logger.info(
                    f"Deleting folder: {folder_path}, path exists: {os.path.exists(folder_path)}"
                )
                if os.path.exists(folder_path):
                    shutil.rmtree(folder_path)
                else:
                    logger.error(f"Folder does not exist: {folder_path}")

        except Exception as e:
            logger.exception(f"Failed to delete data source: {e}")
            raise HTTPException(status_code=500, detail="Failed to delete data source")

    ######
    # DATA INGESTION RUN APIS
    ######
    async def acreate_data_ingestion_run(
        self, data_ingestion_run: CreateDataIngestionRun
    ) -> DataIngestionRun:
        """Create a data ingestion run in the metadata store"""

        run_name = (
            data_ingestion_run.collection_name
            + "-"
            + "".join(random.choices(string.ascii_lowercase + string.digits, k=5))
        )
        created_data_ingestion_run = DataIngestionRun(
            name=run_name,
            collection_name=data_ingestion_run.collection_name,
            data_source_fqn=data_ingestion_run.data_source_fqn,
            parser_config=data_ingestion_run.parser_config,
            data_ingestion_mode=data_ingestion_run.data_ingestion_mode,
            status=DataIngestionRunStatus.INITIALIZED,
            raise_error_on_failure=data_ingestion_run.raise_error_on_failure,
        )

        try:
            run_data = created_data_ingestion_run.dict()
            run_data["parser_config"] = json.dumps(run_data["parser_config"])
            data_ingestion_run = await self.db.ingestionruns.create(data=run_data)
            return DataIngestionRun(**data_ingestion_run.dict())
        except Exception as e:
            logger.exception(f"Failed to create data ingestion run: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to create data ingestion run: {e}"
            )

    async def aget_data_ingestion_run(
        self, data_ingestion_run_name: str, no_cache: bool = False
    ) -> DataIngestionRun | None:
        try:
            data_ingestion_run = await self.db.ingestionruns.find_first(
                where={"name": data_ingestion_run_name}
            )
            logger.info(f"Data ingestion run: {data_ingestion_run}")
            if data_ingestion_run:
                return DataIngestionRun(**data_ingestion_run.dict())
            return None
        except Exception as e:
            logger.exception(f"Failed to get data ingestion run: {e}")
            raise HTTPException(status_code=500, detail=f"{e}")

    async def aget_data_ingestion_runs(
        self, collection_name: str, data_source_fqn: str = None
    ) -> List[DataIngestionRun]:
        """Get all data ingestion runs for a collection"""
        try:
            data_ingestion_runs = await self.db.ingestionruns.find_many(
                where={"collection_name": collection_name}
            )
            return data_ingestion_runs
        except Exception as e:
            logger.exception(f"Failed to get data ingestion runs: {e}")
            raise HTTPException(status_code=500, detail=f"{e}")

    async def aupdate_data_ingestion_run_status(
        self, data_ingestion_run_name: str, status: DataIngestionRunStatus
    ) -> DataIngestionRun:
        """Update the status of a data ingestion run"""
        try:
            updated_data_ingestion_run = await self.db.ingestionruns.update(
                where={"name": data_ingestion_run_name}, data={"status": status}
            )
            return updated_data_ingestion_run
        except Exception as e:
            logger.exception(f"Failed to update data ingestion run status: {e}")
            raise HTTPException(status_code=500, detail=f"{e}")

    async def alog_metrics_for_data_ingestion_run(
        self,
        data_ingestion_run_name: str,
        metric_dict: dict[str, int | float],
        step: int = 0,
    ):
        pass

    async def alog_errors_for_data_ingestion_run(
        self, data_ingestion_run_name: str, errors: Dict[str, Any]
    ):
        """Log errors for the given data ingestion run"""
        try:
            await self.db.ingestionruns.update(
                where={"name": data_ingestion_run_name},
                data={"errors": json.dumps(errors)},
            )
        except Exception as e:
            logger.exception(
                f"Failed to log errors data ingestion run {data_ingestion_run_name}: {e}"
            )
            raise HTTPException(status_code=500, detail=f"{e}")


if __name__ == "__main__":
    # initialize the PrismaStore
    prisma_store = PrismaStore.aconnect()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(prisma_store)

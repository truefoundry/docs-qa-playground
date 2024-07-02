import os
import tempfile
from typing import Dict, List

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from truefoundry.deploy import trigger_job

from backend.constants import DATA_POINT_FQN_METADATA_KEY, DATA_POINT_HASH_METADATA_KEY
from backend.indexer.types import DataIngestionConfig
from backend.logger import logger
from backend.modules.dataloaders.loader import get_loader_for_data_source
from backend.modules.metadata_store.client import get_client
from backend.modules.model_gateway.model_gateway import model_gateway
from backend.modules.parsers.parser import get_parser_for_extension
from backend.modules.vector_db.client import VECTOR_STORE_CLIENT
from backend.settings import settings
from backend.types import (
    Collection,
    CreateDataIngestionRun,
    DataIngestionMode,
    DataIngestionRunStatus,
    DataPointVector,
    IngestDataToCollectionDto,
    LoadedDataPoint,
)


def get_data_point_fqn_to_hash_map(
    data_point_vectors: List[DataPointVector],
) -> Dict[str, str]:
    """
    Returns a map of data point fqn to hash
    """
    data_point_fqn_to_hash: Dict[str, str] = {}
    for data_point_vector in data_point_vectors:
        if data_point_vector.data_point_fqn not in data_point_fqn_to_hash:
            data_point_fqn_to_hash[
                data_point_vector.data_point_fqn
            ] = data_point_vector.data_point_hash

    return data_point_fqn_to_hash


async def sync_data_source_to_collection(inputs: DataIngestionConfig):
    """
    Synchronizes the data source to the collection by performing the following steps:
    1. Updates the data ingestion run status to indicate that existing vectors are being fetched.
    2. Retrieves the existing data point vectors from the vector store.
    3. Logs the total number of existing data point vectors in the collection.
    4. Updates the data ingestion run status to indicate that data ingestion has started.
    5. Calls the _sync_data_source_to_collection function to perform the actual data ingestion.
    6. Updates the data ingestion run status to indicate the completion of data ingestion.
    7. If the data ingestion mode is set to FULL, deletes the outdated data point vectors from the vector store.
    8. Updates the data ingestion run status to indicate the completion of data cleanup.

    Args:
        inputs (DataIngestionConfig): The configuration for data ingestion.

    Raises:
        Exception: If any error occurs during data ingestion or cleanup.

    Returns:
        None
    """
    client = await get_client()
    await client.aupdate_data_ingestion_run_status(
        data_ingestion_run_name=inputs.data_ingestion_run_name,
        status=DataIngestionRunStatus.FETCHING_EXISTING_VECTORS,
    )
    try:
        existing_data_point_vectors = VECTOR_STORE_CLIENT.list_data_point_vectors(
            collection_name=inputs.collection_name,
            data_source_fqn=inputs.data_source.fqn,
        )
        previous_snapshot = get_data_point_fqn_to_hash_map(
            data_point_vectors=existing_data_point_vectors
        )

        logger.info(
            f"Total existing data point vectors in collection {inputs.collection_name}: {len(existing_data_point_vectors)}"
        )
    except Exception as e:
        logger.exception(e)
        await client.aupdate_data_ingestion_run_status(
            data_ingestion_run_name=inputs.data_ingestion_run_name,
            status=DataIngestionRunStatus.FETCHING_EXISTING_VECTORS_FAILED,
        )
        raise e

    await client.aupdate_data_ingestion_run_status(
        data_ingestion_run_name=inputs.data_ingestion_run_name,
        status=DataIngestionRunStatus.DATA_INGESTION_STARTED,
    )
    try:
        await _sync_data_source_to_collection(
            inputs=inputs,
            previous_snapshot=previous_snapshot,
        )
    except Exception as e:
        logger.exception(e)
        await client.aupdate_data_ingestion_run_status(
            data_ingestion_run_name=inputs.data_ingestion_run_name,
            status=DataIngestionRunStatus.DATA_INGESTION_FAILED,
        )
        raise e
    await client.aupdate_data_ingestion_run_status(
        data_ingestion_run_name=inputs.data_ingestion_run_name,
        status=DataIngestionRunStatus.DATA_INGESTION_COMPLETED,
    )
    # Delete the outdated data point vectors from the vector store
    if inputs.data_ingestion_mode == DataIngestionMode.FULL:
        await client.aupdate_data_ingestion_run_status(
            data_ingestion_run_name=inputs.data_ingestion_run_name,
            status=DataIngestionRunStatus.DATA_CLEANUP_STARTED,
        )
        try:
            VECTOR_STORE_CLIENT.delete_data_point_vectors(
                collection_name=inputs.collection_name,
                data_point_vectors=existing_data_point_vectors,
            )
        except Exception as e:
            logger.exception(e)
            await client.aupdate_data_ingestion_run_status(
                data_ingestion_run_name=inputs.data_ingestion_run_name,
                status=DataIngestionRunStatus.DATA_CLEANUP_FAILED,
            )
            raise e
    await client.aupdate_data_ingestion_run_status(
        data_ingestion_run_name=inputs.data_ingestion_run_name,
        status=DataIngestionRunStatus.COMPLETED,
    )


async def _sync_data_source_to_collection(
    inputs: DataIngestionConfig, previous_snapshot: Dict[str, str] = None
):
    """
    Synchronizes data from a data source to a collection.

    Args:
        inputs (DataIngestionConfig): The configuration for data ingestion.
        previous_snapshot (Dict[str, str], optional): A dictionary mapping data point FQNs to their hashes. Defaults to None.

    Raises:
        Exception: If failed to ingest any data points.

    Returns:
        None
    """

    client = await get_client()

    failed_data_point_fqns = []
    documents_ingested_count = 0
    # Create a temp dir to store the data
    with tempfile.TemporaryDirectory() as tmp_dirname:
        # Load the data from the source to the dest dir
        logger.info("Loading data from data source")
        data_source_loader = get_loader_for_data_source(inputs.data_source.type)
        loaded_data_points_batch_iterator = data_source_loader.load_filtered_data(
            data_source=inputs.data_source,
            dest_dir=tmp_dirname,
            previous_snapshot=previous_snapshot,
            batch_size=inputs.batch_size,
            data_ingestion_mode=inputs.data_ingestion_mode,
        )

        for loaded_data_points_batch in loaded_data_points_batch_iterator:
            try:
                await ingest_data_points(
                    inputs=inputs,
                    loaded_data_points=loaded_data_points_batch,
                    documents_ingested_count=documents_ingested_count,
                )
                documents_ingested_count = documents_ingested_count + len(
                    loaded_data_points_batch
                )
            except Exception as e:
                logger.exception(e)
                if inputs.raise_error_on_failure:
                    raise e
                failed_data_point_fqns.extend(
                    [doc.data_point_fqn for doc in loaded_data_points_batch]
                )

        if len(failed_data_point_fqns) > 0:
            logger.error(
                f"Failed to ingest {len(failed_data_point_fqns)} data points. data point fqns:"
            )
            logger.error(failed_data_point_fqns)
            await client.alog_errors_for_data_ingestion_run(
                data_ingestion_run_name=inputs.data_ingestion_run_name,
                errors={"failed_data_point_fqns": failed_data_point_fqns},
            )
            raise Exception(
                f"Failed to ingest {len(failed_data_point_fqns)} data points"
            )


async def ingest_data_points(
    inputs: DataIngestionConfig,
    loaded_data_points: List[LoadedDataPoint],
    documents_ingested_count: int,
):
    """
    Ingests data points into the vector store for a given batch.

    Args:
        inputs (DataIngestionConfig): The configuration for data ingestion.
        loaded_data_points (List[LoadedDataPoint]): The list of loaded data points to be ingested.
        documents_ingested_count (int): The count of documents already ingested.

    Returns:
        None: If no documents are found to index in the given batch.

    Raises:
        None

    """
    embeddings = model_gateway.get_embedder_from_model_config(
        model_name=inputs.embedder_config.model_config.name
    )
    documents_to_be_upserted = []
    logger.info(
        f"Processing {len(loaded_data_points)} new documents and completed: {documents_ingested_count}"
    )
    for index, loaded_data_point in enumerate(loaded_data_points):
        logger.info(
            f"[{index+1}/{len(loaded_data_points)}/{documents_ingested_count}] Parsing [{index+1}/{len(loaded_data_points)}] new document"
        )
        # Get parser for required file extension
        parser = get_parser_for_extension(
            file_extension=loaded_data_point.file_extension,
            parsers_map=inputs.parser_config.parser_map,
            max_chunk_size=inputs.parser_config.chunk_size,
            chunk_overlap=inputs.parser_config.chunk_overlap,
            additional_config=inputs.parser_config.additional_config,
        )
        if parser is None:
            logger.warning(
                f"Could not parse data point {loaded_data_point.data_point_fqn} as no parser found for file extension: {loaded_data_point.file_extension}"
            )
            continue
        # chunk the given document
        chunks = await parser.get_chunks(
            filepath=loaded_data_point.local_filepath,
            metadata=loaded_data_point.metadata,
        )
        # Update data source metadata
        for chunk in chunks:
            if loaded_data_point.metadata:
                chunk.metadata.update(loaded_data_point.metadata)
            # Most importantly, update the metadata with data point fqn and hash
            chunk.metadata.update(
                {
                    f"{DATA_POINT_FQN_METADATA_KEY}": loaded_data_point.data_point_fqn,
                    f"{DATA_POINT_HASH_METADATA_KEY}": loaded_data_point.data_point_hash,
                }
            )
            documents_to_be_upserted.append(chunk)
        logger.info("%s -> %s chunks", loaded_data_point.local_filepath, len(chunks))

        # delete the file from temp dir after processing
        try:
            if loaded_data_point.local_filepath:
                os.remove(loaded_data_point.local_filepath)
                logger.debug(
                    f"Processing done! Deleting file {loaded_data_point.local_filepath}"
                )
        except Exception as e:
            logger.exception(
                f"Failed to delete file {loaded_data_point.local_filepath} after processing. Error: {e}"
            )
        # delete the local_filepath from the loaded_data_point object
        try:
            if loaded_data_point.local_metadata_file_path:
                os.remove(loaded_data_point.local_metadata_file_path)
                logger.debug(
                    f"Processing done! Deleting file {loaded_data_point.local_metadata_file_path}"
                )
        except Exception as e:
            logger.exception(
                f"Failed to delete file {loaded_data_point.local_metadata_file_path} after processing. Error: {e}"
            )

    docs_to_index_count = len(documents_to_be_upserted)
    if docs_to_index_count == 0:
        logger.warning(
            "No documents found to index in given batch. Moving to next batch..."
        )
        return
    logger.info(
        f"Upserting {docs_to_index_count} documents to vector store for given batch"
    )
    # Upserted all the documents_to_be_ingested
    VECTOR_STORE_CLIENT.upsert_documents(
        collection_name=inputs.collection_name,
        documents=documents_to_be_upserted,
        embeddings=embeddings,
        incremental=inputs.data_ingestion_mode == DataIngestionMode.INCREMENTAL,
    )


async def ingest_data(request: IngestDataToCollectionDto):
    """Ingest data into the collection"""
    try:
        client = await get_client()
        collection = await client.aget_collection_by_name(request.collection_name)

        # convert to pydantic model if not already -> For prisma models
        if not isinstance(collection, Collection):
            collection = Collection(**collection.dict())

        if not collection:
            logger.error(
                f"Collection with name {request.collection_name} does not exist."
            )
            raise HTTPException(
                status_code=404,
                detail=f"Collection with name {request.collection_name} does not exist.",
            )

        if not collection.associated_data_sources:
            logger.error(
                f"Collection {request.collection_name} does not have any associated data sources."
            )
            raise HTTPException(
                status_code=400,
                detail=f"Collection {request.collection_name} does not have any associated data sources.",
            )
        associated_data_sources_to_be_ingested = []
        if request.data_source_fqn:
            associated_data_sources_to_be_ingested = [
                collection.associated_data_sources.get(request.data_source_fqn)
            ]
        else:
            associated_data_sources_to_be_ingested = (
                collection.associated_data_sources.values()
            )

        logger.info(f"Associated: {associated_data_sources_to_be_ingested}")
        for associated_data_source in associated_data_sources_to_be_ingested:
            logger.debug(
                f"Starting ingestion for data source fqn: {associated_data_source.data_source_fqn}"
            )
            if not request.run_as_job or settings.LOCAL:
                data_ingestion_run = CreateDataIngestionRun(
                    collection_name=collection.name,
                    data_source_fqn=associated_data_source.data_source_fqn,
                    embedder_config=collection.embedder_config,
                    parser_config=associated_data_source.parser_config,
                    data_ingestion_mode=request.data_ingestion_mode,
                    raise_error_on_failure=request.raise_error_on_failure,
                )
                created_data_ingestion_run = await client.acreate_data_ingestion_run(
                    data_ingestion_run=data_ingestion_run
                )
                await sync_data_source_to_collection(
                    inputs=DataIngestionConfig(
                        collection_name=created_data_ingestion_run.collection_name,
                        data_ingestion_run_name=created_data_ingestion_run.name,
                        data_source=associated_data_source.data_source,
                        embedder_config=collection.embedder_config,
                        parser_config=created_data_ingestion_run.parser_config,
                        data_ingestion_mode=created_data_ingestion_run.data_ingestion_mode,
                        raise_error_on_failure=created_data_ingestion_run.raise_error_on_failure,
                        batch_size=request.batch_size,
                    )
                )
                created_data_ingestion_run.status = DataIngestionRunStatus.COMPLETED
            else:
                if not settings.JOB_FQN or not settings.JOB_COMPONENT_NAME:
                    logger.error(
                        "Job FQN and Job Component Name are required to trigger the job"
                    )
                    raise HTTPException(
                        status_code=500,
                        detail="Job FQN and Job Component Name are required to trigger the job",
                    )
                data_ingestion_run = CreateDataIngestionRun(
                    collection_name=collection.name,
                    data_source_fqn=associated_data_source.data_source_fqn,
                    embedder_config=collection.embedder_config,
                    parser_config=associated_data_source.parser_config,
                    data_ingestion_mode=request.data_ingestion_mode,
                    raise_error_on_failure=request.raise_error_on_failure,
                )
                created_data_ingestion_run = await client.acreate_data_ingestion_run(
                    data_ingestion_run=data_ingestion_run
                )
                trigger_job(
                    application_fqn=settings.JOB_FQN,
                    component_name=settings.JOB_COMPONENT_NAME,
                    params={
                        "collection_name": collection.name,
                        "data_source_fqn": associated_data_source.data_source_fqn,
                        "data_ingestion_run_name": created_data_ingestion_run.name,
                        "data_ingestion_mode": request.data_ingestion_mode.value,
                        "raise_error_on_failure": (
                            "True" if request.raise_error_on_failure else "False"
                        ),
                    },
                )
        return JSONResponse(
            status_code=201,
            content={"message": "triggered"},
        )

    except HTTPException as exp:
        raise exp
    except Exception as exp:
        logger.exception(exp)
        raise HTTPException(status_code=500, detail=str(exp))

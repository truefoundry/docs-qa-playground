from fastapi import APIRouter, HTTPException, Path
from fastapi.responses import JSONResponse

from backend.indexer.indexer import ingest_data as ingest_data_to_collection
from backend.logger import logger
from backend.modules.metadata_store.client import get_client
from backend.modules.model_gateway.model_gateway import model_gateway
from backend.modules.vector_db.client import VECTOR_STORE_CLIENT
from backend.types import (
    AssociateDataSourceWithCollection,
    AssociateDataSourceWithCollectionDto,
    CreateCollection,
    CreateCollectionDto,
    IngestDataToCollectionDto,
    ListDataIngestionRunsDto,
    UnassociateDataSourceWithCollectionDto,
)

router = APIRouter(prefix="/v1/collections", tags=["collections"])


@router.get("")
async def get_collections():
    """API to list all collections with details"""
    try:
        logger.debug("Listing all the collections...")
        client = await get_client()
        collections = await client.aget_collections()
        if collections is None:
            return JSONResponse(content={"collections": []})
        return JSONResponse(
            content={"collections": [obj.model_dump() for obj in collections]}
        )
    except Exception as exp:
        logger.exception("Failed to get collection")
        raise HTTPException(status_code=500, detail=str(exp))


@router.get("/list")
async def list_collections():
    try:
        client = await get_client()
        collections = await client.alist_collections()
        return JSONResponse(content={"collections": collections})
    except Exception as exp:
        logger.exception("Failed to list collections")
        raise HTTPException(status_code=500, detail=str(exp))


@router.get("/{collection_name}")
async def get_collection_by_name(collection_name: str = Path(title="Collection name")):
    """Get the collection config given its name"""
    try:
        client = await get_client()
        collection = await client.aget_collection_by_name(collection_name)
        if collection is None:
            return JSONResponse(content={"collection": []})
        return JSONResponse(content={"collection": collection.model_dump()})
    except HTTPException as exp:
        raise exp
    except Exception as exp:
        logger.exception("Failed to get collection by name")
        raise HTTPException(status_code=500, detail=str(exp))


@router.post("")
async def create_collection(collection: CreateCollectionDto):
    """API to create a collection"""
    try:
        logger.info(f"Creating collection {collection.name}...")
        client = await get_client()
        created_collection = await client.acreate_collection(
            collection=CreateCollection(
                name=collection.name,
                description=collection.description,
                embedder_config=collection.embedder_config,
            )
        )
        logger.info(f"Creating collection {collection.name} on vector db...")
        VECTOR_STORE_CLIENT.create_collection(
            collection_name=collection.name,
            embeddings=model_gateway.get_embedder_from_model_config(
                model_name=collection.embedder_config.model_configuration.name
            ),
        )
        logger.info(f"Created collection... {created_collection}")

        if collection.associated_data_sources:
            for data_source in collection.associated_data_sources:
                await client.aassociate_data_source_with_collection(
                    collection_name=created_collection.name,
                    data_source_association=AssociateDataSourceWithCollection(
                        data_source_fqn=data_source.data_source_fqn,
                        parser_config=data_source.parser_config,
                    ),
                )
            created_collection = await client.aget_collection_by_name(
                collection_name=created_collection.name
            )
        return JSONResponse(
            content={"collection": created_collection.model_dump()}, status_code=201
        )
    except HTTPException as exp:
        raise exp
    except Exception as exp:
        logger.exception(f"Failed to create collection")
        raise HTTPException(status_code=500, detail=str(exp))


@router.post("/associate_data_source")
async def associate_data_source_to_collection(
    request: AssociateDataSourceWithCollectionDto,
):
    """Add a data source to the collection"""
    try:
        client = await get_client()
        collection = await client.aassociate_data_source_with_collection(
            collection_name=request.collection_name,
            data_source_association=AssociateDataSourceWithCollection(
                data_source_fqn=request.data_source_fqn,
                parser_config=request.parser_config,
            ),
        )
        return JSONResponse(content={"collection": collection.model_dump()})
    except HTTPException as exp:
        raise exp
    except Exception as exp:
        logger.exception("Failed to associate data source")
        raise HTTPException(status_code=500, detail=str(exp))


@router.post("/unassociate_data_source")
async def unassociate_data_source_from_collection(
    request: UnassociateDataSourceWithCollectionDto,
):
    """Remove a data source to the collection"""
    try:
        client = await get_client()
        collection = await client.aunassociate_data_source_with_collection(
            collection_name=request.collection_name,
            data_source_fqn=request.data_source_fqn,
        )
        return JSONResponse(content={"collection": collection.model_dump()})
    except HTTPException as exp:
        raise exp
    except Exception as exp:
        logger.exception("Failed to unassociate data source from collection")
        raise HTTPException(status_code=500, detail=str(exp))


@router.post("/ingest")
async def ingest_data(request: IngestDataToCollectionDto):
    """Ingest data into the collection"""
    try:
        return await ingest_data_to_collection(request)
    except HTTPException as exp:
        raise exp
    except Exception as exp:
        logger.exception("Failed to ingest data")
        raise HTTPException(status_code=500, detail=str(exp))


@router.delete("/{collection_name}")
async def delete_collection(collection_name: str = Path(title="Collection name")):
    """Delete collection given its name"""
    try:
        client = await get_client()
        await client.adelete_collection(collection_name, include_runs=True)
        VECTOR_STORE_CLIENT.delete_collection(collection_name=collection_name)
        return JSONResponse(content={"deleted": True})
    except HTTPException as exp:
        raise exp
    except Exception as exp:
        logger.exception("Failed to delete collection")
        raise HTTPException(status_code=500, detail=str(exp))


@router.post("/data_ingestion_runs/list")
async def list_data_ingestion_runs(request: ListDataIngestionRunsDto):
    client = await get_client()
    data_ingestion_runs = await client.aget_data_ingestion_runs(
        request.collection_name, request.data_source_fqn
    )
    return JSONResponse(
        content={
            "data_ingestion_runs": [obj.model_dump() for obj in data_ingestion_runs]
        }
    )


@router.get("/data_ingestion_runs/{data_ingestion_run_name}/status")
async def get_collection_status(
    data_ingestion_run_name: str = Path(title="Data Ingestion Run name"),
):
    """Get status for given data ingestion run"""
    client = await get_client()
    data_ingestion_run = await client.aget_data_ingestion_run(
        data_ingestion_run_name=data_ingestion_run_name, no_cache=True
    )
    if data_ingestion_run is None:
        raise HTTPException(
            status_code=404,
            detail=f"Data ingestion run {data_ingestion_run_name} not found",
        )

    return JSONResponse(
        content={
            "status": data_ingestion_run.status.value,
            "message": f"Data ingestion job run {data_ingestion_run.name} in {data_ingestion_run.status.value}. Check logs for more details.",
        }
    )

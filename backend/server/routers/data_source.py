from urllib.parse import unquote

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from backend.logger import logger
from backend.modules.metadata_store.client import get_client
from backend.types import CreateDataSource

router = APIRouter(prefix="/v1/data_source", tags=["data_source"])


@router.get("")
async def get_data_source():
    """Get data sources"""
    try:
        client = await get_client()
        data_sources = await client.aget_data_sources()
        return JSONResponse(
            content={"data_sources": [obj.model_dump() for obj in data_sources]}
        )
    except Exception as exp:
        logger.exception("Failed to get data source")
        raise HTTPException(status_code=500, detail=str(exp))


@router.get("/list")
async def list_data_sources():
    """Get data sources"""
    try:
        client = await get_client()
        data_sources = await client.alist_data_sources()
        return JSONResponse(content={"data_sources": data_sources})
    except Exception as exp:
        logger.exception("Failed to list data sources")
        raise HTTPException(status_code=500, detail=str(exp))


@router.post("")
async def add_data_source(
    data_source: CreateDataSource,
):
    """Create a data source for the given collection"""
    try:
        client = await get_client()
        created_data_source = await client.acreate_data_source(data_source=data_source)
        return JSONResponse(
            content={"data_source": created_data_source.model_dump()}, status_code=201
        )
    except HTTPException as exp:
        raise exp
    except Exception as exp:
        logger.exception("Failed to add data source")
        raise HTTPException(status_code=500, detail=str(exp))


@router.delete("/delete")
async def delete_data_source(data_source_fqn: str):
    """Delete a data source"""
    decoded_data_source_fqn = unquote(data_source_fqn)
    logger.info(f"Deleting data source: {decoded_data_source_fqn}")
    try:
        client = await get_client()
        await client.adelete_data_source(decoded_data_source_fqn)
        return JSONResponse(content={"deleted": True})
    except HTTPException as exp:
        raise exp
    except Exception as exp:
        logger.exception("Failed to delete data source")
        raise HTTPException(status_code=500, detail=str(exp))

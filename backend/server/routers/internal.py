import os
import uuid
from types import SimpleNamespace
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from truefoundry import ml
from truefoundry.ml import DataDirectory

from backend.logger import logger
from backend.modules.model_gateway.model_gateway import model_gateway
from backend.server.routers.data_source import add_data_source
from backend.settings import settings
from backend.types import CreateDataSource, ModelType, UploadToDataDirectoryDto

router = APIRouter(prefix="/v1/internal", tags=["internal"])


@router.post("/upload-to-local-directory")
async def upload_to_docker_directory(
    upload_name: str = Form(
        default_factory=lambda: str(uuid.uuid4()), regex=r"^[a-z][a-z0-9-]*$"
    ),
    files: List[UploadFile] = File(...),
):
    """This function uploads files within `/app/user_data/` given by the name req.upload_name"""
    if not settings.LOCAL:
        return JSONResponse(
            content={"error": "API only supported for local docker environment"},
            status_code=500,
        )
    try:
        logger.info(f"Uploading files to docker directory: {upload_name}")
        # create a folder within `/volumes/user_data/` that maps to `/app/user_data/` in the docker volume
        # this folder will be used to store the uploaded files
        folder_path = os.path.join("/app/user_data/", upload_name)

        # Create the folder if it does not exist, else raise an exception
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        else:
            return JSONResponse(
                content={"error": f"Folder already exists: {upload_name}"},
                status_code=500,
            )

        # Upload the files to the folder
        for file in files:
            logger.info(f"Copying file: {file.filename}, to folder: {folder_path}")
            file_path = os.path.join(folder_path, file.filename)
            with open(file_path, "wb") as f:
                f.write(file.file.read())

        data_source = CreateDataSource(
            type="localdir",
            uri=folder_path,
        )

        # Add the data source to the metadata store.
        return await add_data_source(data_source)
    except Exception as ex:
        return JSONResponse(
            content={"error": f"Error uploading files to docker directory: {ex}"},
            status_code=500,
        )


@router.post("/upload-to-data-directory")
async def upload_to_data_directory(req: UploadToDataDirectoryDto):
    if settings.METADATA_STORE_CONFIG.provider != "truefoundry":
        raise Exception("API only supported for metadata store provider: truefoundry")
    try:
        truefoundry_client = ml.get_client()

        # Create a new data directory.
        dataset = truefoundry_client.create_data_directory(
            settings.METADATA_STORE_CONFIG.config.get("ml_repo_name"),
            req.upload_name,
        )

        _artifacts_repo = DataDirectory.from_fqn(fqn=dataset.fqn)._get_artifacts_repo()

        urls = _artifacts_repo.get_signed_urls_for_write(
            artifact_identifier=SimpleNamespace(
                artifact_version_id=None, dataset_fqn=dataset.fqn
            ),
            paths=req.filepaths,
        )

        data = [url.model_dump() for url in urls]
        return JSONResponse(
            content={"data": data, "data_directory_fqn": dataset.fqn},
        )
    except Exception as ex:
        raise Exception(f"Error uploading files to data directory: {ex}")


@router.get("/models")
def get_enabled_models(
    model_type: Optional[ModelType] = Query(default=None),
):
    if model_type == ModelType.embedding:
        enabled_models = model_gateway.get_embedding_models()
    elif model_type == ModelType.chat:
        enabled_models = model_gateway.get_llm_models()
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model type: {model_type}",
        )

    # Serialized models
    serialized_models = [model.model_dump() for model in enabled_models]
    return JSONResponse(
        content={"models": serialized_models},
    )

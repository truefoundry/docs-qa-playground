import os

import orjson
from pydantic import BaseSettings

from backend.utils.base import EmbeddingCacheConfig


class Settings(BaseSettings):
    """
    Settings class to hold all the environment variables for FastAPI
    """

    DOCUMENT_ID_SEPARATOR = "::"

    DEBUG_MODE = True if os.getenv("DEBUG_MODE", "false") == "true" else False
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
    ML_REPO_NAME = os.getenv("ML_REPO_NAME")
    VECTOR_DB_CONFIG = os.getenv("VECTOR_DB_CONFIG")
    METADATA_STORE_TYPE = os.getenv("METADATA_STORE_TYPE", "mlfoundry")
    TFY_SERVICE_ROOT_PATH = os.getenv("TFY_SERVICE_ROOT_PATH", "")
    JOB_FQN = os.getenv("JOB_FQN")
    TFY_API_KEY = os.getenv("TFY_API_KEY")
    TFY_HOST = os.getenv("TFY_HOST")
    LLM_GATEWAY_ENDPOINT = os.getenv("LLM_GATEWAY_ENDPOINT")
    EMBEDDING_CACHE_ENABLED = os.getenv("EMBEDDING_CACHE_ENABLED", "false") == "true"
    EMBEDDING_CACHE_CONFIG = (
        EmbeddingCacheConfig.parse_obj(
            orjson.loads(os.getenv("EMBEDDING_CACHE_CONFIG"))
        )
        if os.getenv("EMBEDDING_CACHE_CONFIG", None)
        else None
    )
    if EMBEDDING_CACHE_ENABLED and not EMBEDDING_CACHE_CONFIG:
        raise ValueError("EMBEDDING_CACHE_CONFIG is not set")

    if not ML_REPO_NAME:
        raise ValueError("ML_REPO_NAME is not set")
    # if not VECTOR_DB_CONFIG:
    #     raise ValueError("VECTOR_DB_CONFIG is not set")
    # if not JOB_FQN:
    #     raise ValueError("JOB_FQN is not set")
    if not LLM_GATEWAY_ENDPOINT:
        raise ValueError("LLM_GATEWAY_ENDPOINT is not set")


settings = Settings()

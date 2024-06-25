import os
from typing import Any

from pydantic import ConfigDict, model_validator
from pydantic_settings import BaseSettings

from backend.types import MetadataStoreConfig, VectorDBConfig


class Settings(BaseSettings):
    """
    Settings class to hold all the environment variables
    """

    model_config = ConfigDict(extra="allow")

    MODELS_CONFIG_PATH: str
    METADATA_STORE_CONFIG: MetadataStoreConfig
    VECTOR_DB_CONFIG: VectorDBConfig

    RERANKER_SVC_URL: str = ""
    LOCAL: bool = False

    TFY_HOST: str = ""
    TFY_API_KEY: str = ""
    JOB_FQN: str = ""
    JOB_COMPONENT_NAME: str = ""

    LOG_LEVEL: str = "info"
    TFY_SERVICE_ROOT_PATH: str = ""

    # TODO: This will be removed in future releases
    TFY_LLM_GATEWAY_URL: str = ""

    @model_validator(mode="before")
    @classmethod
    def _validate_values(cls, values: Any) -> Any:
        if isinstance(values, dict):
            models_config_path = values.get("MODELS_CONFIG_PATH")
            if not os.path.isabs(models_config_path):
                this_dir = os.path.abspath(os.path.dirname(__file__))
                root_dir = os.path.dirname(this_dir)
                models_config_path = os.path.join(root_dir, models_config_path)

            if not models_config_path:
                raise Exception(
                    f"{models_config_path} does not exist. "
                    f"You can copy models_config.sample.yaml to {settings.MODELS_CONFIG_PATH} to bootstrap config"
                )

            values["MODELS_CONFIG_PATH"] = models_config_path

            tfy_host = values.get("TFY_HOST")
            tfy_llm_gateway_url = values.get("TFY_LLM_GATEWAY_URL")
            if tfy_host and not tfy_llm_gateway_url:
                tfy_llm_gateway_url = f"{tfy_host.rstrip('/')}/api/llm"
                values["TFY_LLM_GATEWAY_URL"] = tfy_llm_gateway_url
        else:
            # logger has not been initialized at this point, hence the print
            print(
                f"[Validation Skipped] Pydantic v2 validator received "
                f"non dict values of type {type(values)}"
            )

        return values


settings = Settings()

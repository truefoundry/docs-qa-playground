import os
from typing import List

import yaml
from langchain.embeddings.base import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI

from backend.logger import logger
from backend.settings import settings
from backend.types import ModelConfig, ModelProviderConfig, ModelType


class ModelGateway:
    provider_configs: List[ModelProviderConfig]
    model_name_to_provider_config = {}

    def __init__(self):
        logger.info(f"Loading models config from {settings.MODELS_CONFIG_PATH}")
        with open(settings.MODELS_CONFIG_PATH) as f:
            data = yaml.safe_load(f)
        _providers = data.get("model_providers") or []
        # parse the json data into a list of ModelProviderConfig objects
        self.provider_configs = [
            ModelProviderConfig.model_validate(item) for item in _providers
        ]

        # load llm models
        self.llm_models: List[ModelConfig] = []
        # load embedding models
        self.embedding_models: List[ModelConfig] = []

        for provider_config in self.provider_configs:
            if provider_config.api_key_env_var and not os.environ.get(
                provider_config.api_key_env_var
            ):
                raise ValueError(
                    f"Environment variable {provider_config.api_key_env_var} not set. "
                    f"Cannot initialize the model gateway."
                )

            for model_id in provider_config.embedding_model_ids:
                model_name = f"{provider_config.provider_name}/{model_id}"
                self.model_name_to_provider_config[model_name] = provider_config

                # Register the model as an embedding model
                self.embedding_models.append(
                    ModelConfig(
                        name=f"{provider_config.provider_name}/{model_id}",
                        type=ModelType.embedding,
                    )
                )

            for model_id in provider_config.llm_model_ids:
                model_name = f"{provider_config.provider_name}/{model_id}"
                self.model_name_to_provider_config[model_name] = provider_config

                # Register the model as a llm model
                self.llm_models.append(
                    ModelConfig(
                        name=f"{provider_config.provider_name}/{model_id}",
                        type=ModelType.chat,
                    )
                )

    def get_embedding_models(self) -> List[ModelConfig]:
        return self.embedding_models

    def get_llm_models(self) -> List[ModelConfig]:
        return self.llm_models

    def get_embedder_from_model_config(self, model_name: str) -> Embeddings:
        if model_name not in self.model_name_to_provider_config:
            raise ValueError(f"Model {model_name} not registered in the model gateway.")
        model_provider_config: ModelProviderConfig = self.model_name_to_provider_config[
            model_name
        ]
        if not model_provider_config.api_key_env_var:
            api_key = "EMPTY"
        else:
            api_key = os.environ.get(model_provider_config.api_key_env_var, "")
        model_id = "/".join(model_name.split("/")[1:])
        return OpenAIEmbeddings(
            openai_api_key=api_key,
            model=model_id,
            openai_api_base=model_provider_config.base_url,
            check_embedding_ctx_length=(
                model_provider_config.provider_name == "openai"
            ),
        )

    def get_llm_from_model_config(
        self, model_config: ModelConfig, stream=False
    ) -> BaseChatModel:
        if model_config.name not in self.model_name_to_provider_config:
            raise ValueError(
                f"Model {model_config.name} not registered in the model gateway."
            )
        model_provider_config: ModelProviderConfig = self.model_name_to_provider_config[
            model_config.name
        ]
        if not model_config.parameters:
            model_config.parameters = {}
        if not model_provider_config.api_key_env_var:
            api_key = "EMPTY"
        else:
            api_key = os.environ.get(model_provider_config.api_key_env_var, "")
        model_id = "/".join(model_config.name.split("/")[1:])
        return ChatOpenAI(
            model=model_id,
            temperature=model_config.parameters.get("temperature", 0.1),
            streaming=stream,
            api_key=api_key,
            base_url=model_provider_config.base_url,
        )


model_gateway = ModelGateway()

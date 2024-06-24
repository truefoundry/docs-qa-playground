from typing import Any, ClassVar, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from qdrant_client.models import Filter as QdrantFilter

from backend.logger import logger
from backend.types import ModelConfig

GENERATION_TIMEOUT_SEC = 60.0 * 10


class VectorStoreRetrieverConfig(BaseModel):
    """
    Configuration for VectorStore Retriever
    """

    search_type: str = Field(
        default="similarity",
        title="""Defines the type of search that the Retriever should perform. Can be 'similarity' (default), 'mmr', or 'similarity_score_threshold'.
            - "similarity": Retrieve the top k most similar documents to the query.,
            - "mmr": Retrieve the top k most similar documents to the query and then rerank them using Maximal Marginal Relevance (MMR).,
            - "similarity_score_threshold": Retrieve all documents with similarity score greater than a threshold.
        """,
    )

    search_kwargs: dict = Field(default_factory=dict)

    filter: Optional[dict] = Field(
        default_factory=dict,
        title="""Filter by document metadata""",
    )

    allowed_search_types: ClassVar[Sequence[str]] = (
        "similarity",
        "similarity_score_threshold",
        "mmr",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_search_type(cls, values: Any) -> Any:
        """Validate search type."""
        if isinstance(values, dict):
            # TODO (chiragjn): Convert all asserts
            search_type = values.get("search_type")

            assert (
                search_type in cls.allowed_search_types
            ), f"search_type of {search_type} not allowed. Valid values are: {cls.allowed_search_types}"

            search_kwargs = values.get("search_kwargs")

            if search_type == "similarity":
                assert "k" in search_kwargs, "k is required for similarity search"

            elif search_type == "mmr":
                assert (
                    "k" in search_kwargs
                ), "k is required in search_kwargs for mmr search"
                assert (
                    "fetch_k" in search_kwargs
                ), "fetch_k is required in search_kwargs for mmr search"

            elif search_type == "similarity_score_threshold":
                assert (
                    "score_threshold" in search_kwargs
                ), "score_threshold with a float value(0~1) is required in search_kwargs for similarity_score_threshold search"

            filters = values.get("filter")
            if filters:
                search_kwargs["filter"] = QdrantFilter.model_validate(filters)
        else:
            logger.warning(
                f"[Validation Skipped] Pydantic v2 validator received "
                f"non dict values of type {type(values)}"
            )
        return values


class MultiQueryRetrieverConfig(VectorStoreRetrieverConfig):
    retriever_llm_configuration: ModelConfig = Field(
        title="LLM configuration for the retriever",
    )


class ContextualCompressionRetrieverConfig(VectorStoreRetrieverConfig):
    compressor_model_provider: str = Field(
        title="provider of the compressor model",
    )

    compressor_model_name: str = Field(
        title="model name of the compressor",
    )

    top_k: int = Field(
        title="Top K docs to collect post compression",
    )

    allowed_compressor_model_providers: ClassVar[Sequence[str]] = ("mixedbread-ai",)

    @field_validator("compressor_model_provider")
    @classmethod
    def validate_retriever_type(cls, value: str) -> str:
        assert (
            value in cls.allowed_compressor_model_providers
        ), f"Compressor model of {value} not allowed. Valid values are: {cls.allowed_compressor_model_providers}"
        return value


class ContextualCompressionMultiQueryRetrieverConfig(
    ContextualCompressionRetrieverConfig, MultiQueryRetrieverConfig
):
    pass


class LordOfRetrievers(ContextualCompressionRetrieverConfig, MultiQueryRetrieverConfig):
    pass


class ExampleQueryInput(BaseModel):
    """
    Model for Query input.
    Requires a collection name, retriever configuration, query, LLM configuration and prompt template.
    """

    # TODO (chiragjn): This is not the best idea
    model_config = ConfigDict(protected_namespaces=tuple())

    collection_name: str = Field(
        default=None,
        title="Collection name on which to search",
    )

    query: str = Field(title="Question to search for")

    model_configuration: ModelConfig

    prompt_template: str = Field(
        title="Prompt Template to use for generating answer to the question using the context",
    )

    retriever_name: str = Field(
        title="Retriever name",
    )

    retriever_config: dict[str, Any] = Field(
        title="Retriever configuration",
    )

    allowed_retriever_types: ClassVar[Sequence[str]] = (
        "vectorstore",
        "multi-query",
        "contextual-compression",
        "contextual-compression-multi-query",
        "lord-of-the-retrievers",
    )

    stream: bool = Field(default=False, title="Stream the results")

    @model_validator(mode="before")
    @classmethod
    def validate_retriever_type(cls, values: Any) -> Any:
        if isinstance(values, dict):
            retriever_name = values.get("retriever_name")

            assert (
                retriever_name in cls.allowed_retriever_types
            ), f"retriever of {retriever_name} not allowed. Valid values are: {cls.allowed_retriever_types}"

            if retriever_name == "vectorstore":
                values["retriever_config"] = VectorStoreRetrieverConfig(
                    **values.get("retriever_config")
                )

            elif retriever_name == "multi-query":
                values["retriever_config"] = MultiQueryRetrieverConfig(
                    **values.get("retriever_config")
                )

            elif retriever_name == "contextual-compression":
                values["retriever_config"] = ContextualCompressionRetrieverConfig(
                    **values.get("retriever_config")
                )

            elif retriever_name == "contextual-compression-multi-query":
                values[
                    "retriever_config"
                ] = ContextualCompressionMultiQueryRetrieverConfig(
                    **values.get("retriever_config")
                )

            elif retriever_name == "lord-of-the-retrievers":
                values["retriever_config"] = LordOfRetrievers(
                    **values.get("retriever_config")
                )
        else:
            logger.warning(
                f"[Validation Skipped] Pydantic v2 validator received "
                f"non dict values of type {type(values)}"
            )
        return values

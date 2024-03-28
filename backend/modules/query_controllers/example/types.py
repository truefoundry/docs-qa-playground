from pydantic import BaseModel, Field, root_validator, validator
from typing import Literal, Optional, Dict, Collection, ClassVar, Any

from backend.types import LLMConfig

from qdrant_client.models import Filter as QdrantFilter


GENERATION_TIMEOUT_SEC = 60.0 * 5

class CustomLLMConfig(LLMConfig):
    """
    Configuration for LLM Configuration
    """
    # You can add your custom providers too as per usecase
    provider: Literal["openai", "ollama", "truefoundry"] = Field(
        title="Model provider any one between openai, ollama, truefoundry",
    )

class VectorStoreRetrieverConfig(BaseModel):
    """
    Configuration for VectorStore Retriever
    """
    search_type: str = Field(
        default="similarity",
        title=
        """Defines the type of search that the Retriever should perform. Can be 'similarity' (default), 'mmr', or 'similarity_score_threshold'.
            - "similarity": Retrieve the top k most similar documents to the query.,
            - "mmr": Retrieve the top k most similar documents to the query and then rerank them using Maximal Marginal Relevance (MMR).,
            - "similarity_score_threshold": Retrieve all documents with similarity score greater than a threshold.
        """
    )

    search_kwargs: dict = Field(default_factory=dict)

    filter: Optional[dict] = Field(
        default_factory=dict,
        title="""Filter by document metadata""",
    )

    allowed_search_types: ClassVar[Collection[str]] = (
        "similarity",
        "similarity_score_threshold",
        "mmr",
    )

    @root_validator
    def validate_search_type(cls, values: Dict) -> Dict:
        """Validate search type."""
        search_type = values.get("search_type")

        assert search_type in cls.allowed_search_types, f"search_type of {search_type} not allowed. Valid values are: {cls.allowed_search_types}"

        search_kwargs = values.get("search_kwargs")

        if search_type == "similarity":
            assert "k" in search_kwargs, "k is required for similarity search"
            
        elif search_type == "mmr":
            assert "k" in search_kwargs, "k is required in search_kwargs for mmr search"
            assert "fetch_k" in search_kwargs, "fetch_k is required in search_kwargs for mmr search"

        elif search_type == "similarity_score_threshold":
            assert "score_threshold" in search_kwargs, "score_threshold with a float value(0~1) is required in search_kwargs for similarity_score_threshold search"
        
        filters = values.get("filter")
        if filters:
            search_kwargs['filter'] = QdrantFilter.parse_obj(filters)
        return values
    
class MultiQueryRetrieverConfig(VectorStoreRetrieverConfig):
    retriever_llm_configuration : CustomLLMConfig = Field(
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

    allowed_compressor_model_providers: ClassVar[Collection[str]] = (
        "mixbread-ai",
    )

    @validator('compressor_model_provider')
    def validate_retriever_type(cls, value) -> Dict:
        assert value in cls.allowed_compressor_model_providers, f"Compressor model of {value} not allowed. Valid values are: {cls.allowed_compressor_model_providers}"
        return value


class ContextualCompressionMultiQueryRetrieverConfig(ContextualCompressionRetrieverConfig, MultiQueryRetrieverConfig):
    pass

class LordOfRetrievers(ContextualCompressionRetrieverConfig, MultiQueryRetrieverConfig):
    pass

class DefaultQueryInput(BaseModel):
    """
    Model for Query input.
    Requires a collection name, retriever configuration, query, LLM configuration and prompt template.
    """

    collection_name: str = Field(
        default=None,
        title="Collection name on which to search",
    )

    query: str = Field(title="Question to search for", max_length=1000)

    model_configuration: CustomLLMConfig

    prompt_template: str = Field(
        title="Prompt Template to use for generating answer to the question using the context",
    )

    retriever_name: str = Field(
        title="Retriever name",
    )

    retriever_config: Dict[str, Any]= Field(
        title="Retriever configuration",
    )

    allowed_retriever_types: ClassVar[Collection[str]] = (
        "vectorstore",
        "multi-query",
        "contexual-compression",
        "contexual-compression-multi-query",
        "lord-of-the-retrievers"
    )

    stream: Optional[bool] = Field(
        title="Stream the results",
        default=False
    )

    @root_validator()
    def validate_retriever_type(cls, values: Dict) -> Dict:

        retriever_name = values.get("retriever_name")

        assert retriever_name in cls.allowed_retriever_types, f"retriever of {retriever_name} not allowed. Valid values are: {cls.allowed_retriever_types}"

        if retriever_name == "vectorstore":
            values['retriever_config'] = VectorStoreRetrieverConfig(**values.get("retriever_config"))

        elif retriever_name == "multi-query":
            values['retriever_config'] = MultiQueryRetrieverConfig(**values.get("retriever_config"))
        
        elif retriever_name == "contexual-compression":
            values['retriever_config'] = ContextualCompressionRetrieverConfig(**values.get("retriever_config"))

        elif retriever_name == "contexual-compression-multi-query":
            values['retriever_config'] = ContextualCompressionMultiQueryRetrieverConfig(**values.get("retriever_config"))
        
        elif retriever_name == "lord-of-the-retrievers":
            values['retriever_config'] = LordOfRetrievers(**values.get("retriever_config"))
            
        return values
    

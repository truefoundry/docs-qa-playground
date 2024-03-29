from pydantic import BaseModel, Field
from typing import Literal, Optional

from backend.types import LLMConfig, RetrieverConfig

class DefaultLLMConfig(LLMConfig):
    """
    Configuration for LLM Configuration
    """
    # You can add your custom providers too as per usecase
    provider: Optional[Literal["openai", "ollama", "truefoundry"]] = Field(title="Model provider")



class DefaultQueryInput(BaseModel):
    """
    Model for Query input.
    Requires a collection name, retriever configuration, query, LLM configuration and prompt template.
    """

    collection_name: str = Field(
        default=None,
        title="Collection name on which to search",
    )

    retriever_config: RetrieverConfig = Field(
        title="Retriever configuration",
    )
    query: str = Field(title="Question to search for", max_length=1000)
    model_configuration: DefaultLLMConfig
    prompt_template: str = Field(
        title="Prompt Template to use for generating answer to the question using the context",
    )

class RerankerQuery(DefaultQueryInput):
    top_k: int = Field(
        title="Top K docs to collect from reranker", 
        default=5
    )

DEFAULT_QUERY = DefaultQueryInput(
    collection_name="testcollection",
    retriever_config={
        "search_type": "similarity",
        "k": 20,
    },
    query="What are the features of Diners club black metal edition?",
    model_configuration=DefaultLLMConfig(
        name= "openai-devtest/gpt-3-5-turbo",
        provider="truefoundry",
        parameters={"temperature": 0.1}
    ),
    prompt_template="Given the context, answer the question.\n\nContext: {context}\n'''Question: {question}\nAnswer:"
).dict()

DEFAULT_RERANK_QUERY = RerankerQuery(
    collection_name="testcollection",
    retriever_config={
        "search_type": "similarity",
        "k": 20,
    },
    top_k=5,
    query="What are the features of Diners club black metal edition?",
    model_configuration=DefaultLLMConfig(
        name= "openai-devtest/gpt-3-5-turbo",
        provider="truefoundry",
        parameters={"temperature": 0.1}
    ),
    prompt_template="Given the context, answer the question.\n\nContext: {context}\n'''Question: {question}\nAnswer:"
).dict()

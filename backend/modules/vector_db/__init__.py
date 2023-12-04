from backend.modules.vector_db.base import BaseVectorDB
from backend.modules.vector_db.chroma import ChromaVectorDB
from backend.modules.vector_db.qdrant import QdrantVectorDB
from backend.modules.vector_db.weaviate import WeaviateVectorDB
from backend.utils.base import VectorDBConfig

SUPPORTED_VECTOR_DBS = {
    "qdrant": QdrantVectorDB,
    "weaviate": WeaviateVectorDB,
    "chroma": ChromaVectorDB,
}


def get_vector_db_client(
    config: VectorDBConfig, collection_name: str = None
) -> BaseVectorDB:
    if config.__root__.provider in SUPPORTED_VECTOR_DBS:
        return SUPPORTED_VECTOR_DBS[config.__root__.provider](
            config=config.__root__, collection_name=collection_name
        )
    else:
        raise ValueError(f"Unknown vector db provider: {config.__root__.provider}")

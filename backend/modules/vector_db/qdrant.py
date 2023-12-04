import os
from typing import List

from langchain.embeddings.base import Embeddings
from langchain.vectorstores.qdrant import Qdrant
from qdrant_client import QdrantClient, models

from backend.modules.vector_db.base import BaseVectorDB
from backend.utils.base import QdrantDBConfig
from backend.utils.logger import logger


class QdrantVectorDB(BaseVectorDB):
    def __init__(self, config: QdrantDBConfig, collection_name: str = None):
        self.url = config.url
        self.api_key = config.api_key
        self.collection_name = collection_name
        self.qdrant_client = QdrantClient(
            url=self.url, **({"api_key": self.api_key} if self.api_key else {})
        )

    def create_collection(self, embeddings: Embeddings):
        # No provision to create a empty collection
        return

    def upsert_documents(self, documents, embeddings: Embeddings):
        return Qdrant.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=self.collection_name,
            url=self.url,
            api_key=self.api_key,
            prefer_grpc=True,
        )

    def get_collections(self) -> List[str]:
        collections = self.qdrant_client.get_collections().collections
        return [collection.name for collection in collections]

    def delete_collection(self):
        return self.qdrant_client.delete_collection(
            collection_name=self.collection_name
        )

    def get_retriever(self, embeddings: Embeddings, k: int):
        return Qdrant(
            client=self.qdrant_client,
            embeddings=embeddings,
            collection_name=self.collection_name,
        ).as_retriever(search_kwargs={"k": k})

    def list_documents_in_collection(self) -> List[dict]:
        """
        List all documents in a collection
        """
        response = self.qdrant_client.search_groups(
            collection_name=self.collection_name,
            group_by="document_id",
            query_vector=None,
            limit=1000,
            group_size=1,
        )
        groups = response.groups
        documents: List[dict] = []
        for group in groups:
            documents.append(
                {
                    "document_id": group.id,
                }
            )
        return documents

    def delete_documents(self, document_id_match: str):
        """
        Delete a document from the collection
        """
        try:
            self.qdrant_client.get_collection(collection_name=self.collection_name)
        except Exception as exp:
            logger.debug(exp)
            return
        # https://qdrant.tech/documentation/concepts/filtering/#full-text-match
        self.qdrant_client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchText(text=document_id_match),
                        ),
                    ],
                )
            ),
        )

import os
from typing import List

import weaviate
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores.weaviate import Weaviate

from backend.constants import DATA_POINT_FQN_METADATA_KEY, DEFAULT_BATCH_SIZE_FOR_VECTOR_STORE
from backend.modules.vector_db.base import BaseVectorDB
from backend.types import DataPointVector, VectorDBConfig

from backend.logger import logger


def decapitalize(s):
    if not s:
        return s
    return s[0].lower() + s[1:]


class WeaviateVectorDB(BaseVectorDB):
    def __init__(self, config: VectorDBConfig):
        if config.local is True:
            self.url = config.url
            self.api_key = config.api_key
            self.weaviate_client = weaviate.Client(
                url=self.url,
                **(
                    {"auth_client_secret": weaviate.AuthApiKey(api_key=self.api_key)}
                    if self.api_key
                    else {}
                ),
            )

    def create_collection(self, collection_name: str, embeddings: Embeddings):
        self.weaviate_client.schema.create_class(
            {
                "class": collection_name.capitalize(),
                "properties": [
                    {
                        "name": f"{DATA_POINT_FQN_METADATA_KEY}",
                        "dataType": ["text"],
                    },
                ],
            }
        )

    def upsert_documents(
        self,
        collection_name: str,
        documents: List[str],
        embeddings: Embeddings,
        incremental,
    ):
        """
        Upsert documents to the collection
        Arguments:
        - collection_name: Name of the collection
        - documents: List of documents
        - embeddings: Embeddings object
        - incremental: If True, the documents will be added incrementally, otherwise the collection will be replaced
        Returns:
        - None
        """
        # TODO: Add documents in incremental format
        # Ref: upsert_documents from qdrant.py
        Weaviate.from_documents(
            documents=documents,
            embedding=embeddings,
            client=self.weaviate_client,
            index_name=collection_name.capitalize(),
        )

    def get_collections(self) -> List[str]:
        collections = self.weaviate_client.schema.get().get("classes", [])
        return [decapitalize(collection["class"]) for collection in collections]

    def delete_collection(
        self,
        collection_name: str,
    ):
        logger.debug(f"[Weaviate Vector Store] Deleting {collection_name} collection")
        self.weaviate_client.schema.delete_class(collection_name.capitalize())
        logger.debug(f"[Weaviate Vector Store] Deleted {collection_name} collection")

    def get_vector_store(self, collection_name: str, embeddings: Embeddings):
        return Weaviate(
            client=self.weaviate_client,
            embedding=embeddings,
            index_name=collection_name.capitalize(),  # Weaviate stores the index name as capitalized
            text_key="text",
            by_text=False,
            attributes=[f"{DATA_POINT_FQN_METADATA_KEY}"],
        )

    def list_documents_in_collection(
        self, collection_name: str, base_document_id: str = None
    ) -> List[str]:
        """
        List all documents in a collection
        """
        # https://weaviate.io/developers/weaviate/search/aggregate#retrieve-groupedby-properties
        response = (
            self.weaviate_client.query.aggregate(collection_name.capitalize())
            .with_group_by_filter([f"{DATA_POINT_FQN_METADATA_KEY}"])
            .with_fields("groupedBy { value }")
            .do()
        )
        groups: List[dict] = (
            response.get("data", {})
            .get("Aggregate", {})
            .get(collection_name.capitalize(), [])
        )
        document_ids = set()
        for group in groups:
            document_ids.add(group.get("groupedBy", {}).get("value", ""))
        return document_ids

    def delete_documents(self, collection_name: str, document_ids: List[str]):
        """
        Delete documents from the collection that match given `document_id_match`
        """
        # https://weaviate.io/developers/weaviate/manage-data/delete#delete-multiple-objects
        res = self.weaviate_client.batch.delete_objects(
            class_name=collection_name.capitalize(),
            where={
                "path": [f"{DATA_POINT_FQN_METADATA_KEY}"],
                "operator": "ContainsAny",
                "valueTextArray": document_ids,
            },
        )
        deleted_vectors = res.get("results", {}).get("successful", None)
        if deleted_vectors:
            print(f"Deleted {len(document_ids)} documents from the collection")

    def get_vector_client(self):
        return self.weaviate_client
    
    def list_data_point_vectors(
        self,
        collection_name: str,
        data_source_fqn: str,
        batch_size: int = 1000,
    ) -> List[DataPointVector]:
        pass
    
    def delete_data_point_vectors(
            self, 
            collection_name: str, 
            data_point_vectors: List[DataPointVector], 
            batch_size: int = 100):
        pass

import os
from typing import List

import weaviate
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.weaviate import Weaviate

from backend.modules.vector_db.base import BaseVectorDB
from backend.utils.base import WeaviateDBConfig


def decapitalize(s):
    if not s:
        return s
    return s[0].lower() + s[1:]


class WeaviateVectorDB(BaseVectorDB):
    def __init__(self, config: WeaviateDBConfig, collection_name: str = None):
        self.url = config.url
        self.api_key = config.api_key
        if collection_name:
            self.collection_name = collection_name.capitalize()
        self.weaviate_client = weaviate.Client(
            url=self.url,
            **(
                {"auth_client_secret": weaviate.AuthApiKey(api_key=self.api_key)}
                if self.api_key
                else {}
            ),
        )

    def create_collection(self, embeddings: Embeddings):
        self.weaviate_client.schema.create_class(
            {
                "class": self.collection_name,
                "properties": [
                    {
                        "name": "text",
                        "dataType": ["text"],
                    },
                    {
                        "name": "document_id",
                        "dataType": ["text"],
                    },
                ],
            }
        )

    def upsert_documents(self, documents: List[str], embeddings: Embeddings):
        return Weaviate.from_documents(
            documents=documents,
            embedding=embeddings,
            client=self.weaviate_client,
            index_name=self.collection_name,
        )

    def get_collections(self) -> List[str]:
        collections = self.weaviate_client.schema.get().get("classes", [])
        return [decapitalize(collection["class"]) for collection in collections]

    def delete_collection(self):
        return self.weaviate_client.schema.delete_class(self.collection_name)

    def get_retriever(self, embeddings: Embeddings, k: int):
        return Weaviate(
            client=self.weaviate_client,
            embedding=embeddings,
            index_name=self.collection_name,  # Weaviate stores the index name as capitalized
            text_key="text",
            by_text=False,
            attributes=["document_id"],
        ).as_retriever(search_kwargs={"k": k})

    def list_documents_in_collection(self) -> List[dict]:
        """
        List all documents in a collection
        """
        # https://weaviate.io/developers/weaviate/search/aggregate#retrieve-groupedby-properties
        response = (
            self.weaviate_client.query.aggregate(self.collection_name)
            .with_group_by_filter(["document_id"])
            .with_fields("groupedBy { value }")
            .do()
        )
        groups: List[dict] = (
            response.get("data", {}).get("Aggregate", {}).get(self.collection_name, [])
        )
        documents: List[dict] = []
        for group in groups:
            documents.append(
                {
                    "document_id": group.get("groupedBy", {}).get("value", ""),
                }
            )
        return documents

    def delete_documents(self, document_id_match: str):
        """
        Delete documents from the collection that match given `document_id_match`
        """
        # https://weaviate.io/developers/weaviate/manage-data/delete#delete-multiple-objects
        res = self.weaviate_client.batch.delete_objects(
            class_name=self.collection_name,
            where={
                "path": ["document_id"],
                "operator": "Like",
                "valueText": document_id_match,
            },
        )
        deleted_vectors = res.get("results", {}).get("successful", None)
        if deleted_vectors:
            print(
                f"Deleted {deleted_vectors} documents from the collection that match {document_id_match}"
            )

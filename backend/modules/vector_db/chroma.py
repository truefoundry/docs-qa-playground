from typing import List

import chromadb
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.chroma import Chroma

from backend.modules.vector_db.base import BaseVectorDB
from backend.utils.base import ChromaDBConfig


class ChromaVectorDB(BaseVectorDB):
    def __init__(self, config: ChromaDBConfig, collection_name: str = None):
        self.collection_name = collection_name
        self.client = chromadb.Client()

    def create_collection(self, embeddings: Embeddings):
        self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=None,
        )

    def upsert_documents(self, documents, embeddings: Embeddings):
        return Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=self.collection_name,
            client=self.client,
        )

    def get_collections(self) -> List[str]:
        collections = self.client.list_collections()
        return [collection.name for collection in collections]

    def delete_collection(self):
        return self.client.delete_collection(name=self.collection_name)

    def get_retriever(self, embeddings: Embeddings, k: int):
        return Chroma(
            client=self.client,
            embedding_function=embeddings,
            collection_name=self.collection_name,
        ).as_retriever(search_kwargs={"k": k})

    def list_documents_in_collection(self) -> List[dict]:
        return []

    def delete_documents(self, document_id_match: str):
        pass

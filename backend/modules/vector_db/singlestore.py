import json
from typing import Any, Iterable, List, Optional

import singlestoredb as s2
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores.singlestoredb import SingleStoreDB

from backend.constants import DATA_POINT_FQN_METADATA_KEY, DATA_POINT_HASH_METADATA_KEY
from backend.logger import logger
from backend.modules.vector_db.base import BaseVectorDB
from backend.types import DataPointVector, VectorDBConfig

MAX_SCROLL_LIMIT = int(1e6)
BATCH_SIZE = 1000


class SSDB(SingleStoreDB):
    def _create_table(self: SingleStoreDB) -> None:
        """Create table if it doesn't exist."""
        conn = self.connection_pool.connect()
        try:
            cur = conn.cursor()
            # Overriding the default table creation behaviour this adds id as autoinc primary key
            try:
                if self.use_vector_index:
                    index_options = ""
                    if self.vector_index_options and len(self.vector_index_options) > 0:
                        index_options = "INDEX_OPTIONS '{}'".format(
                            json.dumps(self.vector_index_options)
                        )
                    cur.execute(
                        """CREATE TABLE IF NOT EXISTS {}
                        (id BIGINT AUTO_INCREMENT PRIMARY KEY, {} TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci,
                        {} VECTOR({}, F32) NOT NULL, {} JSON,
                        VECTOR INDEX {} ({}) {});""".format(
                            self.table_name,
                            self.content_field,
                            self.vector_field,
                            self.vector_size,
                            self.metadata_field,
                            self.vector_index_name,
                            self.vector_field,
                            index_options,
                        ),
                    )
                else:
                    cur.execute(
                        """CREATE TABLE IF NOT EXISTS {}
                        (id BIGINT AUTO_INCREMENT PRIMARY KEY, {} TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci,
                        {} BLOB, {} JSON);""".format(
                            self.table_name,
                            self.content_field,
                            self.vector_field,
                            self.metadata_field,
                        ),
                    )
            finally:
                cur.close()
        finally:
            conn.close()

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        embeddings: Optional[List[List[float]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add more texts to the vectorstore.

        Args:
            texts (Iterable[str]): Iterable of strings/text to add to the vectorstore.
            metadatas (Optional[List[dict]], optional): Optional list of metadatas.
                Defaults to None.
            embeddings (Optional[List[List[float]]], optional): Optional pre-generated
                embeddings. Defaults to None.

        Returns:
            List[str]: empty list
        """
        conn = self.connection_pool.connect()
        try:
            cur = conn.cursor()
            try:
                # Write data to singlestore db
                for i, text in enumerate(texts):
                    # Use provided values by default or fallback
                    metadata = metadatas[i] if metadatas else {}
                    embedding = (
                        embeddings[i]
                        if embeddings
                        else self.embedding.embed_documents([text])[0]
                    )
                    # Overriding insert statement to handle autoincrement id
                    cur.execute(
                        "INSERT INTO {} (content, vector, metadata) VALUES (%s, JSON_ARRAY_PACK(%s), %s)".format(
                            self.table_name
                        ),
                        (
                            text,
                            "[{}]".format(",".join(map(str, embedding))),
                            json.dumps(metadata),
                        ),
                    )
                if self.use_vector_index:
                    cur.execute("OPTIMIZE TABLE {} FLUSH;".format(self.table_name))
            finally:
                cur.close()
        finally:
            conn.close()
        return []


class SingleStoreVectorDB(BaseVectorDB):
    def __init__(self, config: VectorDBConfig):
        self.host = config.url

    def create_collection(self, collection_name: str, embeddings: Embeddings):
        logger.debug(f"[SingleStore] Creating new collection {collection_name}...")

        # Calculate embedding size
        partial_embeddings = embeddings.embed_documents(["Initial document"])
        vector_size = len(partial_embeddings[0])

        # Create collection
        # we keep vector_filed, content_field as default values
        SSDB(
            embedding=embeddings,
            host=self.host,
            table_name=collection_name,
            vector_size=vector_size,
            use_vector_index=True,
            # metadata_field=f"metadata.{DATA_POINT_FQN_METADATA_KEY}",
        )

        logger.debug(f"[SingleStore] Created new collection {collection_name}")

    def upsert_documents(
        self,
        collection_name: str,
        documents: List[Document],
        embeddings: Embeddings,
        incremental: bool = True,
    ):
        if len(documents) == 0:
            logger.warning("No documents to index")
            return
        # get record IDs to be upserted
        logger.debug(
            f"[SingleStore] Adding {len(documents)} documents to collection {collection_name}"
        )
        data_point_fqns = []
        for document in documents:
            if document.metadata.get(DATA_POINT_FQN_METADATA_KEY):
                data_point_fqns.append(
                    document.metadata.get(DATA_POINT_FQN_METADATA_KEY)
                )

        try:
            SSDB.from_documents(
                embedding=embeddings,
                documents=documents,
                table_name=collection_name,
                host=self.host,
            )
            logger.debug(
                f"[SingleStore] Added {len(documents)} documents to collection {collection_name}"
            )
        except Exception as e:
            logger.exception(
                f"[SingleStore] Failed to add documents to collection {collection_name}: {e}"
            )

    def get_collections(self) -> List[str]:
        conn = s2.connect(self.host)
        try:
            cur = conn.cursor()
            try:
                cur.execute("SHOW TABLES")
                return [row[0] for row in cur.fetchall()]
            except Exception as e:
                logger.exception(f"[SingleStore] Failed to get collections: {e}")
            finally:
                cur.close()
        except Exception as e:
            logger.exception(f"[SingleStore] Failed to get collections: {e}")
        finally:
            conn.close()

    def delete_collection(self, collection_name: str):
        conn = s2.connect(self.host)
        try:
            cur = conn.cursor()
            try:
                cur.execute(f"DROP TABLE {collection_name}")
                logger.debug(f"[SingleStore] Deleted collection {collection_name}")
            except Exception as e:
                logger.exception(
                    f"[SingleStore] Failed to delete collection {collection_name}: {e}"
                )
            finally:
                cur.close()
        except Exception as e:
            logger.exception(
                f"[SingleStore] Failed to delete collection {collection_name}: {e}"
            )
        finally:
            conn.close()

    def get_vector_store(self, collection_name: str, embeddings: Embeddings):
        return SSDB(
            embedding=embeddings,
            host=self.host,
            table_name=collection_name,
        )

    def get_vector_client(self):
        return s2.connect(self.host)

    def list_data_point_vectors(
        self,
        collection_name: str,
        data_source_fqn: str,
        batch_size: int = BATCH_SIZE,
    ) -> List[DataPointVector]:
        logger.debug(
            f"[SingleStore] Listing all data point vectors for collection {collection_name}"
        )
        data_point_vectors: List[DataPointVector] = []
        logger.debug(f"data_source_fqn: {data_source_fqn}")

        # Get all data point vectors from table upto MAX_SCROLL_LIMIT
        conn = s2.connect(self.host)
        try:
            curr = conn.cursor()

            # Remove all data point vectors with the same data_source_fqn
            curr.execute(
                f"SELECT * FROM {collection_name} WHERE JSON_EXTRACT_JSON(metadata, '{DATA_POINT_FQN_METADATA_KEY}') LIKE '%{data_source_fqn}%' LIMIT {MAX_SCROLL_LIMIT}"
            )

            for record in curr:
                # id, content, vector, metadata
                id, _, _, metadata = record
                if (
                    metadata
                    and metadata.get(DATA_POINT_FQN_METADATA_KEY)
                    and metadata.get(DATA_POINT_HASH_METADATA_KEY)
                ):
                    data_point_vectors.append(
                        DataPointVector(
                            data_point_vector_id=id,
                            data_point_fqn=metadata.get(DATA_POINT_FQN_METADATA_KEY),
                            data_point_hash=metadata.get(DATA_POINT_HASH_METADATA_KEY),
                        )
                    )
        except Exception as e:
            logger.exception(f"[SingleStore] Failed to list data point vectors: {e}")
        finally:
            conn.close()

        logger.debug(
            f"[SingleStore] Listing {len(data_point_vectors)} data point vectors for collection {collection_name}"
        )
        return data_point_vectors

    def delete_data_point_vectors(
        self,
        collection_name: str,
        data_point_vectors: List[DataPointVector],
        batch_size: int = BATCH_SIZE,
    ):
        """
        Delete data point vectors from the collection
        """
        logger.debug(
            f"[SingleStore] Deleting {len(data_point_vectors)} data point vectors"
        )

        if len(data_point_vectors) > 0:
            # Delete data point vectors from table
            conn = s2.connect(self.host)

            try:
                vectors_to_be_deleted_count = len(data_point_vectors)
                curr = conn.cursor()

                curr.execute(
                    f"DELETE FROM {collection_name} WHERE id in ({', '.join(data_point_vector.data_point_vector_id for data_point_vector in data_point_vectors)})"
                )
                logger.debug(
                    f"[SingleStore] Deleted {vectors_to_be_deleted_count} data point vectors"
                )
            except Exception as e:
                logger.exception(
                    f"[SingleStore] Failed to delete data point vectors: {e}"
                )
            finally:
                conn.close()

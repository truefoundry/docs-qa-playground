import asyncio
import json

import async_timeout
from fastapi import Body, HTTPException
from fastapi.responses import StreamingResponse
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever, MultiQueryRetriever
from langchain.schema.vectorstore import VectorStoreRetriever
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from backend.logger import logger
from backend.modules.metadata_store.client import get_client
from backend.modules.model_gateway.model_gateway import model_gateway
from backend.modules.query_controllers.multimodal.payload import (
    PROMPT,
    QUERY_WITH_CONTEXTUAL_COMPRESSION_MULTI_QUERY_RETRIEVER_SIMILARITY_PAYLOAD,
    QUERY_WITH_CONTEXTUAL_COMPRESSION_RETRIEVER_PAYLOAD,
    QUERY_WITH_VECTOR_STORE_RETRIEVER_PAYLOAD,
)
from backend.modules.query_controllers.multimodal.types import (
    GENERATION_TIMEOUT_SEC,
    ExampleQueryInput,
)
from backend.modules.vector_db.client import VECTOR_STORE_CLIENT
from backend.server.decorators import post, query_controller
from backend.types import Collection, ModelConfig

EXAMPLES = {
    "vector-store-similarity": QUERY_WITH_VECTOR_STORE_RETRIEVER_PAYLOAD,
    "contextual-compression-similarity": QUERY_WITH_CONTEXTUAL_COMPRESSION_RETRIEVER_PAYLOAD,
    "contextual-compression-multi-query-similarity": QUERY_WITH_CONTEXTUAL_COMPRESSION_MULTI_QUERY_RETRIEVER_SIMILARITY_PAYLOAD,
}


@query_controller("/multimodal-rag")
class MultiModalRAGQueryController:
    def _get_prompt_template(self, input_variables, template):
        """
        Get the prompt template
        """
        return PromptTemplate(input_variables=input_variables, template=template)

    def _format_docs(self, docs):
        formatted_docs = list()
        for doc in docs:
            doc.metadata.pop("image_b64", None)
            formatted_docs.append(
                {"page_content": doc.page_content, "metadata": doc.metadata}
            )
        return "\n\n".join([f"{doc['page_content']}" for doc in formatted_docs])

    def _format_docs_for_stream(self, docs):
        formatted_docs = list()
        for doc in docs:
            doc.metadata.pop("image_b64", None)
            formatted_docs.append(
                {"page_content": doc.page_content, "metadata": doc.metadata}
            )
        return formatted_docs

    def _get_llm(self, model_configuration: ModelConfig, stream=False) -> BaseChatModel:
        """
        Get the LLM
        """
        return model_gateway.get_llm_from_model_config(model_configuration)

    async def _get_vector_store(self, collection_name: str):
        """
        Get the vector store for the collection
        """
        client = await get_client()
        collection = await client.aget_retrieve_collection_by_name(collection_name)
        if collection is None:
            raise HTTPException(status_code=404, detail="Collection not found")

        if not isinstance(collection, Collection):
            collection = Collection(**collection.dict())

        return VECTOR_STORE_CLIENT.get_vector_store(
            collection_name=collection.name,
            embeddings=model_gateway.get_embedder_from_model_config(
                model_name=collection.embedder_config.model_config.name
            ),
        )

    def _get_vector_store_retriever(self, vector_store, retriever_config):
        """
        Get the vector store retriever
        """
        return VectorStoreRetriever(
            vectorstore=vector_store,
            search_type=retriever_config.search_type,
            search_kwargs=retriever_config.search_kwargs,
        )

    def _get_contextual_compression_retriever(self, vector_store, retriever_config):
        """
        Get the contextual compression retriever
        """
        try:
            retriever = self._get_vector_store_retriever(vector_store, retriever_config)
            logger.info("Using MxBaiRerankerSmall th' service...")

            compressor = model_gateway.get_reranker_from_model_config(
                model_name=retriever_config.compressor_model_name,
                top_k=retriever_config.top_k,
            )
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=retriever
            )

            return compression_retriever
        except Exception as e:
            logger.exception(f"Error in getting contextual compression retriever: {e}")
            raise HTTPException(
                status_code=500,
                detail="Error in getting contextual compression retriever",
            )

    def _get_multi_query_retriever(
        self, vector_store, retriever_config, retriever_type="vectorstore"
    ):
        """
        Get the multi query retriever
        """
        if retriever_type == "vectorstore":
            base_retriever = self._get_vector_store_retriever(
                vector_store, retriever_config
            )
        elif retriever_type == "contextual-compression":
            base_retriever = self._get_contextual_compression_retriever(
                vector_store, retriever_config
            )
        else:
            raise ValueError(f"Unknown retriever_type {retriever_type}")

        return MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=self._get_llm(retriever_config.retriever_llm_configuration),
        )

    async def _get_retriever(self, vector_store, retriever_name, retriever_config):
        """
        Get the retriever
        """
        if retriever_name == "vectorstore":
            logger.debug(
                f"Using VectorStoreRetriever with {retriever_config.search_type} search"
            )
            retriever = self._get_vector_store_retriever(vector_store, retriever_config)

        elif retriever_name == "contextual-compression":
            logger.debug(
                f"Using ContextualCompressionRetriever with {retriever_config.search_type} search"
            )
            retriever = self._get_contextual_compression_retriever(
                vector_store, retriever_config
            )

        elif retriever_name == "multi-query":
            logger.debug(
                f"Using MultiQueryRetriever with {retriever_config.search_type} search"
            )
            retriever = self._get_multi_query_retriever(vector_store, retriever_config)

        elif retriever_name == "contextual-compression-multi-query":
            logger.debug(
                f"Using MultiQueryRetriever with {retriever_config.search_type} search and "
                f"retriever type as {retriever_name}"
            )
            retriever = self._get_multi_query_retriever(
                vector_store, retriever_config, retriever_type="contextual-compression"
            )

        else:
            raise HTTPException(status_code=404, detail="Retriever not found")

        return retriever

    async def _stream_answer(self, rag_chain, query):
        async with async_timeout.timeout(GENERATION_TIMEOUT_SEC):
            try:
                async for chunk in rag_chain.astream(query):
                    if "question " in chunk:
                        # print("Question: ", chunk['question'])
                        yield json.dumps({"question": chunk["question"]})
                        await asyncio.sleep(0.1)
                    elif "context" in chunk:
                        # print("Context: ", self._format_docs_for_stream(chunk['context']))
                        yield json.dumps(
                            {"docs": self._format_docs_for_stream(chunk["context"])}
                        )
                        await asyncio.sleep(0.1)
                    elif "answer" in chunk:
                        # print("Answer: ", chunk['answer'])
                        yield json.dumps({"answer": chunk["answer"]})
                        await asyncio.sleep(0.1)

                yield json.dumps({"end": "<END>"})
            except asyncio.TimeoutError:
                raise HTTPException(status_code=504, detail="Stream timed out")

    async def _stream_vlm_answer(self, llm, message_payload, docs):
        try:
            async with async_timeout.timeout(GENERATION_TIMEOUT_SEC):
                yield json.dumps(
                    {
                        "docs": self._format_docs_for_stream(docs),
                    }
                )
                await asyncio.sleep(0.1)

                async for chunk in llm.astream(message_payload):
                    yield json.dumps({"answer": chunk.content})
                    await asyncio.sleep(0.1)

                await asyncio.sleep(0.1)
                yield json.dumps({"end": "<END>"})
                await asyncio.sleep(0.1)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Stream timed out")

    def _generate_payload_for_vlm(self, prompt: str, images_set: set):
        content = [
            {
                "type": "text",
                "text": prompt,
            }
        ]

        for b64_image in images_set:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64_image}",
                        "detail": "high",
                    },
                }
            )
        return [HumanMessage(content=content)]

    @post("/answer")
    async def answer(
        self,
        request: ExampleQueryInput = Body(
            openapi_examples=EXAMPLES,
        ),
    ):
        """
        Sample answer method to answer the question using the context from the collection
        """
        try:
            # Get the vector store
            vector_store = await self._get_vector_store(request.collection_name)

            # get retriever
            retriever = await self._get_retriever(
                vector_store=vector_store,
                retriever_name=request.retriever_name,
                retriever_config=request.retriever_config,
            )
            llm = self._get_llm(request.model_configuration, request.stream)

            try:
                prompt = request.prompt_template.format(question=request.query)
            except Exception as e:
                logger.error(f"Error in formatting prompt: {e}")
                logger.info(f"Using default prompt")
                prompt = PROMPT.format(question=request.query)

            # Generate payload for VLM
            images_set = set()

            setup_and_retrieval = RunnableParallel(
                {"context": retriever, "question": RunnablePassthrough()}
            )
            outputs = await setup_and_retrieval.ainvoke(request.query)

            if "context" in outputs:
                docs = outputs["context"]
                for doc in docs:
                    image_b64 = doc.metadata.get("image_b64", None)
                    if image_b64 is not None:
                        images_set.add(image_b64)
                        # Remove the image_b64 from the metadata
                        doc.metadata.pop("image_b64")

            message_payload = self._generate_payload_for_vlm(
                prompt=prompt, images_set=images_set
            )

            if request.stream:
                return StreamingResponse(
                    self._stream_vlm_answer(llm, message_payload, outputs["context"]),
                    media_type="text/event-stream",
                )

            else:
                response = await llm.ainvoke(message_payload)
                return {
                    "answer": response.content,
                    "docs": outputs["context"],
                }

        except HTTPException as exp:
            raise exp
        except Exception as exp:
            logger.exception(exp)
            raise HTTPException(status_code=500, detail=str(exp))

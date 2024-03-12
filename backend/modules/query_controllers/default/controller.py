from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.schema.vectorstore import VectorStoreRetriever
from langchain_community.chat_models.openai import ChatOpenAI

from backend.logger import logger
from backend.modules.embedder.embedder import get_embedder
from backend.modules.metadata_store.client import METADATA_STORE_CLIENT
from backend.modules.query_controllers.default.types import DefaultQueryInput
from backend.modules.vector_db.client import VECTOR_STORE_CLIENT
from backend.server.decorators import post, query_controller
from backend.settings import settings

from typing import AsyncIterator
from pydantic import BaseModel
import json


@query_controller()
class DefaultQueryController:
    """
    Default Query Controller
    uses langchain retrieval qa to answer the query
    """

    @post("/answer")
    async def answer(self, request: DefaultQueryInput):
        """
        Sample answer method to answer the question using the context from the collection
        """
        try:
            # Get the vector store
            collection = METADATA_STORE_CLIENT.get_collection_by_name(
                request.collection_name
            )

            if collection is None:
                raise HTTPException(status_code=404, detail="Collection not found")

            vector_store = VECTOR_STORE_CLIENT.get_vector_store(
                collection_name=collection.name,
                embeddings=get_embedder(collection.embedder_config),
            )

            # Get the LLM
            llm = ChatOpenAI(
                model=request.model_configuration.name,
                api_key=settings.TFY_API_KEY,
                base_url=f"{settings.TFY_LLM_GATEWAY_URL}/openai",
            )

            # Create the retriever using langchain VectorStoreRetriever
            retriever = VectorStoreRetriever(
                vectorstore=vector_store,
                search_type=request.retriever_config.get_search_type,
                search_kwargs=request.retriever_config.get_search_kwargs,
            )
            DOCUMENT_PROMPT = PromptTemplate(
                input_variables=["page_content"],
                template="<document>{page_content}</document>",
            )
            QA_PROMPT = PromptTemplate(
                input_variables=["context", "question"],
                template=request.prompt_template,
            )

            # Create the QA chain
            qa = RetrievalQA(
                retriever=retriever,
                combine_documents_chain=load_qa_chain(
                    llm=llm,
                    chain_type="stuff",
                    prompt=QA_PROMPT,
                    document_variable_name="context",
                    document_prompt=DOCUMENT_PROMPT,
                    verbose=True,
                ),
                return_source_documents=True,
                verbose=True,
            )

            # Get the answer
            logger.info(f"Request query: {request.query}")
            outputs = await qa.ainvoke({"query": request.query})

            return {
                "answer": outputs["result"],
                "docs": outputs.get("source_documents") or [],
            }
        except HTTPException as exp:
            raise exp
        except Exception as exp:
            logger.exception(exp)
            raise HTTPException(status_code=500, detail=str(exp))
        
    
    async def stream_answer(self, request: DefaultQueryInput) -> AsyncIterator[BaseModel]:
        """
        Sample answer method to `stream` the answer the question using the context from the collection
        """
        try:
            # Get the vector store
            collection = METADATA_STORE_CLIENT.get_collection_by_name(
                request.collection_name
            )

            if collection is None:
                raise HTTPException(status_code=404, detail="Collection not found")

            vector_store = VECTOR_STORE_CLIENT.get_vector_store(
                collection_name=collection.name,
                embeddings=get_embedder(collection.embedder_config),
            )

            # Get the LLM
            llm = ChatOpenAI(
                model=request.model_configuration.name,
                api_key=settings.TFY_API_KEY,
                base_url=f"{settings.TFY_LLM_GATEWAY_URL}/openai",
                # streaming=True,
            )

            # Create the retriever using langchain VectorStoreRetriever
            retriever = VectorStoreRetriever(
                vectorstore=vector_store,
                search_type=request.retriever_config.get_search_type,
                search_kwargs=request.retriever_config.get_search_kwargs,
            )
            DOCUMENT_PROMPT = PromptTemplate(
                input_variables=["page_content"],
                template="<document>{page_content}</document>",
            )
            QA_PROMPT = PromptTemplate(
                input_variables=["context", "question"],
                template=request.prompt_template,
            )

            # Create the QA chain
            qa = RetrievalQA(
                retriever=retriever,
                combine_documents_chain=load_qa_chain(
                    llm=llm,
                    chain_type="stuff",
                    prompt=QA_PROMPT,
                    document_variable_name="context",
                    document_prompt=DOCUMENT_PROMPT,
                    verbose=True,
                ),
                return_source_documents=False,
                verbose=True,
            )

            # Get the answer
            logger.info(f"Request query: {request.query}")

            async for chunk in qa.astream({"query": request.query}):
                print(chunk)
                yield chunk['result']

            yield "event end"
                
        except HTTPException as exp:
            raise exp
        except Exception as exp:
            logger.exception(exp)
            raise HTTPException(status_code=500, detail=str(exp))

    @post("/stream")
    async def stream(self, request: DefaultQueryInput):
        """
        Sample answer method to `stream` the answer the question using the context from the collection
        """
        return StreamingResponse(
                self.stream_answer(request),
                media_type="text/event-stream",
            )
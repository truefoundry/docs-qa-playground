from langchain.embeddings.base import Embeddings
from typing import Dict, List

import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from backend.logger import logger
from tqdm.auto import tqdm

# https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1
class MixBreadEmbeddings(Embeddings):

    def __init__(self, model) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model).to(self.device)
        self.embedding_ctx_length = 512
        self.chunk_size = 4

    
    # The model works really well with cls pooling (default) but also with mean poolin.
    def pooling(self, outputs: torch.Tensor, inputs: Dict,  strategy: str = 'cls') -> np.ndarray:
        if strategy == 'cls':
            outputs = outputs[:, 0]
        elif strategy == 'mean':
            outputs = torch.sum(
                outputs * inputs["attention_mask"][:, :, None], dim=1) / torch.sum(inputs["attention_mask"])
        else:
            raise NotImplementedError
        return outputs.detach().cpu().numpy()
    
    # For retrieval you need to pass this prompt.
    # https://www.mixedbread.ai/blog/mxbai-embed-large-v1
    def transform_query(self, query: str) -> str:
        """ For retrieval, add the prompt for query (not for documents).
        """
        return f'Represent this sentence for searching relevant passages: {query}'

    def _get_len_safe_embeddings(self, texts: str) -> List[List[float]]:
        """
        Generate length-safe embeddings for a list of texts.

        This method handles tokenization and embedding generation, 
        respecting the set embedding context length and chunk size.
        """

        tokens = []
        indices = []

        for i, text in enumerate(texts):
            # Tokenize the text using HuggingFace transformers
            tokenized = self.tokenizer.encode(text, add_special_tokens=False)

            # Split tokens into chunks respecting the embedding_ctx_length
            for j in range(0, len(tokenized), self.embedding_ctx_length):
                token_chunk = tokenized[j : j + self.embedding_ctx_length]

                # Convert token IDs back to a string
                chunk_text = self.tokenizer.decode(token_chunk)
                tokens.append(chunk_text)
                indices.append(i)
        
        _iter = tqdm(range(0, len(tokens), self.chunk_size))

        batched_embeddings: List[List[float]] = []
        for i in _iter:
            batch = tokens[i : i + self.chunk_size]
            inputs = self.tokenizer(batch, padding=True, return_tensors='pt', truncation=True)
            outputs = self.model(**inputs)
            embeddings = self.pooling(outputs.last_hidden_state, inputs)
            batched_embeddings.extend(embeddings.tolist())

        return batched_embeddings


    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        return self._get_len_safe_embeddings(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.embed_documents([self.transform_query(text)])[0]

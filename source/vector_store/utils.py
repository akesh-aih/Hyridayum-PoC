from typing import List
from aih_rag.vector_stores.weaviate import WeaviateVectorStore
import tqdm
from aih_rag.schema import Document, TextNode
import asyncio
from .embedding import query_text_embedding
from aih_rag.embeddings import AzureOpenAIEmbedding

import os
from dotenv import load_dotenv; load_dotenv()
azure_api_key = os.getenv("API_Key")
azure_endpoint = os.getenv("End_point")
azure_api_version = os.getenv("API_version")
azure_embedding = AzureOpenAIEmbedding(
    model="text-embedding-3-small",
    azure_endpoint=azure_endpoint,
    api_key=azure_api_key,
    api_version="2024-02-01",
    azure_deployment="text-embedding-3-small",
)
async def create_node(chunk):
    embedding = await query_text_embedding(text=chunk, model=azure_embedding)
    # text=chunks[index], model=azure_embedding
    #         )
    node = TextNode(
        text=chunk,
        embedding=embedding,
    )
    return node


async def async_create_nodes(chunks) -> list[TextNode]:
    tasks = [create_node(chunk) for chunk in chunks]
    nodes = await asyncio.gather(*tasks)
    return nodes
def chunk_text(text, chunk_size=1000, overlap=100):
    """
    Splits text into chunks with a specific overlap.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap  # Move to the next chunk with overlap
    return chunks

def chunk_text_with_overlap(text, max_tokens=100, overlap=20):
    """
    Splits text into chunks with overlapping tokens.
    
    Args:
        text (str): The input text to be chunked.
        max_tokens (int): The maximum number of tokens per chunk.
        overlap (int): The number of overlapping tokens between chunks.

    Returns:
        List[str]: List of overlapping chunks.
    """
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), max_tokens - overlap):
        chunk = " ".join(words[i:i + max_tokens])
        chunks.append(chunk)
        
        # Stop if the chunk reaches the end of the text
        if i + max_tokens >= len(words):
            break
    
    return chunks




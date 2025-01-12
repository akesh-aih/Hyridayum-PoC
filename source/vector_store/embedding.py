from aih_rag.llms.ai21 import AI21
from aih_rag.embeddings.azure_openai import AzureOpenAIEmbedding
from typing import Optional, Any


async def query_text_embedding(query: Optional[str] = None, text: Optional[str] = None, model: AzureOpenAIEmbedding = None) -> Any:
    """
    Asynchronously get embeddings for a given query or text using a specified model.
    
    Args:
        query (Optional[str]): The query string from which to generate an embedding.
        text (Optional[str]): The text from which to generate an embedding.
        model (Any): The model used to generate embeddings, defaults to jina_processor if not specified.

    Note: Either query or text must be provided. If provided both, embedding for query will be returned.
    
    Returns:
        Any: The embedding generated from the input query or text.
    
    Raises:
        ValueError: If neither query nor text is provided.
    """
    if query is not None:
        return model.get_query_embedding(query=query)
    elif text is not None:
        # print('TEXT',text)
        return model.get_text_embedding(text=text)
    else:
        raise ValueError("Please provide either a 'query' or 'text' parameter.")



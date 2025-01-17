from datetime import datetime
import os
import sys
import json
import time
import pandas as pd
from dotenv import load_dotenv
from rich import print
from aih_rag.embeddings.azure_openai import AzureOpenAIEmbedding
from openai import AzureOpenAI
import json
import os
from aih_automaton import Task, Agent, LinearSyncPipeline
from source.AzureOpenai import AzureOpenAIModel
from aih_automaton.tasks.task_literals import OutputType
from aih_rag.vector_stores.deeplake import DeepLakeVectorStore
from source.utils import read_json, write_json
from source.vector_store.embedding import query_text_embedding
from aih_rag.schema import TextNode
from aih_rag.vector_stores import VectorStoreQuery
from source.logger import logger
TOP_K = 4

load_dotenv()
# uuid = "test-uuid"
root = os.path.dirname(os.path.abspath(__file__))
print(root)

# Initialize Azure OpenAI
azure_api_key = os.getenv("API_Key")
azure_endpoint = os.getenv("End_point")
azure_api_version = os.getenv("API_version")
# Initialize AzureOpenAIEmbedding for embedding generation

azure_embedding = AzureOpenAIEmbedding(
    model="text-embedding-3-small",
    azure_endpoint=azure_endpoint,
    api_key=azure_api_key,
    api_version="2024-02-01",
    azure_deployment="text-embedding-3-small",
)


def chatbot(
    query=None,
    reference_text=None,
    messages: list[dict] = None,
    chat_with="summary",
    summary_dir=None,
):
    """
    Generates a chatbot-like response to the query using the reference text and summary in a descriptive, pointwise format.

    Parameters:
    - query (str): The query or question to be answered.
    - reference_text (str): The text containing reference information.

    Returns:
    - str: A descriptive, pointwise response addressing the query using the reference text and summary.
    """
    client = AzureOpenAIModel(
        azure_endpoint=azure_endpoint,
        azure_api_key=azure_api_key,
        azure_api_version=azure_api_version,
        parameters={
            "temperature": 0.7,
            "model": "gpt-35-turbo",
        },
    )

    # Construct the prompt

    prompt = (
        "Generate a detailed and descriptive response to the following query in a chatbot-friendly format\n\n"
        + f"Query: {query}\n\n"
    )
    if reference_text is not None:
        prompt += "\nReference Text:\n" + reference_text

    # f"Instructions:\n"

    # System content with response generation instructions
    system_content = """
    Generate a chatbot-like response adhering to the following rules:
    1. Use the reference text and summary to provide a detailed, descriptive, and structured answer.
    2. Respond in a point-wise format for clarity.
    3. Use a professional yet conversational tone suitable for chatbot interactions.
    4. Ensure accuracy and avoid adding unsupported information from your knowledge base, you will only answer from the facts of reference text and summary.
    5. Conclude with a polite and friendly closing statement.
    6. Clearly highlight key points or steps.\n"
    7. Conclude with a friendly and professional closing statement.\n"
    
    Most important Note:
    - You will only answer from the facts of reference text and summary and not from your knowledge base.
    - You will not answer unrelevenet, vulgar, or offensive queries and out or the persona you are playing.
    """

    if messages is None:
        print("Messages are none", messages)
        messages = [
            {"role": "system", "content": system_content},
        ]
        if chat_with == "summary":
            summary = read_json(os.path.join(summary_dir, "summary.json"))[-1][
                "summary"
            ]
            messages.extend(
                [
                    {"role": "assistant", "content": summary},
                    {"role": "user", "content": prompt},
                ]
            )

        elif chat_with == "document":
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": "Hello, I am a chatbot. How can I help you?",
                    },
                    {"role": "user", "content": prompt},
                ]
            )
        print("Messages are now", messages) 
        
        
    # c:\Users\akliv\OneDrive\Desktop\Akesh kumar\POC\Doc_summarize_and_chat\assets\user-test_global\deeplake\Deeplake_unsummarized'

    # c:\\Users\\akliv\\OneDrive\\Desktop\\Akesh kumar\\POC\\Doc_summarize_and_chat\\assets\\user-user-test_global\\deeplake\\Deeplake_unsummarized

    else:
        print("Messages are present", messages)
        messages.append({"role": "user", "content": prompt})

    agent = Agent(role="Chatbot")
    task = Task(
        model=client,
        agent=agent,
        output_type=OutputType.TEXT,
        messages=messages,
        # instructions=system_content,
    )
    pipeline = LinearSyncPipeline(tasks=[task])
    response = pipeline.run()

    # print(response)
    messages.append({"role": "assistant", "content": response[0]["task_output"]})

    return messages


async def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--uuid", type=str, required=True)
    parser.add_argument("--unique_content_id", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--chat_with", type=str, required=True)  # summary/document

    args = parser.parse_args()
    uuid = args.uuid
    user_query = args.query
    chat_with = args.chat_with
    import time

    user_dir = os.path.join(root, "assets", f"user-{uuid}")
    deeplake_dir = os.path.join(user_dir, "deeplake")
    summaries_dir = os.path.join(user_dir, "summaries")
    user_content_dir = os.path.join(user_dir, "content")
    process_dir = os.path.join(user_dir, "systems")

    VECTOR_STORE_NAMEs = {
        # "summarized": "Deeplake_summarized",
        "unsummarized": "Deeplake_unsummarized",
    }
    chat_sources_clients = [
        DeepLakeVectorStore(
            dataset_path=os.path.join(deeplake_dir, store_name),
            # read_only=True,
            overwrite=False,
            # verbose=False,
        )
        for store_name in VECTOR_STORE_NAMEs.values()
    ]
    print(chat_sources_clients)
    # exit()
    # summaries_dir = os.path.join(user_dir, "summaries")
    chat_dir = os.path.join(user_dir, "chat")
    summary_chat_history = os.path.join(user_dir, "chat", "summary", "messages.json")
    doc_chat_history = os.path.join(user_dir, "chat", "doc", "messages.json")
    chat_history_path = os.path.join(chat_dir)

    starttime = time.time()
    messages_path = None
    if chat_with == "document":
        if os.path.exists(doc_chat_history):
            messages = read_json(doc_chat_history)
        else:
            messages = None
        messages_path = doc_chat_history
        query = VectorStoreQuery(
            query_embedding=await query_text_embedding(
                query=user_query, model=azure_embedding
            ),
            similarity_top_k=TOP_K,
        )
        query_results = [store.query(query).nodes for store in chat_sources_clients]
        nodes = [node.text for nodes in query_results for node in nodes]
        reference_text = "\n\n".join(nodes)
    elif chat_with == "summary":
        if os.path.exists(summary_chat_history):
            messages = read_json(summary_chat_history)
        else:
            messages = None
        messages_path = summary_chat_history
        reference_text = None
    chat_message = chatbot(
        user_query,
        reference_text=reference_text,
        messages=messages,
        chat_with=chat_with,
        summary_dir=summaries_dir,
    )
    # print("Chat messages", chat_message)
    print("Time Taken", time.time() - starttime)
    write_json(chat_message, messages_path)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

# python chat.py --uuid id12423 --query "What is the purpose of this document?" chat_with document|summary

import os
import sys
import time
import json
import pdfplumber
import docx
from dotenv import load_dotenv
from rich import print
from source.config import summarizer_text_function
import asyncio
from source.vector_store.embedding import query_text_embedding
from source.utils import read_json, write_json
from source.utils import chunk_text
from aih_rag.vector_stores.deeplake import DeepLakeVectorStore
from aih_rag.schema import TextNode

from source.loaders.file_loaders import file_loader, load_files_async

# from openai import AzureOpenAI
from source.AzureOpenai import AzureOpenAIModel
from aih_rag.embeddings.azure_openai import AzureOpenAIEmbedding

# Load environment variables
load_dotenv()

import time


root = os.path.dirname(os.path.abspath(__file__))
from source.loaders.utils import create_user_directories


# Initialize Azure OpenAI
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

client = AzureOpenAIModel(
    azure_endpoint=azure_endpoint,
    azure_api_key=azure_api_key,
    azure_api_version=azure_api_version,
    parameters={
        "temperature": 0.3,
        "model": "gpt-35-turbo",
    },
)
from source.vector_store.utils import async_create_nodes, create_node

# Retry logic decorator
from source.utils import retry_async, retry_sync
from source.utils import chunk_text
from aih_automaton import Task, Agent, LinearSyncPipeline
from aih_automaton.tasks.task_literals import OutputType


@retry_async(backoff=1.17, retries=1, fallback="")
async def generate_chunk_summary_and_title(chunk):
    """Generate a structured JSON response using query data and title."""
    client = AzureOpenAIModel(
        azure_endpoint=azure_endpoint,
        azure_api_key=azure_api_key,
        azure_api_version=azure_api_version,
        parameters={
            "temperature": 0.3,
            "model": os.environ,
        },
    )
    system_prompt = """You are specialized in summarizing given text.
    You will be given raw text data, and your task is to summarize and organize given text data and create organized, meaningful, and complete information. You will also create a title for that information.
    
    - Their is no any threshold or ratio for summarization, but goal is to maintain, organize essence of  
    infomation.
    - Highlight the most important points or key information from the text, use **bold**, *italic* ***other*** for this.
    Important Note: We should not loose any information in the summary.
    """

    prompt = f"""
    Generate a concise summary of the given text data:
    **Text Data**: {chunk}
    """
    agent = Agent(role="Text Summarizer Agent", prompt_persona=prompt)
    task = Task(
        name="Text Summarizer Task",
        agent=agent,
        output_type=OutputType.FUNCTION_CALL,
        function_call="auto",
        functions=summarizer_text_function,
        instructions=system_prompt,
        model=client,
    )
    pipeline = LinearSyncPipeline(tasks=[task], completion_message="Chunk Summarized")
    response = pipeline.run()[0]["task_output"].arguments
    response_message = json.loads(response)["summarized_text"]
    return response_message


# @retry_async(backoff=1.17, retries=1, fallback="")
async def generate_chunk_summary_and_title_no_pipeline(chunk):
    """Generate a structured JSON response No pipeline"""
    client = AzureOpenAIModel(
        azure_endpoint=azure_endpoint,
        azure_api_key=azure_api_key,
        azure_api_version=azure_api_version,
        parameters={
            "temperature": 0.4,
            "model": os.getenv("Engine"),
        },
    )
    system_prompt = """You are a highly advanced AI specialized in text summarization. Your task is to summarize the provided content while maintaining the key points, structure, and context. Ensure no critical information is lost, and the summary is concise yet complete. Highlight the most important points clearly, with logical organization. Utilize `text_summary_and_title` function to generate the summary and title.
    - Highlight the most important points or key information from the text, use **bold**, *italic* ***other*** for this.
    - Dont mention **The text discusses, The Provided text discusses..., Introduction, Outro, etc. in the summary**
    """

    prompt = f"""
    Here is a chunk of text from a PDF. Please summarize it in a way that preserves all essential details while highlighting key points. Ensure the summary is concise, clear, and retains the original meaning. Do not omit any critical information, and organize the points logically for easy understanding. The input chunk is as follows: {chunk}
    """

    response = client.generate_text(
        prompt=prompt,
        system_persona=system_prompt,
        function_call="auto",
        functions=summarizer_text_function,
    )

    # print(response_message)
    response_message = json.loads(response.arguments)

    return response_message


import openai


async def generate_summary(context):
    """Generate a structured JSON response using query data and title."""
    client = AzureOpenAIModel(
        azure_endpoint=azure_endpoint,
        azure_api_key=azure_api_key,
        azure_api_version=azure_api_version,
        parameters={
            "temperature": 0.5,
            "model": os.getenv("Engine"),
            "presence_penalty": 0.5,
        },
    )

    system_prompt = """
    You are a highly advanced AI specialized in text summarization. Your task is to summarize the provided content while maintaining the key points, structure, and context. Ensure no critical information is lost, and the summary is concise yet complete. 
    
    - Highlight the most important points or key information from the text, use **bold**, *italic*      **other** for this.
    - Name, places, amount, key metrics and other quantifiable should be included in summary.
    - Ensure to divide summary in topics and subtopic wise in paragraphs.
    - Summary should be concise.
    
    """

    prompt = f"""
    Generate a summary of the given text data:
    **Text Data**: {context}
    """
    agent = Agent(role="Text Summarizer Agent", prompt_persona=prompt)
    task = Task(
        name="Text Summarizer Task",
        agent=agent,
        output_type=OutputType.TEXT,
        # function_call="auto",
        # functions=summarizer_text_function,
        instructions=system_prompt,
        model=client,
    )

    pipeline = LinearSyncPipeline(tasks=[task], completion_message="Chunk Summarized")
    response = pipeline.run()[0]["task_output"]
    # response_message = json.loads(response)["summarized_text"]
    return response


async def async_summarize_document(chunks, no_pipeline=True):
    """Summarize a document.
    returns list of dict of title and summary
    """
    if no_pipeline:
        tasks = [
            generate_chunk_summary_and_title_no_pipeline(chunk) for chunk in chunks
        ]
    else:
        tasks = [generate_chunk_summary_and_title(chunk) for chunk in chunks]
    summaries_of_chunks = await asyncio.gather(*tasks)
    return summaries_of_chunks


from aih_rag.schema import TextNode

from source.logger import logging

logger = logging.getLogger(__name__)


@retry_async(retries=1, fallback=False, logger=logger)
async def main(unique_content_id, session_id, pdf_path: str):
    """
    Takes a pdf process it to create summary and vector store for chat
    """
    # user idc
    start_time = datetime.now()
    assets = os.path.join(root, "assets")
    user_dir = os.path.join(assets, f"u-{uuid}")
    session_dir = os.path.join(user_dir, f"s-{session_id}")

    deeplake_dataset_path = os.path.join(
        session_dir, "deeplake", f"Deeplake_{unique_content_id}"
    )

    summary_path = os.path.join(
        session_dir, "summaries", f"Summary_{unique_content_id}.json"
    )
    process_dir = os.path.join(session_dir, "systems")

    print("deeplake dataset", deeplake_dataset_path)
    print("pdf_path", pdf_path)
    print("summary_path", summary_path)
    chunks = await file_loader(pdf_path)

    logger.log(logging.INFO, f"File divided into {len(chunks)} parts")
    summary_title_dict_list = await async_summarize_document(chunks=chunks)

    logger.log(logging.INFO, f"Summarized {len(summary_title_dict_list)} parts")
    sumarized_chunks_corpus = "\n\n".join(
        [
            f"Data: {data['summary']} Info: {data['title']}"
            for data in summary_title_dict_list
        ]
    )
    logger.log(logging.INFO, "Generating final summary")
    summary = await generate_summary(sumarized_chunks_corpus)
    logger.log(logging.INFO, "Final summary generated")

    write_json(
        [
            {
                "summary": summary,
                "metadata": {
                    "timetaken": datetime.now().timestamp() - start_time.timestamp(),
                    "sources": {"document_files": pdf_path},
                },
                "Retry": 5,
            }
        ],
        summary_path,
    )

    logger.log(logging.INFO, "Creating vector store")
    store_unsummarized = DeepLakeVectorStore(
        dataset_path=deeplake_dataset_path, overwrite=True
    )
    logger.log(logging.INFO, "Vector store created. Adding data to vector store")
    overall_chunks = []
    for chunk in chunks:
        sub_chunks = await chunk_text(chunk, chunk_size=1000, overlap=50)
        overall_chunks.extend(sub_chunks)
    nodes_unsummarized = await async_create_nodes(overall_chunks)

    await store_unsummarized.async_add(nodes_unsummarized)
    logger.log(logging.INFO, "Data added to vector store")
    return True


if __name__ == "__main__":
    from uuid import uuid4
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--uuid", type=str, required=True)
    parser.add_argument("--session_id", type=str, required=True)
    parser.add_argument("--document_paths", required=True, nargs="+")
    args = parser.parse_args()
    uuid = args.uuid
    document_paths = args.document_paths
    session_id = args.session_id
    from datetime import datetime

    create_user_directories(root, uuid, session_id)
    assets = os.path.join(root, "assets")
    user_dir = os.path.join(assets, f"u-{uuid}")
    session_dir = os.path.join(user_dir, f"s-{session_id}")
    system_dir = os.path.join(session_dir, "systems")  # contain system dir
    session_details_path = os.path.join(system_dir, "session_details.json")
    start_time = datetime.now()

    # user_content_dir = os.path.join(session_dir, "content")
    # process_dir = os.path.join(session_id, "systems")
    # user_chat_dir = os.path.join(session_id, "chat")
    if not os.path.exists(session_details_path):
        doc_to_process_for_user = {
            "user": {
                "user_id": uuid,
                "session_id": session_id,
                "documents": [],
                "request": {
                    "start_time": start_time.timestamp(),
                    "last_updated": None,
                },
            }
        }

        write_json(doc_to_process_for_user, session_details_path)
    else:
        doc_to_process_for_user = read_json(session_details_path)

    logger.log(
        logging.INFO,
        f"Processing {len(document_paths)} documents for user {uuid} and session {session_id}",
    )

    for i in range(len(document_paths)):
        logger.info(
            f"Processing document {i + 1} of {len(document_paths[i])} for user {uuid} and session {session_id}"
        )

        doc_details = {
            "path": document_paths[i],
            "unique_content_id": str(uuid4()),
            "status": False,
            "start_time": start_time.timestamp(),
            "end_time": None,
            "duration": None,
        }
        # print(doc_details)
        status = asyncio.run(
            main(
                pdf_path=doc_details["path"],
                session_id=session_id,
                unique_content_id=doc_details["unique_content_id"],
            )
        )
        # print(status)
        doc_details["end_time"] = datetime.now().timestamp()
        doc_details["duration"] = doc_details["end_time"] - doc_details["start_time"]
        doc_details["status"] = status

        doc_to_process_for_user["user"]["documents"].append(doc_details)

        write_json(
            doc_to_process_for_user,
            os.path.join(system_dir, "session_details.json"),
        )

    doc_to_process_for_user["user"]["request"]["last_updated"] = (
        datetime.now().timestamp()
    )
    write_json(
        doc_to_process_for_user,
        os.path.join(session_dir, "systems", "session_details.json"),
    )
    

# # python summarize.py --uuid id12423 --document_paths "C:\Users\akliv\OneDrive\Desktop\Akesh kumar\POC\Doc_summarize_and_chat\assets\user-test_global\content\longrag.pdf" "C:\Users\akliv\OneDrive\Desktop\Akesh kumar\POC\Doc_summarize_and_chat\assets\user-test_global\content\longrag.pdf"

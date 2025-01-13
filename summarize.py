import os
import sys
import time
import json
import pdfplumber
import pandas as pd
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

import os


def create_user_directories(root, uuid):
    directories_to_create = [
        f"user-{uuid}",
        
        "deeplake",
        "summaries",
        "content",
        "chat",
        "chat/doc",
        "chat/summary",
        "systems",
    ]
    user_dir = os.path.join(root, "assets", f"user-{uuid}")
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)

    for directory in directories_to_create:
        path = os.path.join(user_dir, directory)
        if not os.path.exists(path):
            os.makedirs(path)


# Example usage
# Replace with your actual root directory

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
            "model": "gpt-35-turbo",
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
            "temperature": 0.2,
            "model": "gpt-35-turbo",
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


async def generate_summary(context):
    """Generate a structured JSON response using query data and title."""
    client = AzureOpenAIModel(
        azure_endpoint=azure_endpoint,
        azure_api_key=azure_api_key,
        azure_api_version=azure_api_version,
        parameters={
            "temperature": 0.5,
            "model": "gpt-4o",
        },
    )
    system_prompt = """You are a highly advanced AI specialized in text summarization. Your task is to summarize the provided content while maintaining the key points, structure, and context. Ensure no critical information is lost, and the summary is concise yet complete. Highlight the most important points clearly, with logical organization.
    Important Note: - Do not loose any information in the summary.
    - Highlight the most important points or key information from the text, use **bold**, *italic* ***other*** for this.
    - the summary is well formated in markdown but not be given in this format ```markdown\n content    ``` 
    - Dont mention **The text discusses, The Provided text discusses..., Introduction, Outro, etc. in the summary. It should be direct**
    - Ensure to seperate topics if you get multiple topic and they are not related
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


async def main(**kwargs):
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--uuid", type=str, required=True)
    parser.add_argument("--document_paths", required=True, nargs="+")
    args = parser.parse_args()
    uuid = args.uuid
    document_paths = args.document_paths
    import time

    start_time = time.time()
    create_user_directories(root, uuid)
    assets = os.path.join(root, "assets")
    user_dir = os.path.join(assets, f"user-{uuid}")
    deeplake_dir = os.path.join(user_dir, "deeplake")
    summaries_dir = os.path.join(user_dir, "summaries")
    user_content_dir = os.path.join(user_dir, "content")
    process_dir = os.path.join(user_dir, "systems")
    user_chat_dir = os.path.join(user_dir, "chat")

    write_json(
        {"process": "Started working with pdf."},
        os.path.join(process_dir, "process.json"),
    )

    # dataset_path_summarized = os.path.join(
    #     deeplake_dir, f"Deeplake_summarized"
    # )
    dataset_path_unsummarized = os.path.join(deeplake_dir, f"Deeplake_unsummarized")
    pdf_paths = document_paths

    write_json(
        {"process": "Loading your document"}, os.path.join(process_dir, "process.json")
    )
    chunks = await asyncio.gather(*(file_loader(file_path) for file_path in pdf_paths))
    modules_for_summary = [f"{item}" for sublist in chunks for item in sublist]
    print("module", modules_for_summary[0])
    write_json(
        {"process": "Pdf loaded, summarizing your document"},
        os.path.join(process_dir, "process.json"),
    )
    # print("len. of chunks", [len(chunk) for chunk in chunks])
    # print("No. of chunks", len(modules_for_summary) )

    print(
        "Time to load chunks",
        time.time() - start_time,
    )
    summary_title_dict_list = await async_summarize_document(chunks=modules_for_summary)

    write_json(
        {
            "summaries": summary_title_dict_list,
            "metadata": {
                "timetaken": time.time() - start_time,
                "no_of_summaries": len(summary_title_dict_list),
            },
        },
        os.path.join(summaries_dir, "summary_title_dict_list.json"),
    )

    print("Time to summarize chunks", time.time() - start_time)
    sumarized_chunks_corpus = "\n\n".join(
        [
            f"Data: {data['summary']} Info: {data['title']}"
            for data in summary_title_dict_list
        ]
    )
    summary = await generate_summary(sumarized_chunks_corpus)

    write_json(
        {"process": "Summary of your document generated"},
        os.path.join(process_dir, "process.json"),
    )
    print("Time to generate complete summary", time.time() - start_time)
    write_json(
        {"summary": summary, "metadata": {"timetaken": time.time() - start_time}},
        os.path.join(summaries_dir, "summary.json"),
    )

    # store_summarized = DeepLakeVectorStore(
    #     dataset_path=dataset_path_summarized, overwrite=True
    # # )
    # nodes_summarized = await async_create_nodes([str(data) for data in summary_title_dict_list])
    # await store_summarized.async_add(nodes_summarized)

    store_unsummarized = DeepLakeVectorStore(
        dataset_path=dataset_path_unsummarized, overwrite=True
    )
    print("total_chunks", len(modules_for_summary))
    overall_chunks = []
    print("chunksss", modules_for_summary[0])
    for chunk in modules_for_summary:
        sub_chunks = await chunk_text(chunk, chunk_size=1000, overlap=50)
        overall_chunks.extend(sub_chunks)

    print("total_sub_chunks", len(overall_chunks))
    nodes_unsummarized = await async_create_nodes(overall_chunks)

    await store_unsummarized.async_add(nodes_unsummarized)
    print("Overall Time to generate summary", time.time() - start_time)
    return


if __name__ == "__main__":
    asyncio.run(main())

# python summarize.py --uuid id12423 --document_paths "C:\Users\akliv\OneDrive\Desktop\Akesh kumar\POC\Doc_summarize_and_chat\assets\user-test_global\content\longrag.pdf" "C:\Users\akliv\OneDrive\Desktop\Akesh kumar\POC\Doc_summarize_and_chat\assets\user-test_global\content\longrag.pdf"
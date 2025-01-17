import time
from logging import Logger
from functools import wraps
import asyncio
import random
import sys
async def chunk_text(text, chunk_size=1000, overlap=100):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks
def retry_async(
    exceptions: tuple = (Exception,),
    retries=3,
    delay=1,
    fallback=None,
    backoff=2,
    logger: Logger = None,
):
    """
    Retry an asynchronous function call upon encountering a specific exception.

    :param exceptions: The exceptions to catch and retry on.
    :param retries: Number of retry attempts (default is 3).
    :param delay: Delay between retries in seconds (default is 1 second).
    :param fallback: Value to return if retries are exhausted (default is None).
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            exp_delay = delay
            for attempt in range(retries):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = exc_tb.tb_frame.f_code.co_filename
                    message = f"Attempt {attempt + 1} Failed, line no. {exc_tb.tb_lineno}, {exc_type}, {fname}. Error: {e} Trying again in {exp_delay} seconds"
                   
                    if logger:
                        logger.critical(message)
                    else:
                        print(message)
                    # print(f"Attempt {attempt + 1} failed: {e} Trying again in {exp_delay} seconds")
                    if attempt < retries - 1:
                        await asyncio.sleep(exp_delay)
                        exp_delay = exp_delay**backoff + abs(
                            random.normalvariate(0, 1.2)
                        )
                    else:
                        if fallback is not None:
                            message = "Retries exhausted. Returning fallback value. "
                            if logger:
                                logger.error(message)
                            else:
                                print(message)
                            return fallback
                        else:
                            raise

        return wrapper

    return decorator


def retry_sync(
    exceptions: tuple = (Exception,), retries=3, delay=1, fallback=None, backoff=1.17
):
    """
    Retry a function call upon encountering a specific exception.

    :param exception: The exception to catch and retry on.
    :param retries: Number of retry attempts (default is 3).
    :param delay: Delay between retries in seconds (default is 1 second).
    :param fallback: Value to return if retries are exhausted (default is None).
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # backoff_factor = backoff
            exp_delay = delay
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    print(
                        f"Attempt {attempt + 1} failed executing {func.__name__}: {e} Trying again in {exp_delay} seconds"
                    )
                    if attempt < retries - 1:
                        time.sleep(exp_delay)
                        exp_delay = exp_delay**backoff + random.random()
                    else:
                        print("Retries exhausted. Returning fallback value.")
                        if fallback is not None:
                            return fallback

                        else:
                            raise
                # delay = 2.17**delay

        return wrapper

    return decorator
import os
from  aih_rag.readers.file.docs_reader import PDFReader, DocxReader
from aih_rag.readers.file.markdown_reader import MarkdownReader


def create_user_directories(root, uuid,session_id):
    directories_to_create = [        
        "deeplake",
        "summaries",
        "content",
        "chat",
        "chat/doc",
        "chat/summary",
        "systems"
    ]
    user_dir = os.path.join(root, "assets", f"u-{uuid}",f"s-{session_id}")
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)

    for directory in directories_to_create:
        path = os.path.join(user_dir, directory)
        if not os.path.exists(path):
            os.makedirs(path)
            



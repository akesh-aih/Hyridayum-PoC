import pymupdf4llm
from docx import Document
from .utils import retry_async, retry_sync, chunk_text
import asyncio
import os

CHUNK_SIZE = 10000
OVERLAP = 300

@retry_async(fallback='',delay=.1)
async def read_pdf_in_markdown(pdf_path:str,**kwargs):
    md_text = pymupdf4llm.to_markdown(pdf_path,**kwargs)
    return md_text

@retry_sync(fallback='')
def read_docx(docx_path):
    # Open the docx file
    doc = Document(docx_path)
    text = []
    # doc.
    for para in doc.paragraphs:
        text.append(para.text)
    return '\n'.join(text)
from PyPDF4 import PdfFileReader
import pymupdf

async def read_pdf(pdf_path):

    pdf = pymupdf.open(pdf_path)
    text = ' '.join ([page.get_text() for page in pdf])
    return text
        
# @retry_async(fallback='')
async def file_loader(file_path):
    extension = os.path.splitext(file_path)[1].lower()  

    if extension == ".pdf":
        # pdf_text = await read_pdf_in_markdown(file_path)
        pdf_text = await read_pdf(file_path)
        
        chunks = await chunk_text(pdf_text,chunk_size=CHUNK_SIZE,overlap=OVERLAP)
    elif extension in [".docx", ".doc"]:
        chunks =  await chunk_text(read_docx(file_path))
        print("Unsupported file format.")
    else:
        return []
    return chunks

async def load_files_async(list_of_files) -> list[str]:
    """ 
    Load files from a list of file paths asynchronously.
    returns list of contents
    """
    tasks = [file_loader(file_path) for file_path in list_of_files]
    return  await asyncio.gather(*tasks)

if __name__ =="__main__":
    pdf_path = r"C:\Users\akliv\OneDrive\Desktop\Akesh kumar\POC\Doc_summarize_and_chat\LongRAG Enhancing Retrieval-Augmented Generation with Long-context LLMs (1).pdf"
    # print((read_pdf(pdf_path,)))
    print((read_pdf_in_markdown(pdf_path,)))
import streamlit as st
import time
import random
import os
import asyncio
from summarize import create_user_directories
from source.loaders.file_loaders import file_loader
from summarize import async_summarize_document, generate_summary
from source.utils import read_json, write_json, chunk_text
from aih_rag.vector_stores.deeplake import DeepLakeVectorStore
from source.vector_store.utils import async_create_nodes
from RLHF_summarizer import retry_summary_update

root = os.getcwd()

async def main():
    st.title("Document Summarization App")

    # Initialize session state variables
    if "user_id" not in st.session_state:
        st.session_state.user_id = f"{random.randint(1000, 9999)}{''.join(random.choices(list('abcdefghijklmnopqrstuvwxyz'), k=4))}"
    if "summary" not in st.session_state:
        st.session_state.summary = None
    if "feedback" not in st.session_state:
        st.session_state.feedback = ""
    if "accepted" not in st.session_state:
        st.session_state.accepted = False

    st.sidebar.header("Input Parameters")

    uuid = st.sidebar.text_input("Enter username", st.session_state.user_id)
    
    if uuid != st.session_state.user_id:
        st.session_state.user_id = uuid

    uploaded_files = st.file_uploader(
        "Upload your documents (PDF only)", type=["pdf"], accept_multiple_files=True
    )

    if st.button("Process Documents"):
        if not uuid:
            st.error("Please enter a username.")
        elif not uploaded_files:
            st.error("Please upload at least one document.")
        else:
            create_user_directories(root, uuid)
            user_dir = os.path.join(root, "assets", f"user-{uuid}")
            content_dir = os.path.join(user_dir, "content")
            deeplake_dir = os.path.join(user_dir, "deeplake")
            summaries_dir = os.path.join(user_dir, "summaries")

            st.toast("Loading Documents...")

            document_paths = []
            for uploaded_file in uploaded_files:
                file_path = os.path.join(content_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())
                document_paths.append(file_path)

            start_time = time.time()
            chunks = await asyncio.gather(
                *(file_loader(file_path) for file_path in document_paths)
            )

            modules_for_summary = [f"{item}" for sublist in chunks for item in sublist]
            st.toast("Documents are being processed, getting summary...")

            summary_title_dict_list = await async_summarize_document(
                chunks=modules_for_summary
            )

            write_json(
                {
                    "summaries": summary_title_dict_list,
                    "metadata": {
                        "time_taken": time.time() - start_time,
                        "no_of_summaries": len(summary_title_dict_list),
                    },
                },
                os.path.join(summaries_dir, "summary_title_dict_list.json"),
            )

            summarized_chunks_corpus = "\n\n".join(
                [
                    f"Data: {data['summary']} Info: {data['title']}"
                    for data in summary_title_dict_list
                ]
            )

            st.toast("Generating final summary...")
            summary = await generate_summary(summarized_chunks_corpus)

            write_json(
                [{
                    "summary": summary,
                    "metadata": {
                        "time_taken": time.time() - start_time,
                    },
                    "retry": 5,
                    "feedback": ''
                }],
                os.path.join(summaries_dir, "summary.json")
            )

            st.session_state.summary = summary
            st.write("Generated Summary:", st.session_state.summary)

            st.session_state.modules_for_summary = modules_for_summary
            st.session_state.summarized_chunks_corpus = summarized_chunks_corpus

            if st.session_state.summary:
                store_unsummarized = DeepLakeVectorStore(
                    dataset_path=os.path.join(deeplake_dir, "Deeplake_unsummarized"), overwrite=True
                )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Accept"):
                st.session_state.accepted = True
                overall_chunks = []
                for chunk in st.session_state.modules_for_summary:
                    sub_chunks = await chunk_text(chunk, chunk_size=1000, overlap=50)
                    overall_chunks.extend(sub_chunks)

                nodes = await async_create_nodes(overall_chunks)
                await store_unsummarized.async_add(nodes)
                st.success("Summary accepted and data stored successfully!")

        with col2:
            if st.button("Reject"):
                st.session_state.accepted = False
                st.session_state.feedback = st.text_area(
                    "Enter Feedback", 
                    placeholder="Provide feedback to refine the summary...",
                    value=st.session_state.feedback
                )

                if st.session_state.feedback:
                    st.toast("Retrying summarization with feedback...")
                    updated_summary = retry_summary_update(
                        st.session_state.summary,
                        feedback_from_user=st.session_state.feedback,
                        additional_context=st.session_state.summarized_chunks_corpus
                    )
                    st.session_state.summary = updated_summary
                    st.write("Updated Summary:", st.session_state.summary)
                else:
                    st.error("Please provide feedback to refine the summary.")

if __name__ == "__main__":
    asyncio.run(main())

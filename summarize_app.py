import streamlit as st
import time
import random
import os
from summarize import create_user_directories

root = os.getcwd()


async def main():
    st.title("Document Summarization App")

    if "user_id" not in st.session_state:
        st.session_state.user_id = f"{random.randint(1000, 9999)}{''.join(random.choices(list('abcdefghijklmnopqrstuvwxyz'), k=4))}"

    st.sidebar.header("Input Parameters")

    uuid = st.sidebar.text_input("Enter username", st.session_state.user_id)

    # If the user updates the username, update session state
    if uuid != st.session_state.user_id:
        st.session_state.user_id = uuid
        # uuid = st.sidebar.text_input("Enter username",random_user_id)
    uploaded_files = st.file_uploader(
        "Upload your documents (PDF only)", type=["pdf"], accept_multiple_files=True
    )

    if st.button("Process Documents"):
        if not uuid:
            st.error("Please enter a UUID.")
        elif not uploaded_files:
            st.error("Please upload at least one document.")
        else:
            import os
            from dotenv import load_dotenv; load_dotenv()
            create_user_directories(root, uuid)
            user_dir = os.path.join(root, "assets", f"user-{uuid}")
            # process_dir = os.path.join(user_dir, "systems")
            content_dir = os.path.join(user_dir, "content")
            assets = os.path.join(root, "assets")
            # user_dir = os.path.join(assets,f"user-{uuid}")
            deeplake_dir = os.path.join(user_dir, "deeplake")
            summaries_dir = os.path.join(user_dir, "summaries")
            user_content_dir = os.path.join(user_dir, "content")
            process_dir = os.path.join(user_dir, "systems")
            user_chat_dir = os.path.join(user_dir, "chat")
            dataset_path_unsummarized = os.path.join(deeplake_dir, f"Deeplake_unsummarized")
            st.toast("Loading Documents...")
            document_paths = []
            for uploaded_file in uploaded_files:
                file_path = os.path.join(content_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())
                document_paths.append(file_path)
            starttime = time.time()
            from source.loaders.file_loaders import file_loader

            chunks = await asyncio.gather(
                *(file_loader(file_path) for file_path in document_paths)
            )
            modules_for_summary = [f"{item}" for sublist in chunks for item in sublist]
            st.toast("Documents are being processed, Getting summary...")

            from summarize import async_summarize_document

            summary_title_dict_list = await async_summarize_document(
                chunks=modules_for_summary
            )
            st.toast("All most Done...")

            from source.utils import read_json, write_json

            write_json(
                {
                    "summaries": summary_title_dict_list,
                    "metadata": {
                        "timetaken": time.time() - starttime,
                        "no_of_summaries": len(summary_title_dict_list),
                    },
                },
                os.path.join(summaries_dir, "summary_title_dict_list.json"),
            )

            sumarized_chunks_corpus = "\n\n".join(
                [
                    f"Data: {data['summary']} Info: {data['title']}"
                    for data in summary_title_dict_list
                ]
            )
            st.toast('Summarizing...')
            from summarize import generate_summary
            import os
            summary = await generate_summary(sumarized_chunks_corpus)

            st.toast("Summarization Done...")

            write_json(
                [{
                    "summary": summary,
                    "metadata": {
                        "timetaken": time.time() - starttime,
                        "documents": document_paths
                    },
                    "Retry": 5,
                    "feedback": ''
                }],
                os.path.join(summaries_dir, "summary.json"))
            
            st.session_state['summary'] = summary
            st.write(st.session_state['summary'])
            
            from aih_rag.vector_stores.deeplake import DeepLakeVectorStore
            from source.vector_store.utils import async_create_nodes

            from RLHF_summarizer import retry_summary_update
            from source.utils  import chunk_text
            store_unsummarized = DeepLakeVectorStore(
                dataset_path=os.path.join(deeplake_dir, "Deeplake_unsummarized"), overwrite=True
            )
            
            overall_chunks = []
            for chunk in modules_for_summary:
                sub_chunks = await chunk_text(chunk, chunk_size=1000, overlap=50)
                overall_chunks.extend(sub_chunks)

            nodes = await async_create_nodes(overall_chunks)
            await store_unsummarized.async_add(nodes)
            st.success("Documents processed and summarized successfully!")
if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

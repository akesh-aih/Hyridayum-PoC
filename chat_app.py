import streamlit as st
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()
from chat import chatbot
from aih_rag.vector_stores import DeepLakeVectorStore
from source.utils import read_json, write_json
from aih_rag.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode
from source.vector_store.utils import query_text_embedding
from aih_rag.embeddings import AzureOpenAIEmbedding
from aih_automaton import Task, Agent, LinearSyncPipeline
from source.AzureOpenai import AzureOpenAIModel
from aih_automaton.tasks.task_literals import OutputType

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

from summarize_app import main as summmarizer_main

def chat_bot(messages):
    client = AzureOpenAIModel(
        azure_endpoint=os.getenv("End_point"),
        azure_api_key=os.getenv("API_Key"),
        azure_api_version=os.getenv("API_version"),
        parameters={
            "temperature": 0.7,
            "model": "gpt-35-turbo",
        },
    )
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

    return response[0]["task_output"]


# Load environment variables
load_dotenv()

TOP_K = 5

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    system_prompt = """
    Generate a chatbot-like response adhering to the following rules:
    1. Use the reference text and summary to provide a detailed, descriptive, and structured answer.
    2. Respond in a point-wise format for clarity.
    3. Use a professional yet conversational tone suitable for chatbot interactions.
    4. Ensure accuracy and avoid adding unsupported information from your knowledge base, you will only answer from the facts of reference text and summary.

    However, you must adhere to the following guidelines:

    Never give medical advice
    Maintain confidentiality
    Never use harmful language
    Prioritize user safety
    Provide accurate and reliable information
    Do not provide false or misleading information
    Always answer from the facts given to you as input like summary, reference text. Dont answer out of context.
    
    Always focus on user query for answer
    """
    st.session_state.chat_history = [{"role": "system", "content": system_prompt}]
    st.session_state.display_chat_history = []

# Function to initialize vector store if not already set


# Streamlit UI
st.title("Chatbot Application")
st.write("Welcome! This is a simple chatbot interface. Start chatting below:")

# Sidebar inputs
st.sidebar.header("Input Parameters")


async def main():
    import random

    if "user_id" not in st.session_state:
        st.session_state.user_id = f"{random.randint(1000, 9999)}{''.join(random.choices(list('abcdefghijklmnopqrstuvwxyz'), k=4))}"

    uuid = st.sidebar.text_input("Enter username", st.session_state.user_id)

    # If the user updates the username, update session state
    if uuid != st.session_state.user_id:
        st.success("Username updated to: " + uuid)
        st.session_state.user_id = uuid

    root = os.path.dirname(os.path.abspath(__file__))
    user_dir = os.path.join(root, "assets", f"user-{st.session_state.user_id}")
    deeplake_dir = os.path.join(user_dir, "deeplake")

    # uuid = st.sidebar.text_input("Enter username", "akeshkumar")
    if not uuid:
        st.error("Please enter a username.")
    else:
        chat_with = st.sidebar.selectbox(
            "Select a chat option", ["summary", "document"]
        )
        summaries_dir = os.path.join(
            "assets", f"user-{st.session_state.user_id}", "summaries"
        )
        
        if os.path.exists(summaries_dir):
        
            docs_path = read_json(  
                os.path.join(summaries_dir, "summary.json"))[-1]["metadata"]["documents"]
            for doc in docs_path:
                file_name = os.path.basename(doc)
                st.sidebar.download_button(
                            label=f"Download {file_name}",
                            file_name=file_name,
                            data=open(doc, "rb").read(),
                        )

        # Handle chat history and user query
        user_query = st.chat_input("Type your message here...")

        show_chat_from = 1
        
        if chat_with == "summary":
            if not os.path.exists(os.path.join(summaries_dir, "summary.json")):
                st.error("Please process the documents first.")
            else:
                if "summary" not in st.session_state:
                    summary_dict = read_json(
                        os.path.join(summaries_dir, "summary.json")
                    )
                    print("summary dict", summary_dict)
                    st.session_state["summary"] = summary_dict[-1]["summary"]
                    st.session_state["current_docs"] = summary_dict[-1]["metadata"][
                        "documents"
                    ]


                    message = {
                        "role": "assistant",
                        "content": f"This is the summary {st.session_state['summary']}. \n\n How can I help you?",
                    }
                    st.session_state.chat_history.append(message)
                    st.session_state.display_chat_history.append(message)
                # for doc in st.session_state["current_docs"]:
                #     file_name = os.path.basename(doc)
                #     st.sidebar.download_button(
                #         label=f"Download {file_name}",
                #         file_name=file_name,
                #         data=open(doc, "rb").read(),
                #     )

        elif chat_with == "document":
            st.session_state["vector_store"] = DeepLakeVectorStore(
                dataset_path=os.path.join(deeplake_dir, "Deeplake_unsummarized"),
                overwrite=False,
                verbose=True,
            )
            print("Vector store initialized.")
            print(st.session_state["vector_store"])
            message = {
                "role": "assistant",
                "content": f"Okay I will provide the answers from the documents....",
            }
            st.session_state.chat_history.append(message)
            st.session_state.display_chat_history.append(message)
            # for doc in st.session_state["current_docs"]:
            #     file_name = os.path.basename(doc)
            #     st.sidebar.download_button(
            #             label=f"Download {file_name}",
            #             file_name=file_name,
            #             data=open(doc, "rb").read(),
            #         )
            # else :
            #     retrived_text = None
            #     st.warning('Please enter a query')

        if user_query:
            if chat_with == "document":
                st.info("Searching for information...")
                vs_query = VectorStoreQuery(
                    query_embedding=await query_text_embedding(
                        query=user_query, model=azure_embedding
                    ),
                    similarity_top_k=TOP_K,
                    mode=VectorStoreQueryMode.DEFAULT,
                )
                query_results = st.session_state.vector_store.query(vs_query)
                retrived_text = "\n".join([node.text for node in query_results.nodes])

                # print("retrived_text",retrived_text)

                prompt = f"As a assistant you will provide a chatbot-like response to the User, Query: {user_query}  Reference Context: {retrived_text}, Take this reference context into account while responding."

            elif chat_with == "summary":
                prompt = f"As a assistant you will provide a chatbot-like response to the User, Query: {user_query} from the summary "

            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.session_state.display_chat_history.append(
                {"role": "user", "content": user_query}
            )
            bot_response = chat_bot(st.session_state.chat_history)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": bot_response}
            )
            st.session_state.display_chat_history.append(
                {"role": "assistant", "content": bot_response}
            )
            # print(st.session_state.chat_history)

        else:
            # st.warning('Please enter a query')
            pass

        # Display chat history
        for message in st.session_state.display_chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

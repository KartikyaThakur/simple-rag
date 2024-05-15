from dotenv import load_dotenv
import os

from core.node_postprocessors.duplicate_postprocessing import (
    DuplicateRemoverNodePostprocessor,
)
from helpers.filename_log_helper import FilenameLogHelper

load_dotenv()
import streamlit as st
from pinecone import Pinecone
from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.core.chat_engine.types import ChatMode
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.vector_stores import ( MetadataFilters, MetadataFilter )
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.core.postprocessor import SentenceEmbeddingOptimizer

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager(handlers=[llama_debug])
service_context = ServiceContext.from_defaults(callback_manager=callback_manager)

# Add option for user to choose context
filenames = FilenameLogHelper().read_all().split("\n")
# Add an empty option to the list of filenames
filenames.insert(0, "-- No context selected --")

st.set_page_config(
    page_title="Simple RAG",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

context_filename = st.selectbox("Choose a context", filenames)

if context_filename == "-- No context selected --":
    st.write("Select a context file to continue.")
    st.stop()

@st.cache_resource(show_spinner=False)
def get_index() -> VectorStoreIndex:
    pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY"),
        environment=os.environ.get("PINECONE_ENVIRONMENT"),
    )
    index_name = os.environ.get("PINECONE_INDEX_NAME")
    pinecone_index = pc.Index(index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store, service_context=service_context
    )


index = get_index()

print(index.summary)

if "chat_engine" not in st.session_state.keys():
    postprocessor = SentenceEmbeddingOptimizer(
        embed_model=service_context.embed_model,
        percentile_cutoff=0.5,
        threshold_cutoff=0.72,
    )
    filters = MetadataFilters (filters=[MetadataFilter(key="file_name", value=f"uploads/{context_filename}")])

    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode=ChatMode.CONTEXT,
        verbose=True,
        filters=filters,
        node_postprocessors=[postprocessor, DuplicateRemoverNodePostprocessor()],
    )

has_rag_title = "rag_title" in st.session_state.keys() and st.session_state.rag_title != ""

if not has_rag_title:
    st.session_state.rag_title = "Simple RAG"
else:
    st.session_state.rag_title = st.session_state.rag_title

st.title(f'{st.session_state.rag_title}')

if has_rag_title:
    cite_nodes = st.toggle('Cite Nodes')
else:
    cite_nodes = False

if has_rag_title:
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": f"Hi. I'm here to help you. Ask away!",
            }
        ]

    if prompt := st.chat_input("Your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.chat_engine.chat(message=prompt)
                    nodes = [node for node in response.source_nodes]
                    # if there are no nodes, we can't show anything
                    if len(nodes) == 0:
                        st.write("Sorry, I don't know the answer to that question.")
                    else:
                        st.write(response.response)
                        if cite_nodes:
                            for col, node, i in zip(st.columns(len(nodes)), nodes, range(len(nodes))):
                                with col:
                                    st.header(f"Source node {i+1}: score={node.score}")
                                    st.write(node.text)
                    message = {"role": "assistant", "content": response.response}
                    st.session_state.messages.append(message)
                except ValueError:
                    st.write("Sorry, I don't know the answer to that question.")
else:
    with st.chat_message("assistant"):
        st.write("I'm like Jon Snow,")
        st.write("I know nothing. 🤷🏽‍♂️")
        st.page_link("pages/1_Upload_Files.py", label= "Upload Files to Provide Context 📚", icon="⬆️")
        st.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExcmNwajl2NzYyOXQ5bWo3eG5ldGs0MmZncjQ1OGxsdXh0Zmhyc2Z2YyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/ZxAlfNwFZIA6I/giphy.gif")


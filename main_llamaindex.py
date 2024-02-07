from dotenv import load_dotenv
import os

from node_postprocessors.duplicate_postprocessing import (
    DuplicateRemoverNodePostprocessor,
)

load_dotenv()
import streamlit as st
from pinecone import Pinecone
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.chat_engine.types import ChatMode
from llama_index.vector_stores import PineconeVectorStore
from llama_index.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.indices.postprocessor import SentenceEmbeddingOptimizer

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager(handlers=[llama_debug])
service_context = ServiceContext.from_defaults(callback_manager=callback_manager)


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

if "chat_engine" not in st.session_state.keys():
    postprocessor = SentenceEmbeddingOptimizer(
        embed_model=service_context.embed_model,
        percentile_cutoff=0.5,
        threshold_cutoff=0.7,
    )

    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode=ChatMode.CONTEXT,
        verbose=True,
        node_postprocessors=[postprocessor, DuplicateRemoverNodePostprocessor()],
    )

st.set_page_config(
    page_title="Llama calling Llama",
    page_icon="👨🏽‍💻",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

st.title("🦙👉🏼 Llama calling Llama 👈🏼🦙")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi, I'm Llama. I'm here to help you use LlamaIndex properly. What do you need help with?",
        }
    ]

if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] == "user":
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
                    # for col, node, i in zip(st.columns(len(nodes)), nodes, range(len(nodes))):
                    #     with col:
                    #         st.header(f"Source node {i+1}: score={node.score}")
                    #         st.write(node.text)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message)
            except ValueError:
                st.write("Sorry, I don't know the answer to that question.")

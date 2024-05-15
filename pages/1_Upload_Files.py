import os
import time
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from io import StringIO

from core.ingestion import ingest_and_upload_to_pinecone
from helpers.filename_log_helper import FilenameLogHelper


st.title("Configure your RAG app")
# st.write("What do you want to name your RAG app?")

if not "rag_title" in st.session_state.keys():
    st.session_state.rag_title = "Simple RAG"

# has_rag_title = st.session_state.rag_title != "Simple RAG"

# if not has_rag_title:
#     st.session_state.rag_title = "Simple RAG"
#     rag_title_input = st.text_input("Set a title for your RAG app", key="rag_title", value="Simple RAG")
# else:
#     st.session_state.rag_title = st.session_state.rag_title
#     rag_title_input = st.text_input("Set a title for your RAG app", key="rag_title", value="Simple RAG")
# if not has_rag_title and st.button("Save App Title", type="primary"):
#     if "rag_title" not in st.session_state.keys():
#         st.session_state.rag_title = rag_title_input

st.write("Upload files, to be used as context")
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    
    # Write the uploaded_file to disk
    uploaded_file_path = f"uploads/{uploaded_file.name}"
    with open(uploaded_file_path, "wb") as f:
        f.write(bytes_data)

    # Store the file in session state
    st.session_state.uploaded_file = {
        "name": uploaded_file.name,
        "type": uploaded_file.type,
        "size": uploaded_file.size,
        "data": bytes_data,
        "hash": hash(bytes_data)
    }

    filename_log_helper = FilenameLogHelper()
    filenames = filename_log_helper.read_all().split("\n")

    print(filenames)
    print(uploaded_file.name)

    if (uploaded_file.name not in filenames):
        with st.spinner('Wait for it...'):
            load_dotenv()
            api_key=os.environ.get("PINECONE_API_KEY")
            environment=os.environ.get("PINECONE_ENVIRONMENT")
            index_name = os.environ.get("PINECONE_INDEX_NAME")
            for status in ingest_and_upload_to_pinecone(uploaded_file_path, index_name, api_key, environment):
                st.write(status)

        # Add the filename to the filename log file
        filename_log_helper.write(uploaded_file.name)

        st.write("✅ File uploaded successfully ⬆️")
        st.page_link("0_Home.py", label= "Back to the app", icon="⚡️")
    else:
        st.write("⚠️ File already uploaded ⚠️")


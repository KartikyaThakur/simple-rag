from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import os
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import download_loader, ServiceContext, VectorStoreIndex, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import Document
from pinecone import Pinecone
from llama_index.readers.file import PDFReader
from PIL import Image
import pytesseract
import logging

def extract_text_from_pdf_with_images(filepath: str):
    from PyPDF2 import PdfReader

    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        # Extract text from page
        text += page.extract_text() or ""

        # Extract images from page
        if hasattr(page, "images") and page.images:
            for image_index, image in enumerate(page.images):
                try:
                    with open(f"temp_image_{image_index}.png", "wb") as f:
                        f.write(image.data)
                    image_text = pytesseract.image_to_string(Image.open(f"temp_image_{image_index}.png"))
                    text += "\n" + image_text
                except Exception as e:
                    logging.error(f"Error extracting text from image: {e}")
                finally:
                    # Clean up temporary image file
                    if os.path.exists(f"temp_image_{image_index}.png"):
                        os.remove(f"temp_image_{image_index}.png")
    return text

def ingest_and_upload_to_pinecone(filepath: str, index_name: str, api_key: str, environment: str):
    # Initialize logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        if not Path(filepath).is_file() or not filepath.endswith('.pdf'):
            raise ValueError("The provided file path is invalid or not a PDF.")

        pc = Pinecone(
            api_key=api_key,
            environment=environment,
        )

        logging.info("Ingestion started.")
        yield f"\n🎬 Ingestion started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        yield "\n🟡 Extracting text from PDF, including images..."
        logging.info("Extracting text from PDF, including images...")
        text = extract_text_from_pdf_with_images(filepath)

        yield "\n🟡 Splitting text into chunks/nodes..."
        logging.info("Splitting text into chunks/nodes...")
        node_parser = SimpleNodeParser.from_defaults(chunk_size=500, chunk_overlap=20)

        # Correct Document creation and parsing
        document = Document(text=text, extra_info={"source": filepath})  # Include metadata
        logging.info("Parsing document...")
        nodes = node_parser.get_nodes_from_documents(documents=[document])  # Updated method for parsing
        logging.info("Document parsed into nodes successfully.")

        # Convert nodes back to documents
        documents = [
            Document(text=node.get_text(), extra_info={"node_id": node.node_id})
            for node in nodes
        ]
        logging.info("Nodes converted back to documents successfully.")

        yield "\n🟡 Generating embeddings..."
        logging.info("Generating embeddings using OpenAI...")
        llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
        embed_model = OpenAIEmbedding(model="text-embedding-ada-002", embed_batch_size=100)
        service_context = ServiceContext.from_defaults(
            llm=llm, embed_model=embed_model, node_parser=node_parser
        )

        yield "\n🟡 Initializing vector store..."
        logging.info("Initializing vector store with Pinecone...")
        pinecone_index = pc.Index(index_name)
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        yield "\n🟡 Indexing..."
        logging.info("Indexing documents into Pinecone...")
        index = VectorStoreIndex.from_documents(
            documents=documents,
            storage_context=storage_context,
            service_context=service_context,
            show_progress=True,
        )

        logging.info("Ingestion completed successfully.")
        yield f"\n✅ Ingestion completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    except Exception as e:
        logging.error(f"Error during ingestion: {e}")
        yield f"\n❌ Error: {e}"

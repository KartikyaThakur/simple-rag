from datetime import datetime
from dotenv import load_dotenv
import os
from llama_index.readers import SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser
from llama_index.llms import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index import (
    download_loader,
    ServiceContext,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.vector_stores import PineconeVectorStore
from pinecone import Pinecone


load_dotenv()
pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY"),
    environment=os.environ.get("PINECONE_ENVIRONMENT"),
)
if __name__ == "__main__":
    # print("Hello World!")
    # print(f'OPENAI_API_KEY: {os.environ.get("OPENAI_API_KEY")}')
    # print(f'PINECONE_ENVIRONMENT: {os.environ.get("PINECONE_ENVIRONMENT")}')
    # print(f'PINECONE_API_KEY: {os.environ.get("PINECONE_API_KEY")}')

    print(f"\n🎬 Ingestion started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print("\n🟡 Loading UnstructuredReader...")
    UnstructuredReader = download_loader("UnstructuredReader")
    dir_reader = SimpleDirectoryReader(
        input_dir="./llamaindex-docs",
        file_extractor={".html": UnstructuredReader()},
    )

    print("\n🟡 Loading data...")
    documents = dir_reader.load_data()
    print("\n🟡 Breaking into chunks/nodes...")
    node_parser = SimpleNodeParser.from_defaults(chunk_size=500, chunk_overlap=20)

    print("\n🟡 Generate embeddings...")
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
    embed_model = OpenAIEmbedding(model="text-embedding-ada-002", embed_batch_size=100)
    service_context = ServiceContext.from_defaults(
        llm=llm, embed_model=embed_model, node_parser=node_parser
    )

    print("\n🟡 Initializing vector store...")
    index_name = os.environ.get("PINECONE_INDEX_NAME")
    pinecone_index = pc.Index(index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("\n🟡 Indexing...")
    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        service_context=service_context,
        show_progress=True,
    )

    print(f"\n⬆️ Ingestion completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    pass

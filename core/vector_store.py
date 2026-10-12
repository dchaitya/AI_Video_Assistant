import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_classic.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()
CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcript"
EMBEDDINGMODEL = 'gemini-embedding-2-preview'

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model=EMBEDDINGMODEL)

def build_vector_store(transcript: str, collection_name: str = COLLECTION_NAME, clear_existing: bool = False) -> Chroma:
    print("Building vector Store")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(transcript)

    docs = [
        Document(page_content=chunk, metadata={'chunk_index': i})
        for i, chunk in enumerate(chunks) 
    ]

    embeddings = get_embeddings()

    if clear_existing:
        try:
            import chromadb
            client = chromadb.PersistentClient(path=CHROMA_DIR)
            client.delete_collection(name=collection_name)
        except Exception:
            pass

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=CHROMA_DIR
    )

    return vector_store

def load_vector_store(collection_name: str = COLLECTION_NAME) -> Chroma:
    embeddings = get_embeddings()
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )

    return vector_store

def get_retriever(vector_store:Chroma,k:int=4):
    return vector_store.as_retriever(search_type='similarity',
                                     search_kwargs={"k":k}
                                     )
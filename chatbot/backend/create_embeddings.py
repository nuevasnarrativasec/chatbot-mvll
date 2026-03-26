from uuid import uuid4
import openai
import tiktoken
from openai import OpenAI
import pandas as pd
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
import time
import streamlit as st

DATA_PATH = "./datos/docs"
METADATA_PATH = "./datos"
METADATA_FIELDS = {
        "AÑO": "AÑO",
        "FECHA": "FECHA",
        "TITULO": "TÍTULO",
        "NOMBRE_ARCHIVO": "NOMBRE_ARCHIVO"
}

openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
# load_dotenv()
tokenizer = tiktoken.get_encoding("cl100k_base")
EMBEDDING_MODEL = "text-embedding-3-small"
# openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
ENCODING_FORMAT = "Windows-1252"
metadata = pd.read_csv(f"{METADATA_PATH}/metadata.csv",sep=";",encoding=ENCODING_FORMAT)

def token_counter(text):
    return len(tokenizer.encode(text))

text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", ".", "\n", " "],
        chunk_size=8192,
        chunk_overlap=100,
        # chunk_size=600,
        # chunk_overlap=200,
        length_function=token_counter
)

def document_to_text(document):
        document_text = ""
        for page in document.pages:
                document_text += page.extract_text()
        return document_text

def entries_from_path(path):
        entries = []
        print("Imprimiendo ruta: ", path)
        document_file_name = path.split("/")[-1]
        document = PdfReader(path)
        document_text = document_to_text(document)
        path_metadata_row = metadata[metadata.NOMBRE_ARCHIVO == document_file_name]
        
        print(f"Procesando archivo: {document_file_name}")
        print(f"Metadata encontrada: {path_metadata_row}")

        document_metadata = {
               key: str(path_metadata_row[METADATA_FIELDS[key]].values[0]) 
               for key in METADATA_FIELDS.keys()
        }
        # print({"text": document_text, "metadata": document_metadata})
        return {"text": document_text, "metadata": document_metadata}


def join_embeddings_chunks(chunks, embeddings):
        print("Joining documents and embeddings...")
        chunks_as_dict = [chunk.metadata for chunk in chunks]
        
        for chunk, metadata in zip(chunks, chunks_as_dict):
                metadata["text"] = chunk.page_content
        
        embeddings_with_metadata = [
                {
                "values": embed,
                "metadata": chunk_metadata,
                "id": str(uuid4())
                }
                for embed, chunk_metadata in zip(embeddings, chunks_as_dict)
        ]

        return embeddings_with_metadata

def embeddings_from_chunks(chunks):

    print("Embedding Documents...")

    embeddings_response = openai_client.embeddings.create(
        input=[chunk.page_content for chunk in chunks],
        model=EMBEDDING_MODEL
    )
    
    embeddings = [entry.embedding for entry in embeddings_response.data]
    
    embeddings_with_metadata = join_embeddings_chunks(chunks, embeddings)
    
    return embeddings_with_metadata

# def embeddings_from_chunks(chunks):
#     print("Embedding Documents...")
#     BATCH_SIZE = 600
#     embeddings_with_metadata = []

#     for i in range(0, len(chunks), BATCH_SIZE):
#         batch = chunks[i:i + BATCH_SIZE]
#         print(f"Processing batch {i} - {i + len(batch)}. Total chunks: {len(batch)}")
#         total_tokens = sum(token_counter(chunk.page_content) for chunk in batch)
#         print(f"Total tokens in batch: {total_tokens}")
        
#         try:
#             embeddings_response = openai_client.embeddings.create(
#                 input=[chunk.page_content for chunk in batch],
#                 model=EMBEDDING_MODEL
#             )
#             embeddings = [entry.embedding for entry in embeddings_response.data]
#             embeddings_with_metadata.extend(join_embeddings_chunks(batch, embeddings))
#         except openai.RateLimitError as e:
#             print(f"Rate limit exceeded for batch {i}. Retrying in 65 seconds...")
#             time.sleep(65)
#             continue
#     return embeddings_with_metadata


def main():
    """
    Proceso principal que:
    1. Lee documentos PDF desde una ruta específica.
    2. Extrae su texto y metadata asociada.
    3. Divide los textos en fragmentos manejables (chunks).
    4. Genera embeddings para los fragmentos y los combina con su metadata.
    5. Guarda los embeddings generados en un archivo JSON.
    
    """
    
    doc_paths = os.listdir(DATA_PATH)
    doc_paths = [f"{DATA_PATH}/{doc}" for doc in doc_paths]

    print("Reading documents...")

    corpus_texts = [entries_from_path(path) for path in doc_paths]
    
    chunks = text_splitter.create_documents(
        texts=[entry["text"] for entry in corpus_texts],
        metadatas=[entry["metadata"] for entry in corpus_texts]
    )
    
    embed_entries = embeddings_from_chunks(chunks)
    with open("embeddings_v0.json", "w", encoding=ENCODING_FORMAT) as f:
        json.dump(embed_entries, f)


if __name__ == "__main__":
    main()

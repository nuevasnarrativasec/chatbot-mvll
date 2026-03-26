from openai import OpenAI
from dotenv import load_dotenv
from chatbot.backend.create_embeddings import ENCODING_FORMAT
from pinecone import Pinecone, PodSpec
import os
import itertools
import json

load_dotenv()
BATCH_SIZE = 10
openai_client = OpenAI(api_key = os.getenv('OPENAI_API_KEY'))
INDEX_NAME = "vo-articles"
# INDEX_NAME = "vo-articles-v2"
EMBEDDING_MODEL = "text-embedding-3-small"


def batches_generator(vectors, batch_size):
    iterable_vectors = iter(vectors)
    
    batch = tuple(itertools.islice(iterable_vectors, batch_size))
    
    while batch:
        yield batch
        
        batch = tuple(itertools.islice(iterable_vectors, batch_size))


def main():
    print("Loading Vectors")

    with open("embeddings.json", "r", encoding=ENCODING_FORMAT) as f:
        vectors = json.load(f)

    print("Initializing Pinecone client")
    pinecone_client = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

    # print("Creating Index (if it's necessary)")

    # # Si el índice ya existe, elimínalo antes de crearlo.
    # if INDEX_NAME in pinecone_client.list_indexes().names():
    #     pinecone_client.delete_index(INDEX_NAME)
    
    # # Crea un nuevo índice con las especificaciones dadas.
    # pinecone_client.create_index(
    #     name=INDEX_NAME,  # Nombre del índice.
    #     dimension=1536,  # Dimensión de los vectores de embeddings.
    #     metric="dotproduct",  # Métrica de similitud para búsquedas vectoriales.
    #     spec=PodSpec(
    #         environment="gcp-starter"  # Configuración del entorno (en este caso, un nivel básico en GCP).
    #     )
    # )

    # pinecone_client.create_index(
    #     name=INDEX_NAME,
    #     dimension=1536,
    #     metric="dotproduct",
    #     spec=PodSpec(
    #         environment="aws-us-east1",
    #         pod_type="s1"
    #     )
    # )
    
    index = pinecone_client.Index(INDEX_NAME)

    print("Upserting Vectors")
    i = 0
    for vectors_batches in batches_generator(vectors, BATCH_SIZE):
        i += 1
        print("Número de iteración: ",i)
        # Entró aquí 53 veces
        # chunk_size = 600 | overlapping = 200 --> Entró aquí 175 veces
        index.upsert(
            vectors=list(vectors_batches)
        )

if __name__ == '__main__':
    # Protege la ejecución directa del script y llama a la función principal.
    main()
    print("Fin del proceso")

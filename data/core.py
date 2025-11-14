import chromadb
from chromadb.utils import embedding_functions
import pandas as pd

path_database = "./data.chromadb"
client = chromadb.PersistentClient(path_database)
embf = embedding_functions.SentenceTransformerEmbeddingFunction(
    "dangvantuan/vietnamese-document-embedding", trust_remote_code=True
)
collections = client.get_or_create_collection(name="production", embedding_function=embf)


def add_product_to_production(dataframe, production_id, collection):
    content = f"{dataframe['tên']} {dataframe['thông tin chi tiết']}"
    
    collection.add(
        ids=[str(production_id)],
        documents=[content],
        metadatas=[{
            "title": dataframe["tên"],
            "price": dataframe["giá"],
            "url": dataframe["url"]
        }]
    )

try:
    df = pd.read_csv("/home/big/Projects/rag-mobile/data/hoangha.csv")
except Exception as e:
    raise e

for index, product in df.iterrows():
    add_product_to_production(product, index, collections)



import os
import subprocess
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
PINECONE_INDEX = os.getenv('PINECONE_INDEX')
dimension = 768

EMBEDDING_MODEL_REPO = "sentence-transformers/all-mpnet-base-v2"

tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_REPO)
model = AutoModel.from_pretrained(EMBEDDING_MODEL_REPO)

def create_pinecone_collection(pc, index_name):
    # try:
    #     print(f"Creating 768-dimensional index called '{index_name}'...")
    #     pc.create_index(index_name, dimension=dimension, spec=ServerlessSpec(
    #         cloud="aws",
    #         region="us-east-1"
    #     ))
    #     print("Success")
    # except Exception as e:
    #     if hasattr(e, "status_code") and e.status_code == 409:
    #         print(f"Index '{index_name}' already exists. Continuing without creating a new index.")
    #     else:
    #         print(f"Failed to create index '{index_name}': {e}")
    #         raise
    
    print("Checking Pinecone for active indexes...")
    active_indexes = pc.list_indexes()
    print("Active indexes:")
    print(active_indexes)
    print(f"Getting description for '{index_name}'...")
    index_description = pc.describe_index(index_name)
    print("Description:")
    print(index_description)

    print(f"Getting '{index_name}' as object...")
    pinecone_index = pc.Index(index_name)
    print("Success")

    return pinecone_index

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_embeddings(sentence):
    sentences = [sentence]
    encoded_input = tokenizer(sentences, padding='max_length', truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings.tolist()[0]

def insert_embedding(pinecone_index, id_path, text):
    print("Upserting vectors...")
    vectors = [(text[:512], get_embeddings(text), {"file_path": id_path})]
    try:
        upsert_response = pinecone_index.upsert(vectors=vectors)
        print("Success")
    except Exception as e:
        print(f"Failed to upsert vectors: {e}")
        raise

def main():
    try:
        print("Initializing Pinecone connection...")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        print("Pinecone initialized")

        collection = create_pinecone_collection(pc, PINECONE_INDEX)
        print("Pinecone index is up, collection created")

        doc_dir = '/home/cdsw/data'
        for file in Path(doc_dir).glob('**/*.txt'):
            with open(file, "r") as f:
                print("Generating embeddings for: %s" % file.name)
                text = f.read()
                insert_embedding(collection, os.path.abspath(file), text)
        print('Finished loading Knowledge Base embeddings into Pinecone')

    except Exception as e:
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()
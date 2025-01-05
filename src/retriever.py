import torchvision
torchvision.disable_beta_transforms_warning()

import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import json

class Retriever_SBERT:
    def __init__(self):
        load_dotenv('./config.env')
        sbert_embeddings_path = os.getenv('sbert_embeddings_path')
        metu_dataset_path = os.getenv('dataset_path')

        # Load the dataset
        with open(metu_dataset_path, 'r', encoding='utf-8') as file:
            self.dataset = json.load(file)

        # Load embeddings and convert to correct format
        sbert_embeddings_df = pd.read_csv(sbert_embeddings_path)
        self.docnos = sbert_embeddings_df['URL'].values
        embeddings = sbert_embeddings_df.drop(columns=['URL']).values.astype('float32').copy()
        
        # Normalize embeddings manually using numpy
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms  # L2 normalization

        # Initialize FAISS index
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        # Initialize the SentenceTransformer model
        model_name = 'distiluse-base-multilingual-cased-v1'
        self.model = SentenceTransformer(f'sentence-transformers/{model_name}')

    def retrieve(self, query, top_k=10):  # Changed default from 5 to 10
        # Generate embedding and normalize
        query_embedding = self.model.encode([query]).astype('float32')
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Perform the search
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Get the documents
        retrieved_docs = []
        for i, idx in enumerate(indices[0]):
            url = self.docnos[idx]
            # Find the corresponding document in the dataset
            doc = next((item for item in self.dataset if item["URL"] == url), None)
            if (doc):
                retrieved_docs.append({
                    "url": url,
                    "content": doc["content"],
                    "score": float(distances[0][i])
                })
        
        return retrieved_docs


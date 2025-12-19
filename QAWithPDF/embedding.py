from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import sys
from QAWithPDF.exception import customexception
import logging

def download_gemini_embedding(model, document):
    """
    Downloads and initializes a Local Embedding model (HuggingFace) 
    to avoid Gemini API rate limits.
    Returns a VectorStoreIndex query engine.
    """
    try:
        logging.info("Initializing Local HuggingFace Embedding...")
        
        # ---------------------------------------------------------
        # CORRECT FIX: Use a real Embedding Model here.
        # DO NOT use "gemini-1.5-flash" here. That is for the LLM, not embeddings.
        # "BAAI/bge-small-en-v1.5" is the standard free local model.
        # ---------------------------------------------------------
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        
        Settings.llm = model
        Settings.embed_model = embed_model
        Settings.chunk_size = 800
        Settings.chunk_overlap = 20

        logging.info("Creating VectorStoreIndex locally...")
        index = VectorStoreIndex.from_documents(document)

        logging.info("Preparing query engine...")
        
        # 'compact' mode bundles text to reduce LLM calls.
        # 'similarity_top_k=2' retrieves fewer chunks to save tokens.
        query_engine = index.as_query_engine(
            response_mode="compact", 
            similarity_top_k=2
        )
        return query_engine

    except Exception as e:
        raise customexception(e, sys)

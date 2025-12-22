import os
import sys
from dotenv import load_dotenv
import google.generativeai as genai
from llama_index.llms.gemini import Gemini
from QAWithPDF.exception import customexception
from logger import logging

# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables or .env file.")

# Configure SDK
genai.configure(api_key=GOOGLE_API_KEY)

def load_model():
    """
    Loads the Gemini 1.5 Flash model for Answer Generation (LLM).
    """
    try:
        logging.info("Loading Gemini 1.5 Flash model...")
        
        
    
        model = Gemini(
            model_name="models/gemini-flash-lite-latest", 
            temperature=0.2
        )
        
        logging.info("Gemini model loaded successfully.")
        return model

    except Exception as e:
        # Check for common 404 errors usually caused by old library versions
        if "404" in str(e):
            logging.error("Model 404 Error: Try running 'pip install -U llama-index-llms-gemini google-generativeai'")
        raise customexception(e, sys)

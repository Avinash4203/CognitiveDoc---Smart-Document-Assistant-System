import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("Error: GOOGLE_API_KEY not found in .env")
else:
    genai.configure(api_key=api_key)
    print("Checking available models...")
    try:
        found = False
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"- {m.name}")
                if "flash" in m.name:
                    found = True
        
        if not found:
            print("\n❌ ALERT: No 'Flash' model found. Your API Key might be old.")
            print("Try creating a NEW key at: https://aistudio.google.com/")
            
    except Exception as e:
        print(f"Network/API Error: {e}")

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

# Add parent directory to path to import from main
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import *

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use temporary directory specific to Vercel
TEMP_DIR = "/tmp/pdf_comparisons"
os.makedirs(TEMP_DIR, exist_ok=True)

# Import all routes from main.py
app.include_router(app)

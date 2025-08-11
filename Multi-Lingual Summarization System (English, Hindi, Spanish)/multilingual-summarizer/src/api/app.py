"""
FastAPI Backend for Multilingual Summarization System

This module provides REST API endpoints for the summarization service.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from inference import MultilingualSummarizer
    from ingest import load_text_file
except ImportError:
    # Fallback for when modules aren't available
    MultilingualSummarizer = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Multilingual Summarization API",
    description="API for multilingual abstractive text summarization",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global summarizer instance
summarizer = None

# Pydantic models
class SummarizeRequest(BaseModel):
    text: str
    language: Optional[str] = None
    max_length: Optional[int] = None
    use_hierarchical: bool = False

class BatchSummarizeRequest(BaseModel):
    texts: List[str]
    language: Optional[str] = None
    max_length: Optional[int] = None

class SummarizeResponse(BaseModel):
    summary: str
    language: str
    original_length: int
    summary_length: int
    compression_ratio: float
    processing_time: Optional[float] = None

class BatchSummarizeResponse(BaseModel):
    summaries: List[SummarizeResponse]
    total_documents: int

@app.on_event("startup")
async def startup_event():
    """Initialize the summarizer on startup."""
    global summarizer
    try:
        if MultilingualSummarizer:
            logger.info("Loading multilingual summarizer...")
            summarizer = MultilingualSummarizer()
            logger.info("Summarizer loaded successfully!")
        else:
            logger.warning("Summarizer not available - running in demo mode")
    except Exception as e:
        logger.error(f"Failed to load summarizer: {str(e)}")
        summarizer = None

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Multilingual Summarization API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": summarizer is not None
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": summarizer is not None
    }

@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_text(request: SummarizeRequest):
    """Summarize a single text document."""
    try:
        if not summarizer:
            # Demo mode - return mock response
            return SummarizeResponse(
                summary=f"This is a demo summary of the provided text in {request.language or 'detected language'}. The actual summarization model is not loaded.",
                language=request.language or "en",
                original_length=len(request.text.split()),
                summary_length=20,
                compression_ratio=0.1
            )
        
        # Use hierarchical summarization if requested
        if request.use_hierarchical:
            result = summarizer.summarize_hierarchical(
                request.text, 
                request.language
            )
        else:
            result = summarizer.summarize_single(
                request.text, 
                request.language, 
                request.max_length
            )
        
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        return SummarizeResponse(**result)
        
    except Exception as e:
        logger.error(f"Error in summarization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize/batch", response_model=BatchSummarizeResponse)
async def batch_summarize(request: BatchSummarizeRequest):
    """Summarize multiple text documents."""
    try:
        if not summarizer:
            # Demo mode - return mock responses
            summaries = []
            for i, text in enumerate(request.texts):
                summaries.append(SummarizeResponse(
                    summary=f"Demo summary {i+1} of the provided text in {request.language or 'detected language'}.",
                    language=request.language or "en",
                    original_length=len(text.split()),
                    summary_length=15,
                    compression_ratio=0.1
                ))
            
            return BatchSummarizeResponse(
                summaries=summaries,
                total_documents=len(request.texts)
            )
        
        # Process each text
        summaries = []
        for text in request.texts:
            result = summarizer.summarize_single(
                text, 
                request.language, 
                request.max_length
            )
            
            if 'error' not in result:
                summaries.append(SummarizeResponse(**result))
        
        return BatchSummarizeResponse(
            summaries=summaries,
            total_documents=len(request.texts)
        )
        
    except Exception as e:
        logger.error(f"Error in batch summarization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize/multi-document")
async def multi_document_summarize(request: BatchSummarizeRequest):
    """Summarize multiple documents as a single coherent summary."""
    try:
        if not summarizer:
            return {
                "summary": f"This is a demo multi-document summary combining {len(request.texts)} documents in {request.language or 'detected language'}.",
                "language": request.language or "en",
                "num_documents": len(request.texts),
                "total_length": sum(len(text.split()) for text in request.texts)
            }
        
        result = summarizer.summarize_multiple(request.texts, request.language)
        
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        return result
        
    except Exception as e:
        logger.error(f"Error in multi-document summarization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize/file")
async def summarize_file(file: UploadFile = File(...), language: Optional[str] = None):
    """Summarize text from an uploaded file."""
    try:
        # Read file content
        content = await file.read()
        text = content.decode('utf-8')
        
        if not summarizer:
            return {
                "summary": f"Demo summary of uploaded file '{file.filename}' in {language or 'detected language'}.",
                "language": language or "en",
                "filename": file.filename,
                "file_size": len(content)
            }
        
        result = summarizer.summarize_single(text, language)
        result['filename'] = file.filename
        result['file_size'] = len(content)
        
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        return result
        
    except Exception as e:
        logger.error(f"Error in file summarization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/languages")
async def get_supported_languages():
    """Get list of supported languages."""
    return {
        "supported_languages": [
            {"code": "en", "name": "English"},
            {"code": "hi", "name": "Hindi"},
            {"code": "es", "name": "Spanish"}
        ]
    }

@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model."""
    if not summarizer:
        return {
            "model_loaded": False,
            "message": "No model loaded - running in demo mode"
        }
    
    return {
        "model_loaded": True,
        "model_name": summarizer.model_name,
        "device": summarizer.device,
        "generation_config": summarizer.generation_config
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



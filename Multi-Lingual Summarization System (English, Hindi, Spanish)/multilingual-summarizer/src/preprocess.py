"""
Data Preprocessing and Tokenization Module

This module contains functions for language detection, normalization, 
tokenization, and chunking for multilingual text summarization.
"""

import re
import unicodedata
from typing import List, Dict, Tuple, Optional
import langdetect
from transformers import AutoTokenizer


class TextPreprocessor:
    """Text preprocessing pipeline for multilingual summarization."""
    
    def __init__(self, model_name: str = "google/mt5-small"):
        """Initialize preprocessor with tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.supported_languages = {'en': 'english', 'hi': 'hindi', 'es': 'spanish'}
    
    def detect_language(self, text: str) -> str:
        """Detect language of input text."""
        try:
            detected = langdetect.detect(text)
            # Map to supported languages
            if detected in self.supported_languages:
                return detected
            else:
                return 'en'  # Default to English
        except:
            return 'en'  # Default to English if detection fails
    
    def normalize_text(self, text: str, language: str = 'en') -> str:
        """Normalize text based on language."""
        # Unicode normalization
        text = unicodedata.normalize('NFKC', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Language-specific normalization
        if language == 'hi':
            # Devanagari-specific normalization
            text = re.sub(r'[\u0900-\u097F]\s+(?=[\u0900-\u097F])', '', text)
        
        # General cleanup
        text = text.strip()
        
        return text
    
    def chunk_text(self, text: str, max_length: int = 512, overlap: int = 128) -> List[str]:
        """Chunk text into smaller segments with overlap."""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) <= max_length:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(tokens):
            end = min(start + max_length, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)
            
            if end >= len(tokens):
                break
                
            start = end - overlap
        
        return chunks
    
    def preprocess_document(self, text: str, language: Optional[str] = None) -> Dict:
        """Complete preprocessing pipeline for a document."""
        # Detect language if not provided
        if language is None:
            language = self.detect_language(text)
        
        # Normalize text
        normalized_text = self.normalize_text(text, language)
        
        # Chunk if necessary
        chunks = self.chunk_text(normalized_text)
        
        return {
            'original_text': text,
            'normalized_text': normalized_text,
            'language': language,
            'chunks': chunks,
            'num_chunks': len(chunks)
        }
    
    def prepare_training_data(self, articles: List[str], summaries: List[str], 
                            languages: List[str]) -> List[Dict]:
        """Prepare data for training."""
        training_data = []
        
        for article, summary, lang in zip(articles, summaries, languages):
            # Preprocess article
            processed_article = self.preprocess_document(article, lang)
            
            # Normalize summary
            normalized_summary = self.normalize_text(summary, lang)
            
            training_data.append({
                'article': processed_article['normalized_text'],
                'summary': normalized_summary,
                'language': lang,
                'chunks': processed_article['chunks']
            })
        
        return training_data


def create_hierarchical_chunks(text: str, tokenizer, max_chunk_length: int = 512) -> List[Dict]:
    """Create hierarchical chunks for long document summarization."""
    # Split by paragraphs first
    paragraphs = text.split('\n\n')
    
    chunks = []
    current_chunk = ""
    current_tokens = 0
    
    for para in paragraphs:
        para_tokens = len(tokenizer.encode(para, add_special_tokens=False))
        
        if current_tokens + para_tokens <= max_chunk_length:
            current_chunk += para + "\n\n"
            current_tokens += para_tokens
        else:
            if current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'token_count': current_tokens
                })
            current_chunk = para + "\n\n"
            current_tokens = para_tokens
    
    if current_chunk:
        chunks.append({
            'text': current_chunk.strip(),
            'token_count': current_tokens
        })
    
    return chunks



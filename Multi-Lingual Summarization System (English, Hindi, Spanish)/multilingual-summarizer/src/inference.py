"""
Model Inference Module

This module contains functions for running inference with the trained 
multilingual summarization model.
"""

import os
import sys
import logging
from typing import List, Dict, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Resolve TextPreprocessor import for both module and script executions
try:
    # When run as a module: python -m src.inference
    from .preprocess import TextPreprocessor
except ImportError:
    try:
        # When cwd is project root
        from src.preprocess import TextPreprocessor
    except ImportError:
        # When run directly: python src/inference.py
        sys.path.append(os.path.dirname(__file__))
        from preprocess import TextPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultilingualSummarizer:
    """Multilingual abstractive summarization model."""
    
    def __init__(self, model_name: str = "google/mt5-small", device: str = "auto"):
        """Initialize the summarizer with model and tokenizer."""
        self.model_name = model_name
        
        # Set device with CUDA/MPS/CPU preference
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        logger.info(f"Loading model {model_name} on {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize preprocessor
        self.preprocessor = TextPreprocessor(model_name)
        
        # Generation parameters
        self.generation_config = {
            'max_length': 150,
            'min_length': 30,
            'num_beams': 4,
            'length_penalty': 2.0,
            'early_stopping': True,
            'no_repeat_ngram_size': 3
        }
    
    def summarize_single(self, text: str, language: Optional[str] = None, 
                         max_length: Optional[int] = None) -> Dict:
        """Summarize a single document."""
        try:
            # Preprocess the text
            processed = self.preprocessor.preprocess_document(text, language)
            
            # Use detected language if not provided
            if language is None:
                language = processed['language']
            
            # Prepare input for model
            input_text = f"summarize {language}: {processed['normalized_text']}"
            
            # Tokenize
            inputs = self.tokenizer(
                input_text,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate summary
            generation_config = self.generation_config.copy()
            if max_length:
                generation_config['max_length'] = max_length
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    **generation_config
                )
            
            # Decode summary
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return {
                'summary': summary,
                'language': language,
                'original_length': len(text.split()),
                'summary_length': len(summary.split()),
                'compression_ratio': len(summary.split()) / max(1, len(text.split()))
            }
            
        except Exception as e:
            logger.error(f"Error in summarization: {str(e)}")
            return {
                'summary': "Error: Could not generate summary",
                'language': language or 'unknown',
                'error': str(e)
            }
    
    def summarize_hierarchical(self, text: str, language: Optional[str] = None) -> Dict:
        """Summarize long documents using hierarchical approach."""
        try:
            # Preprocess and chunk the text
            processed = self.preprocessor.preprocess_document(text, language)
            
            if processed['num_chunks'] == 1:
                # Single chunk, use regular summarization
                return self.summarize_single(text, language)
            
            # Summarize each chunk
            chunk_summaries = []
            for chunk in processed['chunks']:
                chunk_result = self.summarize_single(chunk, processed['language'])
                chunk_summaries.append(chunk_result['summary'])
            
            # Combine chunk summaries
            combined_summary = " ".join(chunk_summaries)
            
            # Final summarization of combined summaries
            final_result = self.summarize_single(combined_summary, processed['language'])
            
            return {
                'summary': final_result['summary'],
                'language': processed['language'],
                'num_chunks': processed['num_chunks'],
                'chunk_summaries': chunk_summaries,
                'original_length': len(text.split()),
                'summary_length': len(final_result['summary'].split()),
                'compression_ratio': len(final_result['summary'].split()) / max(1, len(text.split()))
            }
            
        except Exception as e:
            logger.error(f"Error in hierarchical summarization: {str(e)}")
            return {
                'summary': "Error: Could not generate hierarchical summary",
                'language': language or 'unknown',
                'error': str(e)
            }
    
    def summarize_multiple(self, texts: List[str], language: Optional[str] = None) -> Dict:
        """Summarize multiple documents."""
        try:
            # Combine all texts
            combined_text = "\n\n".join(texts)
            
            # Use hierarchical summarization for multiple documents
            result = self.summarize_hierarchical(combined_text, language)
            result['num_documents'] = len(texts)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in multi-document summarization: {str(e)}")
            return {
                'summary': "Error: Could not generate multi-document summary",
                'language': language or 'unknown',
                'error': str(e)
            }
    
    def update_generation_config(self, **kwargs):
        """Update generation configuration."""
        self.generation_config.update(kwargs)
        logger.info(f"Updated generation config: {kwargs}")


def load_summarizer(model_path: str = "google/mt5-small") -> MultilingualSummarizer:
    """Load and return a summarizer instance."""
    return MultilingualSummarizer(model_path)


def batch_summarize(texts: List[str], summarizer: MultilingualSummarizer, 
                    language: Optional[str] = None) -> List[Dict]:
    """Batch summarization for multiple texts."""
    results = []
    for i, text in enumerate(texts):
        logger.info(f"Summarizing document {i+1}/{len(texts)}")
        result = summarizer.summarize_single(text, language)
        results.append(result)
    return results


if __name__ == "__main__":
    # Minimal CLI for quick testing
    import argparse
    parser = argparse.ArgumentParser(description="Multilingual Summarizer Inference")
    parser.add_argument("--text", type=str, help="Text to summarize")
    parser.add_argument("--file", type=str, help="Path to a text file to summarize")
    parser.add_argument("--language", type=str, default=None, help="Optional language code/hint")
    parser.add_argument("--hierarchical", action="store_true", help="Use hierarchical summarization")
    args = parser.parse_args()

    if not args.text and not args.file:
        print("Provide --text or --file")
        sys.exit(1)

    input_text = args.text
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            input_text = f.read()

    summarizer = load_summarizer()
    if args.hierarchical:
        output = summarizer.summarize_hierarchical(input_text, language=args.language)
    else:
        output = summarizer.summarize_single(input_text, language=args.language)

    print(output["summary"])